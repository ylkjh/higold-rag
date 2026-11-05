# -*- coding: utf-8 -*-
"""切块与建库：父子块（small-to-big）方案
--------------------------------------------------------------------
为什么要父子块：
  小块（300 字）向量表达集中，检索命中率高，但上下文不完整；
  大块（1200 字）语义完整，但向量被"平均"掉，检索命中率低。
  做法 = 用小块建索引检索，命中后把所属大块交给大模型。

父块怎么存：
  直接写进子块的 metadata["parent"]。
  好处：① 不建第二个集合，避免 Chroma 在没有向量时自动调
        默认 embedding 模型的坑；② 检索一次就把正文带回来，少一次查询。
  代价：同一父块被其下几个子块各存一份（约 4 倍冗余，几百 KB 量级，可忽略）。

构建逻辑同时给 build_kb.py（本地全量）和 app.py（云端公开版）复用，
保证两边切块参数完全一致，否则本地和云端的检索效果会不一样。
--------------------------------------------------------------------
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

KB = "higold_kb"

# 父块：给大模型看的完整上下文
PARENT_SIZE, PARENT_OVERLAP = 1200, 100
# 子块：建索引、做检索的最小单位
CHILD_SIZE, CHILD_OVERLAP = 300, 50

_parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PARENT_SIZE, chunk_overlap=PARENT_OVERLAP,
    separators=["\n\n", "\n", "。", "；", "，", " "])
_child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_SIZE, chunk_overlap=CHILD_OVERLAP,
    separators=["\n\n", "\n", "。", "；", "，", " "])


def split_parent_child(text: str) -> list:
    """把一篇文档切成 [(父块文本, [子块文本, ...]), ...]"""
    out = []
    for parent in _parent_splitter.split_text(text):
        kids = _child_splitter.split_text(parent) or [parent]
        out.append((parent, kids))
    return out


def index_docs(db, items, embed_client, batch_size: int = 16, log=print) -> dict:
    """建库：写子块向量索引，父块正文随子块 metadata 一起存

    items: [(source, category, text, vis_fn)]
           vis_fn(child_text) -> "public" / "internal"，云端可传 lambda c: "public"
    返回：{"children": 子块数, "parents": 父块数}
    """
    try:
        db.delete_collection(KB)
    except Exception:
        pass
    col = db.get_or_create_collection(KB)

    n_child = n_parent = 0
    for source, category, text, vis_fn in items:
        pairs = split_parent_child(text)
        for parent, kids in pairs:
            pid = f"p{n_parent}"
            n_parent += 1
            for i in range(0, len(kids), batch_size):
                batch = kids[i:i + batch_size]
                res = embed_client.embeddings.create(model="BAAI/bge-m3", input=batch)
                col.add(ids=[f"c{n_child + j}" for j in range(len(batch))],
                        documents=batch,
                        embeddings=[d.embedding for d in res.data],
                        metadatas=[{"source": source, "category": category,
                                    "parent_id": pid, "parent": parent,
                                    "visibility": vis_fn(c)}
                                   for c in batch])
                n_child += len(batch)
        log(f"  [建库] {source}: {len(pairs)} 父块 / 子块 {n_child} 累计")
    return {"children": n_child, "parents": n_parent}
