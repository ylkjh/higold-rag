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

import bisect
import os

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


def extract_pages(path: str) -> tuple:
    """按格式抽正文，并保留"页码锚点"

    返回 (全文, marks)。
    marks = [(该页第一个字符在全文中的偏移, 页码), ...]，PDF 按物理页记录；
    docx 没有分页概念（Word 的分页是渲染结果，文件里不存在），
    marks 返回 []，下游页码显示为空。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        parts, marks, pos = [], [], 0
        for i, pg in enumerate(PdfReader(path).pages, start=1):
            t = pg.extract_text() or ""
            marks.append((pos, i))
            parts.append(t)
            pos += len(t) + 1                      # +1 是页与页之间的换行符
        return "\n".join(parts), marks
    if ext == ".docx":
        return _extract_docx(path), []
    if ext == ".xlsx":
        return _extract_xlsx(path), []
    raise ValueError(f"不支持的文件格式：{ext}（目前支持 PDF / docx / xlsx，"
                     f"其他格式请先另存为 docx、xlsx 或 PDF）")


def _extract_docx(path: str) -> str:
    from docx import Document as _Doc
    doc = _Doc(path)
    parts = [p.text for p in doc.paragraphs]
    # 制度文件常把审批权限、标准放表格里，表格内容一并抽出
    parts += ["  ".join(c.text.strip() for c in row.cells)
              for tbl in doc.tables for row in tbl.rows]
    return "\n".join(x for x in parts if x)


def _extract_xlsx(path: str) -> str:
    """Excel：每个工作表逐行读，单元格拼接成一句话一行

    行内的单元格用「｜」分隔，行与行之间换行——这样"岗位｜审批人｜额度"
    这类表结构切块后仍保持列对应关系，检索和阅读都友好。
    空行/全空单元格跳过；工作表名做一行的"节标题"，方便定位内容主题。
    """
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True, data_only=True)
    lines = []
    for ws in wb.worksheets:
        lines.append(f"【{ws.title}】")
        for row in ws.iter_rows(values_only=True):
            cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
            if cells:
                lines.append("｜".join(cells))
    wb.close()
    return "\n".join(lines)


def extract_text(path: str) -> str:
    """只要正文不要页码时的便捷入口（兼容旧调用）"""
    return extract_pages(path)[0]


def page_label(text: str, chunk: str, marks, cursor: int = 0) -> tuple:
    """定位 chunk 在全文中的位置 -> 页码标签，如 "4" 或 "4-5"

    marks 为空（docx / 云端公开版）时返回 ("", 原游标)。
    cursor 是顺序游标：块按文档顺序处理，从上一块之后开始找，
    避免重复文本定位到前面的同名段落。
    返回 (页码标签, 新游标)。
    """
    if not marks:
        return "", cursor
    idx = text.find(chunk, cursor)
    if idx == -1:                                  # 兜底：从头再找一次
        idx = text.find(chunk)
    if idx == -1:                                  # 真找不到（理论不该发生）
        return "", cursor
    starts = [m[0] for m in marks]
    a = bisect.bisect_right(starts, idx) - 1
    b = bisect.bisect_right(starts, idx + max(len(chunk) - 1, 0)) - 1
    pa = marks[max(a, 0)][1]
    pb = marks[max(b, 0)][1]
    label = f"{pa}" if pa == pb else f"{pa}-{pb}"
    return label, idx + 1


def index_docs(db, items, embed_client, batch_size: int = 16, log=print) -> dict:
    """建库：写子块向量索引，父块正文随子块 metadata 一起存

    items: [(source, category, text, vis_fn[, page_marks])]
           vis_fn(child_text) -> "public" / "internal"，云端可传 lambda c: "public"
           page_marks 是 extract_pages 返回的页码锚点，缺省/None 时页码存空串
    返回：{"children": 子块数, "parents": 父块数}
    """
    try:
        db.delete_collection(KB)
    except Exception:
        pass
    col = db.get_or_create_collection(KB)

    n_child = n_parent = 0
    for item in items:
        source, category, text, vis_fn = item[:4]
        marks = item[4] if len(item) > 4 else None
        pairs = split_parent_child(text)
        cursor = 0                            # 页码定位游标：块按文档顺序处理
        for parent, kids in pairs:
            pid = f"p{n_parent}"
            n_parent += 1
            for i in range(0, len(kids), batch_size):
                batch = kids[i:i + batch_size]
                labels = []
                for c in batch:               # 先定位每块的页码，再批量向量化
                    label, cursor = page_label(text, c, marks, cursor)
                    labels.append(label)
                res = embed_client.embeddings.create(model="BAAI/bge-m3", input=batch)
                col.add(ids=[f"c{n_child + j}" for j in range(len(batch))],
                        documents=batch,
                        embeddings=[d.embedding for d in res.data],
                        metadatas=[{"source": source, "category": category,
                                    "parent_id": pid, "parent": parent,
                                    "visibility": vis_fn(c), "page": labels[j]}
                                   for j, c in enumerate(batch)])
                n_child += len(batch)
        log(f"  [建库] {source}: {len(pairs)} 父块 / 子块 {n_child} 累计")
    return {"children": n_child, "parents": n_parent}
