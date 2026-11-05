# -*- coding: utf-8 -*-
"""混合检索：向量 + BM25 + RRF 融合 + rerank 精排 + 父块回填
--------------------------------------------------------------------
为什么要混：
  向量检索懂语义（"几点熄灯"能匹配"作息时间"），但不认精确词
  （制度编号 HG-XZ-ZD-008、专有名词、型号），这类是 BM25 的强项。
  两路各取 25 条，用 RRF 按排名融合，互相补漏。

为什么用 RRF 而不是加权求和：
  向量距离和 BM25 分数不在同一量纲，加权要调参且不稳定；
  RRF 只看排名（score += 1/(60+rank)），无需归一化，工业界通用。

rerank 的作用：
  召回阶段求"全"（25 条），精排阶段求"准"（留 4 条）。
  bge-reranker 用 cross-encoder 把问题和文档拼起来算，精度远高于向量内积。

small-to-big：
  命中的是 300 字子块，但交给大模型的是它所属的 1200 字父块，
  避免"答案被切断"和"上下文不够"。
--------------------------------------------------------------------
"""
import re
import math
from collections import Counter

import requests

from indexing import KB

RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
EMBED_MODEL = "BAAI/bge-m3"

# ==================== 中文分词（零依赖，字符二元组） ====================
# 商务场景用字符 bigram 做 BM25：不用装 jieba，效果好且稳定。
# "宿舍管理" -> 宿舍/舍管/管理，能匹配到"宿舍的管理规定"这类不同分词方式。
_EN = re.compile(r"[a-zA-Z]+|\d+(?:\.\d+)?")
_ZH = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list:
    toks = [m.group().lower() for m in _EN.finditer(text or "")]
    for seg in _ZH.findall(text or ""):
        if len(seg) == 1:
            toks.append(seg)
        else:
            toks.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return toks


# ==================== BM25 ====================
class BM25:
    """标准 BM25：k1 控制词频饱和，b 控制文档长度归一化"""

    def __init__(self, docs: list, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs_tokens = [tokenize(d) for d in docs]
        self.lens = [len(t) for t in self.docs_tokens]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 0
        self.tf = [Counter(t) for t in self.docs_tokens]
        df = Counter()
        for t in self.docs_tokens:
            for term in set(t):
                df[term] += 1
        n = len(docs)
        # idf = ln(1 + (N-df+0.5)/(df+0.5))，保证非负
        self.idf = {term: math.log(1 + (n - c + 0.5) / (c + 0.5))
                    for term, c in df.items()}

    def topn(self, query: str, n: int) -> list:
        q = tokenize(query)
        if not q:
            return []
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            dl = self.lens[i] or 1
            for term in q:
                f = tf.get(term)
                if not f:
                    continue
                s += self.idf.get(term, 0) * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / (self.avg or 1)))
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:n]


# ==================== 索引缓存 ====================
# 库内容变了（count 变化）就重建 BM25 索引
_CACHE = {}


def _index_of(col) -> dict:
    key = (col.name, col.count())
    if key not in _CACHE:
        _CACHE.clear()
        data = col.get(include=["documents", "metadatas"])
        _CACHE[key] = {
            "bm25": BM25(data["documents"]),
            "ids": data["ids"],
            "docs": data["documents"],
            "metas": data["metadatas"],
        }
    return _CACHE[key]


# ==================== rerank ====================
def rerank(question: str, docs: list, api_key: str, top_n: int = 4):
    """调用 bge-reranker 精排。失败时返回 None，由上层降级为 RRF 顺序"""
    if not docs:
        return None
    try:
        r = requests.post(
            RERANK_URL,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": RERANK_MODEL, "query": question,
                  "documents": docs, "top_n": min(top_n, len(docs))},
            timeout=20)
        if r.status_code != 200:
            return None
        return [(x["index"], x["relevance_score"]) for x in r.json().get("results", [])]
    except Exception:
        return None


# ==================== 主入口 ====================
def hybrid_search(db, question: str, embed_client, api_key: str,
                  k: int = 4, recall: int = 25, keep_each: int = 12,
                  use_rerank: bool = True) -> dict:
    """混合检索 -> 父块

    召回策略：向量 top12 与 BM25 top12 取**并集**（最多 24 条），
    再由 rerank 精排取前 k 个父块。

    为什么不用 RRF/加权打分决定顺序（实测结论）：
      "HG-XZ-ZD-008 这份文件规定的是什么" 这类编号查询，BM25 把正确块
      排第 1（24.3 分，第 3 名才 20.7），向量则完全没命中（未进 top12）。
      但查询里的"文件/规定/什么"等通用词让向量 top12 几乎全被 BM25 也命中，
      于是两路都得分、把只被 BM25 命中的精确结果挤到后面。
      任何"按排名融合"的算法都会被这种通用词稀释——所以并集只负责扩大
      召回池，最终排序交给 rerank；rerank 不可用时退回纯向量顺序。

    返回 {"context": 拼好的上下文文本, "hits": [...], "debug": {...}}
    hits 里每项：{"source","category","score","child","parent"}
    """
    col = db.get_collection(KB)
    idx = _index_of(col)

    # ---- ① 向量召回（语义） ----
    vec = embed_client.embeddings.create(model=EMBED_MODEL, input=[question])
    vres = col.query(query_embeddings=[vec.data[0].embedding],
                     n_results=min(recall, col.count()),
                     include=["documents", "metadatas"])
    vec_ids = vres["ids"][0][:keep_each]

    # ---- ② BM25 召回（关键词精确匹配，补向量漏掉的编号/术语） ----
    bm = idx["bm25"].topn(question, recall)
    bm25_ids = [idx["ids"][i] for i, _ in bm]

    # ---- ③ 融合：向量优先，BM25 只补新条目（并集，保证不劣于纯向量） ----
    fused_ids = list(vec_ids)
    for _id in bm25_ids:
        if len(fused_ids) >= keep_each * 2:
            break
        if _id not in fused_ids:
            fused_ids.append(_id)

    doc_map = dict(zip(idx["ids"], idx["docs"]))
    meta_map = dict(zip(idx["ids"], idx["metas"]))
    cand_docs = [doc_map[i] for i in fused_ids if i in doc_map]

    # ---- ④ rerank 精排（多留一倍，避免同一父块的子块占满名额） ----
    scored = None
    if use_rerank:
        scored = rerank(question, cand_docs, api_key, top_n=min(k * 3, len(cand_docs)))
    if scored:
        picked = [(fused_ids[i], s) for i, s in scored]
    else:  # 降级：rerank 不可用时退回纯向量顺序（实测比无精排的混合更稳）
        picked = [(cid, 0.0) for cid in vec_ids[:k * 2]]

    # ---- ⑤ 父块回填（small-to-big），同一父块只取一次，凑满 k 个 ----
    # 父块正文就存在子块 metadata["parent"] 里，无需二次查询
    hits, seen_parent = [], set()
    for cid, score in picked:
        if len(hits) >= k:
            break
        meta = meta_map.get(cid, {})
        pid = meta.get("parent_id")
        if pid in seen_parent:
            continue
        seen_parent.add(pid)
        hits.append({"source": meta.get("source", ""),
                     "category": meta.get("category", ""),
                     "score": round(float(score), 4),
                     "child": doc_map.get(cid, ""),
                     "parent_id": pid,
                     "parent": meta.get("parent") or doc_map.get(cid, "")})

    context = "\n----\n".join(f"《{h['source']}》\n{h['parent']}" for h in hits)
    return {"context": context, "hits": hits,
            "debug": {"vector_hits": len(vec_ids), "bm25_hits": len(bm25_ids),
                      "fused": len(fused_ids), "rerank_used": bool(scored),
                      "returned": len(hits)}}
