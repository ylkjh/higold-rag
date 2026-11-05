# -*- coding: utf-8 -*-
"""悍高知识库问答系统 —— Streamlit 网页版
--------------------------------------------------------------------
功能：读入知识库，浏览器聊天式问答，答案只基于资料原文。
运行：streamlit run app.py
本地密钥从 .env 读取；部署到 Streamlit Cloud 时从 Secrets 读取（代码不变）。

单知识库 + 双重标签设计：
  所有资料统一存入 higold_kb，每块数据带两个标签：
    source   = 来源文件名
    category = 内容大类（人事与考勤 / 行政与后勤 / 财务与内控 ……）
  侧边栏按 category 列出"内容大类"；选大类只改变推荐提问的方向，
  检索始终覆盖全部资料（不缩小搜索范围）。
  能上云端的内容由 .gitignore 白名单决定（云端只有公开资料）。
--------------------------------------------------------------------
"""
import os
import chromadb
import streamlit as st
from dotenv import load_dotenv

from categories import classify, label_of, CATEGORIES, OTHER_KEY, AUTO_PREFIX

BASE = os.path.dirname(__file__)

# ---------- LangSmith 监控配置 ----------
# 先把 .env 加载进环境变量（本地生效）；云端则从 Secrets 读取覆盖
load_dotenv(os.path.join(BASE, ".env"))
try:
    for _k in ("LANGSMITH_API_KEY", "LANGSMITH_TRACING", "LANGSMITH_PROJECT"):
        if _k in st.secrets:
            os.environ[_k] = st.secrets[_k]
except FileNotFoundError:
    pass  # 本地没有 secrets.toml，用 .env 的值

# ---------- 密钥：本地读 .env，云端读 Secrets，两边通吃 ----------
def get_key(name: str) -> str:
    try:
        if name in st.secrets:            # 云端（Streamlit Cloud）
            return st.secrets[name]
    except FileNotFoundError:
        pass                              # 本地没有 secrets.toml，走 .env
    return os.getenv(name)

SILICONFLOW_KEY = get_key("SILICONFLOW_API_KEY")
DEEPSEEK_KEY = get_key("DEEPSEEK_API_KEY")

# ---------- 页面外观 ----------
st.set_page_config(page_title="悍高知识库问答", page_icon="🏢", layout="centered")
st.title("🏢 悍高知识库问答")
st.caption("基于资料的 RAG 问答 · 答案只来自手册原文")

# ---------- 初始化模型与向量库（缓存，只在首次运行时执行） ----------
from openai import OpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai

@st.cache_resource
def load_components():
    # wrap_openai 给客户端装上"监控探头"：每次 API 调用的耗时、token、
    # 完整入参出参都会自动上报 LangSmith（smith.langchain.com）
    embed_client = wrap_openai(OpenAI(api_key=SILICONFLOW_KEY,
                                      base_url="https://api.siliconflow.cn/v1"))
    llm = wrap_openai(OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com"))
    db = chromadb.PersistentClient(path=os.path.join(BASE, "db"))
    return embed_client, llm, db

embed_client, llm, db = load_components()

KB = "higold_kb"   # 唯一知识库


# ---------- 首次启动：自动建库（云端 db 为空时现场重建，只读手边有的资料） ----------
def kb_ready() -> bool:
    """库里已有数据、且数据带类别和公开级别标签 -> 不用重建"""
    if KB not in {c.name for c in db.list_collections()}:
        return False
    col_tmp = db.get_collection(KB)
    if col_tmp.count() == 0:
        return False
    metas = col_tmp.get(limit=1, include=["metadatas"])["metadatas"] or []
    if not metas:
        return False
    meta = metas[0] or {}
    return "category" in meta and "visibility" in meta


def load_docs() -> list:
    """准备建库原料，返回 [(source, category, text), ...]

    本地：docs/ 里有原始 PDF -> 全量读（含内部内容，只在本机）
    云端：PDF 从未上传，只有 docs/public/*.txt（建库时导出的脱敏公开版）
           -> 读这些 txt，云端因此只可能拿到可公开的片段
    """
    out = []
    docs_dir = os.path.join(BASE, "docs")
    pdfs = sorted(f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")) \
        if os.path.isdir(docs_dir) else []
    if pdfs:
        from pypdf import PdfReader
        for name in pdfs:
            text = "\n".join(p.extract_text() or "" for p in
                             PdfReader(os.path.join(docs_dir, name)).pages)
            out.append((name.rsplit(".", 1)[0], classify(name, text), text))
        return out

    pub_dir = os.path.join(docs_dir, "public")
    if os.path.isdir(pub_dir):
        for name in sorted(f for f in os.listdir(pub_dir) if f.endswith(".txt")):
            with open(os.path.join(pub_dir, name), encoding="utf-8") as f:
                text = f.read()
            stem = name[:-4]
            out.append((stem, classify(stem + ".pdf", text), text))
    return out


def ensure_kb():
    if kb_ready():
        return
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    try:
        db.delete_collection(KB)          # 旧版本数据没有类别标签，删掉重建
    except Exception:
        pass
    docs = load_docs()
    if not docs:
        st.error("没有找到任何资料：本地需要 docs/*.pdf，云端需要 docs/public/*.txt")
        return
    col_new = db.get_or_create_collection(KB)
    total = 0
    for stem, cat, text in docs:
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50).split_text(text)
        for i in range(0, len(chunks), 16):
            batch = chunks[i:i + 16]
            res = embed_client.embeddings.create(model="BAAI/bge-m3", input=batch)
            col_new.add(ids=[f"kb_{total + j}" for j in range(len(batch))],
                        documents=batch,
                        embeddings=[d.embedding for d in res.data],
                        metadatas=[{"source": stem, "category": cat,
                                    "visibility": "public"}] * len(batch))
            total += len(batch)
        print(f"[自动建库] {stem} -> {label_of(cat)}: {len(chunks)} 块")

ensure_kb()
col = db.get_collection(KB)


# ---------- 工具函数：类别统计、检索、问答 ----------
@st.cache_data(show_spinner=False)
def category_stats(collection: str, n_docs: int) -> list:
    """统计每个内容大类下有几份资料、几个文本块（n_docs 变化时缓存自动失效）"""
    metas = db.get_collection(collection).get(include=["metadatas"])["metadatas"] or []
    bucket = {}
    for m in metas:
        cat = (m or {}).get("category", OTHER_KEY)
        item = bucket.setdefault(cat, {"files": set(), "chunks": 0})
        item["files"].add((m or {}).get("source", ""))
        item["chunks"] += 1

    def order(k):
        """排序：固定大类在前，自动识别的类别居中，其他垫底"""
        if k in CATEGORIES:
            return (0, list(CATEGORIES).index(k), "")
        if k.startswith(AUTO_PREFIX):
            return (1, 0, label_of(k))
        return (2, 0, "")

    return [(k, label_of(k), len(v["files"]), v["chunks"])
            for k, v in sorted(bucket.items(), key=lambda kv: order(kv[0]))]


SYSTEM_PROMPT = (
    "你是悍高集团的知识库助手。只根据下面提供的手册资料回答问题，"
    "资料里没有的内容就直说'手册中未提及'，不要编造。\n"
    "【手册资料】\n{context}"
)


def retrieve(question: str, k: int = 5) -> str:
    """检索：问题变向量 -> Chroma 全库找最相关的 k 块原文
    注意：检索始终覆盖全部资料，不受侧边栏所选类别限制。"""
    vec = embed_client.embeddings.create(model="BAAI/bge-m3", input=[question])
    res = col.query(query_embeddings=vec.data[0].embedding, n_results=k)
    return "\n----\n".join(res["documents"][0])


# @traceable：每次问答在 LangSmith 生成一条完整追踪记录
# （调用树：检索 -> 拼 Prompt -> LLM，网页上层层展开查看）
@traceable
def ask(question: str):
    """完整 RAG：检索 -> 拼 Prompt -> DeepSeek 流式回答"""
    context = retrieve(question)
    stream = llm.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": question},
        ],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ---------- 侧边栏：内容大类 + 清空对话 ----------
with st.sidebar:
    st.markdown("### 📂 资料类别")
    stats = category_stats(KB, col.count())
    key_of = {label: k for k, label, _f, _c in stats}          # 显示名 -> 内部代号
    options = ["全部资料"] + [label for _k, label, _f, _c in stats]
    picked = st.radio("想了解哪方面？", options, index=0,
                      label_visibility="collapsed")
    CATEGORY = key_of.get(picked) if picked != "全部资料" else None

    # 每个类别下面标注资料份数，让人知道这块有什么
    for _k, label, files, chunks in stats:
        st.caption(f"· {label}：{files} 份资料 / {chunks} 个片段")

    st.caption("提示：类别只影响下方推荐提问的方向，搜索始终覆盖全部资料")
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------- 推荐提问：按所选类别的真实内容现场生成（生成一次后缓存复用） ----------
@st.cache_data(show_spinner="正在根据该类资料生成推荐问题……")
def suggest_questions(collection: str, category, n_docs: int) -> list:
    """从库里抽几段原文，让 DeepSeek 猜'用户最可能问什么'

    category 有值时只从这个大类的片段里抽样 -> 推荐问题围绕该业务方向；
    抽样范围与检索范围互不影响。
    """
    try:
        kw = {"limit": 6}
        if category:                                     # 只从所选类别抽原文
            kw["where"] = {"category": category}
        docs_ = db.get_collection(collection).get(**kw)["documents"]
        if not docs_:                                    # 该类别没数据 -> 退回全库
            docs_ = db.get_collection(collection).get(limit=6)["documents"]
        if not docs_:
            return []
        sample = "\n".join(d[:150] for d in docs_)
        scope = label_of(category) if category else "公司全部资料"
        res = llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": f"你是{scope}方向的业务助手。根据以下资料，"
                            "提出3个用户最可能问的问题。"
                            "每行一个，不要编号，不要解释，直接输出问题。"},
                {"role": "user", "content": sample},
            ])
        qs = [q.strip(" 0123456789.、·-") for q in
              res.choices[0].message.content.splitlines() if q.strip()]
        return qs[:3]
    except Exception:
        return []          # 网络或接口异常时静默降级：不显示推荐，不影响问答


clicked = None
sugs = suggest_questions(KB, CATEGORY, col.count())
if sugs:
    st.caption(f"💡 可以这样问（{picked}）：" if CATEGORY else "💡 可以这样问：")
    row = st.columns(len(sugs))
    for box, question in zip(row, sugs):
        if box.button(question, key=f"sug_{picked}_{question}"):
            clicked = question

# ---------- 聊天界面（会话历史保存在 st.session_state） ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

placeholder = (f"想了解{picked}的哪些规定？" if CATEGORY
               else "输入问题，例如：宿舍管理有什么规定？")

if q := (st.chat_input(placeholder) or clicked):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)

    with st.chat_message("assistant"):
        answer = st.write_stream(ask(q))
    st.session_state.messages.append({"role": "assistant", "content": answer})
