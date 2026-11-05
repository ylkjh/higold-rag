# -*- coding: utf-8 -*-
"""悍高知识库问答系统 —— Streamlit 网页版
--------------------------------------------------------------------
功能：读入知识库，浏览器聊天式问答，答案只基于资料原文。
运行：streamlit run app.py
本地密钥从 .env 读取；部署到 Streamlit Cloud 时从 Secrets 读取（代码不变）。

单知识库 + 来源标签设计：
  所有资料统一存入 higold_kb，每块带 source 标签（来源文件名）；
  侧边栏按标签自动分组为"资料模块"，选中模块后检索只在该资料内进行。
  能上云端的内容由 .gitignore 白名单决定（云端只有公开资料）。
--------------------------------------------------------------------
"""
import os
import chromadb
import streamlit as st
from dotenv import load_dotenv

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
st.caption("基于资料的 RAG 问答 · 答案只来自所选资料原文")

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

# ---------- 首次启动：自动建库（云端 db 为空时现场重建，只有公开资料可用） ----------
def ensure_kb():
    if KB not in {c.name for c in db.list_collections()}:
        from pypdf import PdfReader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        docs_dir = os.path.join(BASE, "docs")
        col_new = db.get_or_create_collection(KB)
        total = 0
        for name in sorted(f for f in os.listdir(docs_dir)
                           if f.lower().endswith(".pdf")):
            text = "\n".join(p.extract_text() or "" for p in
                             PdfReader(os.path.join(docs_dir, name)).pages)
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50).split_text(text)
            for i in range(0, len(chunks), 16):
                batch = chunks[i:i + 16]
                res = embed_client.embeddings.create(model="BAAI/bge-m3", input=batch)
                col_new.add(ids=[f"kb_{total + j}" for j in range(len(batch))],
                            documents=batch,
                            embeddings=[d.embedding for d in res.data],
                            metadatas=[{"source": name.rsplit(".", 1)[0]}] * len(batch))
                total += len(batch)
            print(f"[自动建库] {name}: {len(chunks)} 块")

ensure_kb()

# ---------- 工具函数：模块（来源资料）列表、检索、问答 ----------
@st.cache_data(show_spinner=False)
def get_sources(collection: str, n_docs: int) -> list:
    """列出库里有哪些资料模块（按入库时打的 source 标签去重）"""
    metas = db.get_collection(collection).get(include=["metadatas"])["metadatas"] or []
    return sorted({(m or {}).get("source", "未分类") for m in metas})

SYSTEM_PROMPT = (
    "你是悍高集团的知识库助手。只根据下面提供的手册资料回答问题，"
    "资料里没有的内容就直说'手册中未提及'，不要编造。\n"
    "【手册资料】\n{context}"
)

def retrieve(question: str, k: int = 4, source=None) -> str:
    """检索：问题变向量 -> Chroma 找最相关的 k 块原文（可限定某个资料模块）"""
    vec = embed_client.embeddings.create(model="BAAI/bge-m3", input=[question])
    kw = {"query_embeddings": vec.data[0].embedding, "n_results": k}
    if source:                                   # 选中模块时只在该资料内检索
        kw["where"] = {"source": source}
    res = col.query(**kw)
    return "\n----\n".join(res["documents"][0])

# @traceable：每次问答在 LangSmith 生成一条完整追踪记录
# （调用树：检索 -> 拼 Prompt -> LLM，网页上层层展开查看）
@traceable
def ask(question: str, source=None):
    """完整 RAG：检索 -> 拼 Prompt -> DeepSeek 流式回答"""
    context = retrieve(question, source=source)
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

# ---------- 侧边栏：资料模块 + 清空对话 ----------
# 模块显示名美化（没有映射的直接用文件名去掉 .pdf）
PRETTY = {"higold_manual": "公司简介（公开）"}

with st.sidebar:
    n_docs = db.get_collection(KB).count()
    sources = get_sources(KB, n_docs)
    src_of = {PRETTY.get(s, s): s for s in sources}       # 显示名 -> 标签名
    cat_label = st.radio("按资料模块提问", ["全部资料"] + list(src_of),
                         index=0,
                         help="选定模块后，检索与推荐问题只来自这份资料")
    CATEGORY = None if cat_label == "全部资料" else src_of[cat_label]

    st.caption("提示：选定模块后，答案只来自该模块对应的资料")
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()

col = db.get_collection(KB)

# ---------- 推荐提问：按所选模块的真实内容现场生成（生成一次后缓存复用） ----------
@st.cache_data(show_spinner="正在根据该模块资料生成推荐问题……")
def suggest_questions(collection: str, category, n_docs: int) -> list:
    """从库里抽几段原文，让 DeepSeek 猜'用户最可能问什么'"""
    try:
        kw = {"limit": 6}
        if category:                                     # 只从所选模块抽原文
            kw["where"] = {"source": category}
        docs_ = db.get_collection(collection).get(**kw)["documents"]
        if not docs_:
            return []
        sample = "\n".join(d[:150] for d in docs_)
        res = llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": "根据以下企业资料，提出3个用户最可能问的问题。"
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
    st.caption(("【" + cat_label + "】") if CATEGORY else "💡 可以这样问：")
    row = st.columns(len(sugs))
    for box, question in zip(row, sugs):
        if box.button(question, key=f"sug_{cat_label}_{question}"):
            clicked = question

# ---------- 聊天界面（会话历史保存在 st.session_state） ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if q := (st.chat_input("输入问题，例如：宿舍管理有什么规定？") or clicked):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)

    with st.chat_message("assistant"):
        answer = st.write_stream(ask(q, CATEGORY))
    st.session_state.messages.append({"role": "assistant", "content": answer})
