# -*- coding: utf-8 -*-
"""悍高知识库问答系统 —— Streamlit 网页版
--------------------------------------------------------------------
功能：读入手册知识库，浏览器聊天式问答，答案只基于手册原文。
运行：streamlit run app.py
本地密钥从 .env 读取；部署到 Streamlit Cloud 时从 Secrets 读取（代码不变）。

两套知识库（权限隔离设计）：
  公开版 higold_public   ：整理自公开资料，可随代码部署到公网
  内部版 higold_internal ：公司内部控制手册，仅本地存在（不进 Git、不上公网）
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

BASE = os.path.dirname(__file__)

# ---------- 密钥：本地读 .env，云端读 Secrets，两边通吃 ----------
def get_key(name: str) -> str:
    try:
        if name in st.secrets:            # 云端（Streamlit Cloud）
            return st.secrets[name]
    except FileNotFoundError:
        pass                              # 本地没有 secrets.toml，走 .env
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE, ".env"))
    return os.getenv(name)

SILICONFLOW_KEY = get_key("SILICONFLOW_API_KEY")
DEEPSEEK_KEY = get_key("DEEPSEEK_API_KEY")

# ---------- 页面外观 ----------
st.set_page_config(page_title="悍高知识库问答", page_icon="🏢", layout="centered")
st.title("🏢 悍高知识库问答")
st.caption("基于手册的 RAG 问答 · 答案只来自所选知识库原文")

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

# ---------- 云端首次启动：自动建公开版知识库 ----------
# 公网环境没有 db 文件夹，检测到缺少公开库时，现场重建。
# 云端 docs/ 里只有白名单放行的公开资料（.gitignore 拦截了其余文件），
# 所以这里可以放心读取 docs/ 里的全部 PDF——能到云端的本来就都是公开的。
def ensure_public_kb():
    if "higold_public" not in {c.name for c in db.list_collections()}:
        from pypdf import PdfReader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        docs_dir = os.path.join(BASE, "docs")
        col_new = db.get_or_create_collection("higold_public")
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
                            embeddings=[d.embedding for d in res.data])
                total += len(batch)
            print(f"[云端建库] {name}: {len(chunks)} 块")

ensure_public_kb()

# ---------- 侧边栏：选择知识库（按本机实际存在的库动态显示） ----------
col_names = {c.name for c in db.list_collections()}
choices = {"公开版手册（对外演示）": "higold_public"}
if "higold_internal" in col_names:
    choices["内部控制手册（仅本地）"] = "higold_internal"

with st.sidebar:
    kb_label = st.radio("选择知识库", list(choices.keys()), index=0)
    COLLECTION = choices[kb_label]
    st.info(
        "**公开版**：整理自公开资料，可部署公网\n\n"
        "**内部版**：公司受控文件，仅本机可用，禁止上传公网",
        icon="🔒")
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()

col = db.get_collection(COLLECTION)

SYSTEM_PROMPT = (
    "你是悍高集团的知识库助手。只根据下面提供的手册资料回答问题，"
    "资料里没有的内容就直说'手册中未提及'，不要编造。\n"
    "【手册资料】\n{context}"
)

def retrieve(question: str, k: int = 4) -> str:
    """检索：问题变向量 -> Chroma 找最相关的 k 块原文"""
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

# ---------- 聊天界面（会话历史保存在 st.session_state） ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# ---------- 推荐提问：根据所选知识库的真实内容现场生成（生成一次后缓存复用） ----------
@st.cache_data(show_spinner="正在根据资料生成推荐问题……")
def suggest_questions(collection: str, n_docs: int) -> list:
    """从库里抽几段原文，让 DeepSeek 猜'用户最可能问什么'"""
    try:
        docs_ = db.get_collection(collection).get(limit=6)["documents"]
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
sugs = suggest_questions(COLLECTION, col.count())
if sugs:
    st.caption("💡 可以这样问：")
    row = st.columns(len(sugs))
    for box, question in zip(row, sugs):
        if box.button(question, key=f"sug_{COLLECTION}_{question}"):
            clicked = question

if q := (st.chat_input("输入问题，例如：宿舍管理有什么规定？") or clicked):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)

    with st.chat_message("assistant"):
        answer = st.write_stream(ask(q))
    st.session_state.messages.append({"role": "assistant", "content": answer})
