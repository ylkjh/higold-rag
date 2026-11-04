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

@st.cache_resource
def load_components():
    embed_client = OpenAI(api_key=SILICONFLOW_KEY,
                          base_url="https://api.siliconflow.cn/v1")
    llm = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
    db = chromadb.PersistentClient(path=os.path.join(BASE, "db"))
    return embed_client, llm, db

embed_client, llm, db = load_components()

# ---------- 云端首次启动：自动建公开版知识库 ----------
# 公网环境没有 db 文件夹，检测到缺少公开库时，现场从手册 PDF 重建（约 10 秒）
def ensure_public_kb():
    if "higold_public" not in {c.name for c in db.list_collections()}:
        from pypdf import PdfReader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text = "\n".join(p.extract_text() or "" for p in
                         PdfReader(os.path.join(BASE, "docs", "higold_manual.pdf")).pages)
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50).split_text(text)
        col_new = db.get_or_create_collection("higold_public")
        for i in range(0, len(chunks), 16):
            batch = chunks[i:i + 16]
            res = embed_client.embeddings.create(model="BAAI/bge-m3", input=batch)
            col_new.add(ids=[f"kb_{i+j}" for j in range(len(batch))],
                        documents=batch,
                        embeddings=[d.embedding for d in res.data])

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

if q := st.chat_input("输入问题，例如：采购业务的审批流程是怎样的？"):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)

    with st.chat_message("assistant"):
        answer = st.write_stream(ask(q))
    st.session_state.messages.append({"role": "assistant", "content": answer})
