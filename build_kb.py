# -*- coding: utf-8 -*-
"""悍高知识库建库脚本：PDF -> 切块 -> 向量化 -> 存入 Chroma
--------------------------------------------------------------------
两套知识库（权限隔离设计）：
  higold_public   ：公开版手册 docs/higold_manual.pdf（整理自公开资料，
                    可随代码上传 GitHub 公开仓库）
  higold_internal ：公司内部控制管理手册（保密级别：内部！只在本机建库，
                    绝对不能上传 GitHub / 公网）
用法：
  python build_kb.py            # 建两套库
  python build_kb.py internal   # 只重建内部版（手册更新后跑这个）
--------------------------------------------------------------------
"""
import sys
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import chromadb

BASE = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE, ".env"))

client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)
db = chromadb.PersistentClient(path=os.path.join(BASE, "db"))


def build(pdf_path: str, collection_name: str):
    """通用建库：读 PDF -> 切块 -> 向量化 -> 入库"""
    text = "\n".join(
        page.extract_text() or "" for page in PdfReader(pdf_path).pages)
    print(f"[{collection_name}] 读取完成：{len(text)} 字符")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", " "],
    ).split_text(text)
    print(f"[{collection_name}] 切分成 {len(chunks)} 个文本块")

    try:
        db.delete_collection(collection_name)
    except Exception:
        pass
    col = db.get_or_create_collection(collection_name)

    for i in range(0, len(chunks), 32):
        batch = chunks[i:i + 32]
        res = client.embeddings.create(model="BAAI/bge-m3", input=batch)
        col.add(
            ids=[f"{collection_name}_{i + j}" for j in range(len(batch))],
            documents=batch,
            embeddings=[d.embedding for d in res.data],
        )
        print(f"  已入库 {min(i + 32, len(chunks))}/{len(chunks)}")
    print(f"[{collection_name}] 完成！共 {col.count()} 条记录\n")


# 公开版：随时可建
build(os.path.join(BASE, "docs", "higold_manual.pdf"), "higold_public")

# 内部版：仅当本地存在真手册时才建（云端没有这个文件，天然隔离）
internal_pdf = os.path.join(BASE, "docs", "悍高集团内部控制管理手册.pdf")
if os.path.exists(internal_pdf):
    print(">>> 检测到内部控制手册（保密级别：内部），仅在本机建库，不用于公网部署")
    build(internal_pdf, "higold_internal")
else:
    print(">>> 未找到内部手册，跳过内部库（公网部署时属正常情况）")
