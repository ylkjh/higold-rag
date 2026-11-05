# -*- coding: utf-8 -*-
"""悍高知识库建库脚本：docs/ 里所有 PDF -> 切块 -> 向量化 -> 存入 Chroma
--------------------------------------------------------------------
分类规则（安全默认，白名单制）：
  公开库 higold_public   ：只收录 PUBLIC_FILES 白名单里的 PDF（可上 GitHub/公网）
  内部库 higold_internal ：docs/ 里其余全部 PDF（仅本机，禁止上传公网）

怎么加新资料：
  1. 把 PDF 丢进 docs/ 文件夹
  2. 运行  python build_kb.py     （自动识别、自动归类、整库重建）
  想把某份资料公开：先确认内容可公开，再把文件名加进 PUBLIC_FILES
--------------------------------------------------------------------
"""
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import chromadb

BASE = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE, ".env"))

# ---------- 公开白名单：只有列在这里的文件才会进入公开库（可上公网） ----------
PUBLIC_FILES = {"higold_manual.pdf"}

client = OpenAI(api_key=os.getenv("SILICONFLOW_API_KEY"),
                base_url="https://api.siliconflow.cn/v1")
db = chromadb.PersistentClient(path=os.path.join(BASE, "db"))


def rebuild(collection_name: str, pdf_names: list) -> int:
    """整库重建：先删旧库，再把多份 PDF 的切块全部入库（避免逐份建库互相覆盖）"""
    try:
        db.delete_collection(collection_name)
    except Exception:
        pass
    col = db.get_or_create_collection(collection_name)
    total = 0
    for name in pdf_names:
        path = os.path.join(BASE, "docs", name)
        text = "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
        print(f"[{collection_name}] {name}: 读取 {len(text)} 字符")
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50,
            separators=["\n\n", "\n", "。", "，", " "]).split_text(text)
        for i in range(0, len(chunks), 32):
            batch = chunks[i:i + 32]
            res = client.embeddings.create(model="BAAI/bge-m3", input=batch)
            col.add(ids=[f"{collection_name}_{total + j}" for j in range(len(batch))],
                    documents=batch,
                    embeddings=[d.embedding for d in res.data],
                    metadatas=[{"source": name.rsplit(".", 1)[0]} for _ in batch])
            total += len(batch)
        print(f"[{collection_name}] {name}: {len(chunks)} 块已入库")
    print(f"[{collection_name}] 完成！共 {total} 条记录\n")
    return total


docs_dir = os.path.join(BASE, "docs")
all_pdfs = sorted(f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf"))
public_docs = [f for f in all_pdfs if f in PUBLIC_FILES]
internal_docs = [f for f in all_pdfs if f not in PUBLIC_FILES]

print("公开库将收录：", public_docs)
print("内部库将收录：", internal_docs, "\n")

if public_docs:
    rebuild("higold_public", public_docs)
if internal_docs:
    rebuild("higold_internal", internal_docs)

print("全部建库完成。本地打开 app.py 即可查询新资料。")
