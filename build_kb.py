# -*- coding: utf-8 -*-
"""悍高知识库建库脚本：docs/ 里所有 PDF -> 切块 -> 向量化 -> 存入 Chroma
--------------------------------------------------------------------
设计（单库 + 来源标签 + 白名单上传）：
  所有 PDF 统一存入 higold_kb，每块数据带 source 标签（= 文件名去 .pdf），
  网页侧边栏按标签自动分组为"资料模块"，检索时可限定模块。

  能不能上公网由白名单决定（配合 .gitignore）：
    PUBLIC_FILES 里列出的文件 -> 会随 git 上传到云端
    其余文件                  -> 仅本机，云端建库时天然缺失

怎么加新资料：
  1. 把 PDF 丢进 docs/ 文件夹
  2. 运行  python build_kb.py
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

# ---------- 公开白名单：只有列在这里的文件才会随 git 上传云端 ----------
PUBLIC_FILES = {"higold_manual.pdf"}

client = OpenAI(api_key=os.getenv("SILICONFLOW_API_KEY"),
                base_url="https://api.siliconflow.cn/v1")
db = chromadb.PersistentClient(path=os.path.join(BASE, "db"))

KB = "higold_kb"   # 唯一知识库


def rebuild(pdf_names: list) -> int:
    """整库重建：先删旧库，再把所有 PDF 的切块带来源标签入库"""
    try:
        db.delete_collection(KB)
    except Exception:
        pass
    col = db.get_or_create_collection(KB)
    total = 0
    for name in pdf_names:
        path = os.path.join(BASE, "docs", name)
        text = "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
        print(f"{name}: 读取 {len(text)} 字符")
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50,
            separators=["\n\n", "\n", "。", "，", " "]).split_text(text)
        for i in range(0, len(chunks), 32):
            batch = chunks[i:i + 32]
            res = client.embeddings.create(model="BAAI/bge-m3", input=batch)
            col.add(ids=[f"kb_{total + j}" for j in range(len(batch))],
                    documents=batch,
                    embeddings=[d.embedding for d in res.data],
                    metadatas=[{"source": name.rsplit(".", 1)[0]}] * len(batch))
            total += len(batch)
        print(f"{name}: {len(chunks)} 块已入库")
    print(f"完成！共 {total} 条记录\n")
    return total


docs_dir = os.path.join(BASE, "docs")
all_pdfs = sorted(f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf"))
public_docs = [f for f in all_pdfs if f in PUBLIC_FILES]
internal_docs = [f for f in all_pdfs if f not in PUBLIC_FILES]

print("将随 git 上云端的资料：", public_docs)
print("仅保留在本机的资料：  ", internal_docs, "\n")

rebuild(all_pdfs)

# 清理旧版双库（已被"单库+来源标签"方案取代，避免占用空间造成混淆）
for legacy in ("higold_public", "higold_internal"):
    try:
        db.delete_collection(legacy)
        print(f"已清理旧库 {legacy}")
    except Exception:
        pass

print("全部建库完成。本地打开 app.py 即可查询。")
