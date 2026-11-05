# -*- coding: utf-8 -*-
"""悍高知识库建库脚本：docs/ 里所有 PDF -> 分级 -> 切块 -> 向量化 -> 存入 Chroma
--------------------------------------------------------------------------
一次运行做三件事：

  1. 建本地全量库（higold_kb）
     每块数据带三个标签：
       source     来源文件名
       category   内容大类（人事与考勤 / 行政与后勤 / 财务与内控 ……）
       visibility public（可公开）/ internal（含实际数据，不可外传）

  2. 按内容自动分级（规则见 visibility.py）
     制度、流程、规定、职责、审批权限  -> public（可公开）
     报价、单价、客户/供应商名单、联系人、工资表、真实经营数据 -> internal

  3. 导出公开版到 docs/public/*.txt
     **只有这些 txt 会上传 GitHub**，云端就用它们重建公开库。
     原始 PDF 一律不上传——仓库里永远没有公司文件原件。

需要人工干预时改下面两个名单：
  FORCE_INTERNAL  整份锁定为内部（例：内控手册这类受控文件）
  FORCE_PUBLIC    整份强制公开（仅用于确认过不含敏感内容的文件）

怎么加新资料：
  1. 把 PDF 丢进 docs/
  2. 运行  python build_kb.py
  类别与公开范围都会自动判定。
--------------------------------------------------------------------------
"""
import os
import shutil

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import chromadb

from categories import classify, label_of
from visibility import is_public_block, why_sensitive

BASE = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE, ".env"))

# ---------- 人工干预名单（填文件名，不含 .pdf 后缀） ----------
# 整份不公开：这类文件是公司受控文件，或确认含大量实际数据
FORCE_INTERNAL = {"悍高集团内部控制管理手册"}
# 整份强制公开：确认无敏感内容的对外宣传资料
FORCE_PUBLIC = set()

client = OpenAI(api_key=os.getenv("SILICONFLOW_API_KEY"),
                base_url="https://api.siliconflow.cn/v1")
db = chromadb.PersistentClient(path=os.path.join(BASE, "db"))

KB = "higold_kb"
PUB_DIR = os.path.join(BASE, "docs", "public")     # 公开版导出目录

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " "])


def merge_chunks(chunks: list) -> str:
    """把相邻切块拼回连续文本，同时去掉块间重叠的 50 字，避免重复内容"""
    out = []
    for c in chunks:
        if out:
            prev = out[-1]
            for L in range(min(len(prev), len(c), 80), 9, -1):
                if prev[-L:] == c[:L]:
                    c = c[L:]
                    break
        out.append(c)
    return "\n".join(out)


def rebuild(pdf_names: list) -> None:
    try:
        db.delete_collection(KB)
    except Exception:
        pass
    col = db.get_or_create_collection(KB)

    # 清空旧的公开版导出，防止已删除的资料残留
    if os.path.isdir(PUB_DIR):
        shutil.rmtree(PUB_DIR)
    os.makedirs(PUB_DIR, exist_ok=True)

    total = 0
    stat = {}          # 类别 -> 份数
    print("\n========== 内容分级报告 ==========")
    for name in pdf_names:
        stem = name.rsplit(".", 1)[0]
        text = "\n".join(p.extract_text() or "" for p in
                         PdfReader(os.path.join(BASE, "docs", name)).pages)
        cat = classify(name, text)
        chunks = splitter.split_text(text)

        # ---------- 分级：哪些块可以公开 ----------
        if stem in FORCE_INTERNAL:
            flags, note = [False] * len(chunks), "整份锁定为内部"
        elif stem in FORCE_PUBLIC:
            flags, note = [True] * len(chunks), "整份强制公开"
        else:
            flags = [is_public_block(c) for c in chunks]
            note = "按内容自动分级"

        pub_chunks = [c for c, ok in zip(chunks, flags) if ok]
        n_pub = len(pub_chunks)
        print(f"{name:<34} {len(chunks):>4} 块 -> 公开 {n_pub:>3} / 内部 {len(chunks) - n_pub:>3}  [{note}]")
        # 打印 1 条内部块的原因，方便人工复核判定是否合理
        for c, ok in zip(chunks, flags):
            if not ok:
                print(f"      例：{why_sensitive(c)}")
                break

        # ---------- 入库（本地全量，含内部块，供本机查询） ----------
        for i in range(0, len(chunks), 32):
            batch = chunks[i:i + 32]
            res = client.embeddings.create(model="BAAI/bge-m3", input=batch)
            col.add(ids=[f"kb_{total + j}" for j in range(len(batch))],
                    documents=batch,
                    embeddings=[d.embedding for d in res.data],
                    metadatas=[{"source": stem, "category": cat,
                                "visibility": "public" if flags[i + j] else "internal"}
                               for j in range(len(batch))])
            total += len(batch)

        # ---------- 导出公开版（只有这部分会上 GitHub） ----------
        if pub_chunks:
            with open(os.path.join(PUB_DIR, f"{stem}.txt"), "w", encoding="utf-8") as f:
                f.write(merge_chunks(pub_chunks))
        else:
            print(f"      注意：整份无公开内容，不导出公开版")

        stat[cat] = stat.get(cat, 0) + 1

    print(f"\n本地库共 {total} 条记录")
    print("类别汇总：")
    for cat, n in stat.items():
        print(f"  {label_of(cat):<10} {n} 份资料")
    print(f"\n公开版已导出到 docs/public/（{len(os.listdir(PUB_DIR))} 个文件）")
    print("只有这个目录会上传 GitHub，原始 PDF 全部留在本机。")


docs_dir = os.path.join(BASE, "docs")
pdfs = sorted(f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf"))
if not pdfs:
    raise SystemExit("docs/ 里没有 PDF，请先放入资料")

rebuild(pdfs)

# 清理旧版双库（早期方案遗留）
for legacy in ("higold_public", "higold_internal"):
    try:
        db.delete_collection(legacy)
        print(f"已清理旧库 {legacy}")
    except Exception:
        pass

print("\n完成。本地打开 app.py 即可查询。")
