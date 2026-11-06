# -*- coding: utf-8 -*-
"""悍高知识库建库脚本：docs/ 里所有 PDF / docx / xlsx -> 分级 -> 切父子块 -> 向量化 -> 存入 Chroma
--------------------------------------------------------------------------
一次运行做三件事：

  1. 建本地全量库（父子块结构，见 indexing.py）
       子块 300 字：建向量索引 + BM25 索引，负责"检索得准"
       父块 1200 字：命中后交给大模型，负责"上下文完整"
     每块数据带四个标签：
       source     来源文件名
       category   内容大类（人事与考勤 / 行政与后勤 / 财务与内控 ……）
       parent_id  所属父块（small-to-big 回填用）
       visibility public（可公开）/ internal（含实际数据，不可外传）

  2. 按内容自动分级（规则见 visibility.py），粒度是**子块**（300 字）
     制度、流程、规定、职责、审批权限  -> public（可公开）
     报价、单价、客户/供应商名单、联系人、工资表、真实经营数据 -> internal

  3. 导出公开版到 docs/public/*.txt
     **只有这些 txt 会上传 GitHub**，云端就用它们重建公开库。
     原始 PDF 一律不上传——仓库里永远没有公司文件原件。

需要人工干预时改下面两个名单：
  FORCE_INTERNAL  整份锁定为内部（例：内控手册这类受控文件）
  FORCE_PUBLIC    整份强制公开（仅用于确认过不含敏感内容的文件）

怎么加新资料：
  1. 把 PDF / docx / xlsx 丢进 docs/（其他格式先另存为这几种）
  2. 运行  python build_kb.py
  类别与公开范围都会自动判定。
--------------------------------------------------------------------------
"""
import os
import shutil

from dotenv import load_dotenv
from openai import OpenAI
import chromadb

from categories import classify, label_of
from visibility import is_public_block, why_sensitive, mask_public
from indexing import index_docs, split_parent_child, extract_pages, KB

BASE = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE, ".env"))

# ---------- 人工干预名单（填文件名，不含 .pdf 后缀） ----------
FORCE_INTERNAL = {"悍高集团内部控制管理手册"}   # 受控文件，整份不上云端
FORCE_PUBLIC = set()

client = OpenAI(api_key=os.getenv("SILICONFLOW_API_KEY"),
                base_url="https://api.siliconflow.cn/v1")
db = chromadb.PersistentClient(path=os.path.join(BASE, "db"))
PUB_DIR = os.path.join(BASE, "docs", "public")


def merge_chunks(chunks: list) -> str:
    """把相邻子块拼回连续文本，同时去掉块间重叠，避免重复内容"""
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


def rebuild(pdf_names: list, do_index: bool = True) -> None:
    # 清空旧的公开版导出，防止已删除的资料残留
    if os.path.isdir(PUB_DIR):
        shutil.rmtree(PUB_DIR)
    os.makedirs(PUB_DIR, exist_ok=True)

    items, stat, report = [], {}, []
    print("\n========== 内容分级报告（按 300 字子块判定）==========")
    for name in pdf_names:
        stem = name.rsplit(".", 1)[0]
        text, page_marks = extract_pages(os.path.join(BASE, "docs", name))
        if not text.strip():
            print(f"{name:<34} 无文字层（可能是扫描件），跳过")
            continue
        cat = classify(name, text)
        pairs = split_parent_child(text)
        all_kids = [k for _p, kids in pairs for k in kids]

        # ---------- 分级 ----------
        if stem in FORCE_INTERNAL:
            flags, note = [False] * len(all_kids), "整份锁定为内部"
        elif stem in FORCE_PUBLIC:
            flags, note = [True] * len(all_kids), "整份强制公开"
        else:
            flags = [is_public_block(c) for c in all_kids]
            note = "按内容自动分级"

        pub_kids = [c for c, ok in zip(all_kids, flags) if ok]
        n_pub = len(pub_kids)
        print(f"{name:<34} {len(all_kids):>4} 子块 -> 公开 {n_pub:>3} / 内部 {len(all_kids) - n_pub:>3}  "
              f"（{len(pairs)} 父块）[{note}]")
        for c, ok in zip(all_kids, flags):
            if not ok:
                reason = why_sensitive(c)
                if reason:
                    print(f"      例：{reason}")
                break

        # 该文件的公开判定函数（供入库时逐子块打 visibility 标签）
        if stem in FORCE_INTERNAL:
            vis = lambda _c: "internal"
        elif stem in FORCE_PUBLIC:
            vis = lambda _c: "public"
        else:
            vis = lambda c: "public" if is_public_block(c) else "internal"

        items.append((stem, cat, text, vis, page_marks))

        # ---------- 导出公开版（只有这部分会上 GitHub） ----------
        # 导出前做定点脱敏：签署栏里的员工姓名 -> ×××（版面保留，内容不丢）
        if pub_kids:
            with open(os.path.join(PUB_DIR, f"{stem}.txt"), "w", encoding="utf-8") as f:
                f.write(mask_public(merge_chunks(pub_kids)))
        else:
            print("      注意：整份无公开内容，不导出公开版")

        stat[cat] = stat.get(cat, 0) + 1

    print("\n========== 向量化入库 ==========")
    if do_index:
        res = index_docs(db, items, client, log=lambda s: print(s, flush=True))
        print(f"\n本地库：{res['children']} 个子块（建索引）/ {res['parents']} 个父块（喂上下文）")
        print("BM25 索引在查询时由子块文本现建，无需额外存储")
    else:
        print("（--export-only）跳过向量化入库，只更新 docs/public/")
    print("类别汇总：")
    for cat, n in stat.items():
        print(f"  {label_of(cat):<10} {n} 份资料")
    print(f"\n公开版已导出到 docs/public/（{len(os.listdir(PUB_DIR))} 个文件）")
    print("只有这个目录会上传 GitHub，原始 PDF 全部留在本机。")


docs_dir = os.path.join(BASE, "docs")
docs = sorted(f for f in os.listdir(docs_dir)
              if f.lower().endswith((".pdf", ".docx", ".xlsx")))
if not docs:
    raise SystemExit("docs/ 里没有 PDF / docx / xlsx，请先放入资料（其他格式请先另存）")

rebuild(docs, do_index="--export-only" not in __import__("sys").argv)

# 清理旧版遗留集合
for legacy in ("higold_public", "higold_internal"):
    try:
        db.delete_collection(legacy)
        print(f"已清理旧库 {legacy}")
    except Exception:
        pass

print("\n完成。本地打开 app.py 即可查询。")
