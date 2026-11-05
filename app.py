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
import uuid
import chromadb
import streamlit as st
from dotenv import load_dotenv

import guard
from categories import classify, label_of, CATEGORIES, OTHER_KEY, AUTO_PREFIX
from indexing import KB, index_docs
from retrieval import hybrid_search

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

# ---------- 访问控制：不设登录门，靠"每个 IP 每日配额"防刷 ----------
# 设计取舍：不做账号登录、不发密码、不发链接，谁有网址都能打开，
# 但同一个 IP 每天只能问 DAILY_QUOTA 次，全站每天有总量上限（熔断）。
# 代价是"挡不住换 IP 的人"，好处是零门槛、零维护。
MAX_QUERIES = int(get_key("MAX_QUERIES") or 60)                  # 单会话软上限


# ---------- 防刷配额（三层，服务端计数，刷新页面绕不过） ----------
# 配在 Secrets / .env 里可覆盖；不配用默认值。
DAILY_GLOBAL_LIMIT = int(get_key("DAILY_GLOBAL_LIMIT") or 300)   # 全站每天总次数（熔断线）
DAILY_QUOTA = int(get_key("DAILY_QUOTA") or 30)                  # 每人（或每 IP）每天次数
MIN_INTERVAL_SEC = float(get_key("MIN_INTERVAL_SEC") or 2)       # 同一主体最小提问间隔（秒）
MAX_Q_CHARS = int(get_key("MAX_Q_CHARS") or 300)                 # 单条问题最长字数
guard.init()


def client_ip():
    """取访问者 IP：云端由反向代理放在 X-Forwarded-For 里

    本地开发没有这个头 -> 返回 None，此时只按"人"计数、不做 IP 维度限制，
    不会因为所有人共用 'unknown' 而被误伤。
    """
    try:
        h = st.context.headers
        for k in ("X-Forwarded-For", "X-Real-Ip"):
            v = h.get(k)
            if v:
                return v.split(",")[0].strip()
    except Exception:
        pass
    return None


CLIENT_IP = client_ip()


def visitor_id() -> str:
    """配额计数用的身份：优先真实 IP

    云端取不到 X-Forwarded-For 时退回"会话随机串"，至少能挡住
    单个会话内的连发；配合全站总配额，最坏情况也有上限。
    """
    if CLIENT_IP:
        return f"ip:{CLIENT_IP}"
    sid = st.session_state.get("sid")
    if not sid:
        sid = uuid.uuid4().hex[:12]
        st.session_state["sid"] = sid
    return f"session:{sid}"


WHO = visitor_id()          # 这里不是"谁登录了"，而是"配额算在谁头上"


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


# ---------- 首次启动：自动建库（云端 db 为空时现场重建，只读手边有的资料） ----------
def kb_ready() -> bool:
    """库里已有数据、且带父子块与公开级别标签 -> 不用重建"""
    if KB not in {c.name for c in db.list_collections()}:
        return False
    col_tmp = db.get_collection(KB)
    if col_tmp.count() == 0:
        return False
    metas = col_tmp.get(limit=1, include=["metadatas"])["metadatas"] or []
    if not metas:
        return False
    meta = metas[0] or {}
    # parent 是父子块方案新增的，缺它就说明是旧版数据，必须重建
    return all(k in meta for k in ("category", "visibility", "parent"))


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
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: F401
    docs = load_docs()
    if not docs:
        st.error("没有找到任何资料：本地需要 docs/*.pdf，云端需要 docs/public/*.txt")
        return
    with st.spinner("首次启动，正在建知识库（切块 + 向量化），约 20 秒……"):
        res = index_docs(db, [(s, c, t, (lambda _c: "public")) for s, c, t in docs],
                         embed_client, log=lambda s: print(s, flush=True))
    print(f"[建库完成] {res['children']} 子块 / {res['parents']} 父块")


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
    "你是悍高集团的知识库助手。只能使用下面【手册资料】里的内容作答，"
    "不得使用资料以外的知识，也不得编造。\n"
    "\n"
    "请严格按以下规则回答：\n"
    "1. 资料能回答用户的问题：直接给出答案，并在末尾另起一行标注依据，"
    "格式为「依据：《资料名》」。\n"
    "2. 资料不能回答用户的问题，按下面两步做：\n"
    "   ① 先用一句话明确说明：手册中没有关于「用户所问内容」的规定或说明；\n"
    "   ② 再检查资料里有没有同一业务领域、可以替代的内容：\n"
    "      - 有：另起一段写「不过手册里有相关内容：」，然后用「问：」提出一个"
    "与该主题接近、且资料能回答的问题，用「答：」给出答案，最后标注依据；\n"
    "      - 没有：不要强行关联，直接说手册里也没有相近内容，并建议用户换一个"
    "问法（例如按制度名称或流程名称提问）。\n"
    "3. 不要输出「作为AI」「我无法确定」这类免责话术；不要说「未提及」三个字就结束，"
    "务必按上面的结构给出替代内容或换问法建议。\n"
    "\n"
    "【手册资料】\n{context}"
)


def retrieve(question: str, k: int = 4):
    """混合检索（向量 + BM25 + RRF + rerank）-> 父块上下文

    返回 (context, hits)。hits 用于在回答下方展示检索来源。
    检索始终覆盖全部资料，不受侧边栏所选类别限制。
    """
    try:
        out = hybrid_search(db, question, embed_client, SILICONFLOW_KEY, k=k)
        return out["context"], out["hits"]
    except Exception as e:                      # 混合检索异常时降级为纯向量
        print(f"[检索降级] {e}")
        vec = embed_client.embeddings.create(model="BAAI/bge-m3", input=[question])
        res = col.query(query_embeddings=vec.data[0].embedding, n_results=k,
                        include=["documents", "metadatas"])
        blocks, hits = [], []
        for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
            blocks.append(f"《{meta.get('source', '手册资料')}》\n{doc}")
            hits.append({"source": meta.get("source", ""), "child": doc})
        return "\n----\n".join(blocks), hits


def condense_question(question: str, history: list) -> str:
    """多轮对话第一步：把"那第二条呢"这类指代补全成可独立检索的问题

    聊天界面最容易踩的坑：用户追问"那这个呢"，原句里没有任何实体，
    直接拿去检索必然跑偏。这里用一次轻量 LLM 调用，把历史上下文压进当前问题。
    改写失败（网络/接口异常）就退回原问题，不影响主流程。
    """
    if not history:
        return question
    recent = history[-6:]                       # 最近 3 轮
    convo = "\n".join(
        f"{'用户' if m['role'] == 'user' else '助手'}：{m['content'][:200]}"
        for m in recent)
    try:
        r = llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": "把用户最新的提问改写成一个不依赖对话上下文、可以独立检索的"
                            "完整问题。只输出改写后的问题本身，不要解释、不要加引号。"
                            "如果最新提问本身已经完整，就原样输出。"},
                {"role": "user", "content": f"对话历史：\n{convo}\n\n最新提问：{question}"},
            ],
            max_tokens=120,          # 改写只要一句话，限死输出长度
        )
        out = (r.choices[0].message.content or "").strip()
        return out or question
    except Exception:
        return question


# @traceable：每次问答在 LangSmith 生成一条完整追踪记录
# （调用树：检索 -> 拼 Prompt -> LLM，网页上层层展开查看）
@traceable
def ask(question: str):
    """完整 RAG：问题改写 -> 检索 -> 拼 Prompt（含历史）-> DeepSeek 流式回答"""
    history = st.session_state.messages[:-1]          # 不含刚追加的当前问题
    search_q = condense_question(question, history)
    if search_q != question:
        st.session_state["last_query"] = search_q     # 界面上可核对改写结果
    else:
        st.session_state.pop("last_query", None)
    context, hits = retrieve(search_q)
    st.session_state["last_hits"] = hits              # 供界面展示来源

    msgs = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
    for m in history[-4:]:                            # 带最近 2 轮进生成端
        if m["content"]:
            msgs.append({"role": m["role"], "content": m["content"][:800]})
    msgs.append({"role": "user", "content": question})

    stream = llm.chat.completions.create(
        model="deepseek-chat",
        messages=msgs,
        stream=True,
        max_tokens=800,          # 单次回答上限，避免被诱导写长文烧额度
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
    _s = guard.stats(WHO, DAILY_QUOTA, DAILY_GLOBAL_LIMIT)
    st.caption(f"🌐 公开访问 · 每个 IP 每天 {DAILY_QUOTA} 次")
    st.caption(f"🧮 你今天还可提问 {_s['who_left']} 次")
    st.caption(f"全站今日剩余额度：{_s['global_left']} / {DAILY_GLOBAL_LIMIT} 次")
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
            ],
            max_tokens=200,
        )
        qs = [q.strip(" 0123456789.、·-") for q in
              res.choices[0].message.content.splitlines() if q.strip()]
        return qs[:3]
    except Exception:
        return []          # 网络或接口异常时静默降级：不显示推荐，不影响问答


def looks_not_found(text: str) -> bool:
    """判断这次回答是不是"手册里没有"——只看开头，因为提示词要求先说结论"""
    head = (text or "")[:150]
    return any(k in head for k in ("手册中没有", "手册里没有", "没有关于",
                                   "没有找到", "未收录", "没有相关"))


@st.cache_data(show_spinner="正在找手册里能回答的相关问题……")
def related_questions(question: str, n_docs: int) -> list:
    """用户问的内容手册没有时，挑几个"相近且手册能回答"的问题让他换个方向

    做法：拿原问题检索到的片段当线索，让 DeepSeek 提 3 个
    同业务领域、且这些片段确实能回答的问题。生成一次后缓存，不重复花钱。
    """
    try:
        ctx, _hits = retrieve(question)
        res = llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": "用户在查一份企业制度手册，问了一个手册里没有直接答案的问题。"
                            "请根据下面提供的手册内容，提出3个用户可能也想问、"
                            "而且这些内容确实能回答的问题（与原问题属于同一业务领域）。"
                            "每行一个，不要编号，不要解释，直接输出问题。"},
                {"role": "user", "content": f"用户问：{question}\n\n手册内容：\n{ctx}"},
            ],
            max_tokens=200,
        )
        qs = [q.strip(" 0123456789.、·-") for q in
              res.choices[0].message.content.splitlines() if q.strip()]
        return qs[:3]
    except Exception:
        return []


def render_related():
    """画"换个方向试试"的按钮，返回被点击的问题"""
    rel = st.session_state.get("related")
    if not rel or not rel.get("list"):
        return None
    st.caption("📌 手册里没有你问的内容，这几个问题手册能答，要不要试试：")
    cols = st.columns(len(rel["list"]))
    for box, rq in zip(cols, rel["list"]):
        if box.button(rq, key=f"rel_{rq}"):
            return rq
    return None


# ---------- 聊天界面（会话历史保存在 st.session_state） ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# ---------- 提问引导：有"相关问题"就优先显示它，否则显示常规推荐 ----------
clicked = None
if st.session_state.get("related"):
    clicked = render_related()
else:
    sugs = suggest_questions(KB, CATEGORY, col.count())
    if sugs:
        st.caption(f"💡 可以这样问（{picked}）：" if CATEGORY else "💡 可以这样问：")
        row = st.columns(len(sugs))
        for box, question in zip(row, sugs):
            if box.button(question, key=f"sug_{picked}_{question}"):
                clicked = question

placeholder = (f"想了解{picked}的哪些规定？" if CATEGORY
               else "输入问题，例如：宿舍管理有什么规定？")

if q := (st.chat_input(placeholder) or clicked):
    q = (q or "").strip()

    # ① 提问长度上限：超长问题既烧 token，也是脚本灌水的典型特征
    if len(q) > MAX_Q_CHARS:
        st.warning(f"问题太长了（{len(q)} 字），请压缩到 {MAX_Q_CHARS} 字以内。")
        st.stop()

    # ② 会话内软限制（挡手滑，不是安全防线）
    st.session_state["qcount"] = st.session_state.get("qcount", 0) + 1
    if st.session_state["qcount"] > MAX_QUERIES:
        st.error(f"本次会话提问已达上限（{MAX_QUERIES} 次），请明天再试。")
        st.stop()

    # ③ 服务端三层配额：按人计数（没有身份时按 IP），刷新页面 / 换无痕都绕不过
    gate = guard.check(WHO, CLIENT_IP, DAILY_QUOTA, DAILY_GLOBAL_LIMIT, MIN_INTERVAL_SEC)
    if not gate["ok"]:
        st.warning(gate["reason"])
        st.stop()

    st.session_state.pop("related", None)          # 已经重新提问，清掉上轮的相关问题
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)

    cached = guard.cache_get(q)                    # ④ 同一问题第二次问：不发请求、不花钱
    with st.chat_message("assistant"):
        if cached:
            st.write(cached)
            st.caption("⚡ 命中缓存，本次未调用模型")
            answer = cached
        else:
            guard.consume(WHO, CLIENT_IP)          # 只有真要发请求才记一次配额
            answer = st.write_stream(ask(q))
        rq = st.session_state.get("last_query")
        if rq and rq != q:
            st.caption(f"🔎 追问理解成：{rq}")
        hits = st.session_state.get("last_hits") or []
        if hits:
            srcs = "、".join(dict.fromkeys(f"《{h['source']}》" for h in hits
                                          if h.get("source")))
            if srcs:
                st.caption(f"📚 检索来源：{srcs}（混合检索 + 精排）")
    st.session_state.messages.append({"role": "assistant", "content": answer})
    guard.cache_put(q, answer)                     # 写缓存，下次同问直接返回

    # 手册里没有 -> 生成"相近且手册能回答"的问题，rerun 后显示在回答下方
    if looks_not_found(answer):
        st.session_state["related"] = {"list": related_questions(q, col.count())}
        st.rerun()
