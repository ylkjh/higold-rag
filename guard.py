# -*- coding: utf-8 -*-
"""调用配额守卫：防止公网被别人循环刷 API 额度

为什么不用 session_state 计数：
    session_state 只在一个浏览器会话里有效。刷新页面 / 换无痕窗口 /
    curl 直接构造请求，都会得到全新会话 -> 计数归零。
    要防刷，计数必须落在**服务端**，跟会话无关。

三层限流（都基于服务端 SQLite，重启不清零）：
    ① 全站 24 小时总配额  —— 最终熔断：真被刷了，最多损失这么多
    ② 单 IP 24 小时配额   —— 挡住同一个人/同一台机器
    ③ 单 IP 最小间隔      —— 挡住脚本高频连发（正常人手速达不到）

另附答案缓存：同一个问题第二次问不发请求、不花钱。
"""
import os
import sqlite3
import time
import threading
import re

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".guard")
_DB = os.path.join(_DIR, "usage.db")
_LOCK = threading.Lock()          # 多线程写 SQLite 时串行化，避免 database is locked
WINDOW = 24 * 3600                # 滚动 24 小时


def _conn():
    os.makedirs(_DIR, exist_ok=True)
    c = sqlite3.connect(_DB, timeout=10)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init():
    """建表（幂等，每次启动调用一次即可）"""
    with _LOCK, _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS usage(
                         id INTEGER PRIMARY KEY AUTOINCREMENT,
                         ts REAL NOT NULL,
                         ip TEXT NOT NULL)""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage(ts)")
        c.execute("""CREATE TABLE IF NOT EXISTS cache(
                         qhash TEXT PRIMARY KEY,
                         answer TEXT NOT NULL,
                         ts REAL NOT NULL)""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_cache_ts ON cache(ts)")


def norm(question: str) -> str:
    """问题归一化：去掉空白和标点，避免' 报价单？'和'报价单'算两个问题"""
    return re.sub(r"[\s\u3000，。！？、；：\"'“”‘’（）()\[\]{}]+", "", question or "").lower()


def cache_get(question: str):
    try:
        with _LOCK, _conn() as c:
            row = c.execute("SELECT answer FROM cache WHERE qhash=?",
                            (norm(question),)).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def cache_put(question: str, answer: str):
    if not answer or len(answer) < 20:
        return
    try:
        with _LOCK, _conn() as c:
            c.execute("INSERT OR REPLACE INTO cache(qhash, answer, ts) VALUES(?,?,?)",
                      (norm(question), answer, time.time()))
    except Exception:
        pass


def check(ip, daily_ip: int = 30, daily_global: int = 300,
          min_interval: float = 2.0) -> dict:
    """判断这次提问是否放行（不写记录，放行后要再调 consume）

    返回 {"ok": bool, "reason": str, "ip_left": int|None,
          "global_left": int, "retry_after": int}
    """
    now = time.time()
    since = now - WINDOW
    with _LOCK, _conn() as c:
        c.execute("DELETE FROM usage WHERE ts < ?", (since,))      # 顺手清理过期记录
        g = c.execute("SELECT COUNT(*) FROM usage WHERE ts>=?", (since,)).fetchone()[0]
        n = last = None
        if ip:
            n = c.execute("SELECT COUNT(*) FROM usage WHERE ts>=? AND ip=?",
                          (since, ip)).fetchone()[0]
            last = c.execute("SELECT MAX(ts) FROM usage WHERE ip=?", (ip,)).fetchone()[0]

    res = {"ok": True, "reason": "", "ip_left": None,
           "global_left": max(0, daily_global - g), "retry_after": 0}

    # ① 全站总配额（最终熔断，保护钱包）
    if g >= daily_global:
        res.update(ok=False, global_left=0,
                   reason=f"今日全站调用额度已用完（{daily_global} 次），请明天再试。")
        return res

    if ip:
        # ③ 最小间隔（脚本连发会撞这一条，正常人手速不会）
        if last and now - last < min_interval:
            wait = int(min_interval - (now - last)) + 1
            res.update(ok=False, retry_after=wait,
                       reason=f"提问太快了，请等 {wait} 秒再问。")
            return res
        # ② 单 IP 配额
        res["ip_left"] = max(0, daily_ip - n)
        if n >= daily_ip:
            res.update(ok=False,
                       reason=f"你今天的提问次数已用完（{daily_ip} 次），请明天再试。")
            return res

    return res


def consume(ip):
    """放行后记一笔（只记成功发出去的请求，被拦的不计入）"""
    try:
        with _LOCK, _conn() as c:
            c.execute("INSERT INTO usage(ts, ip) VALUES(?,?)", (time.time(), ip or "unknown"))
    except Exception:
        pass


def stats(daily_ip: int = 30, daily_global: int = 300) -> dict:
    """给侧边栏显示用：今天全站用了多少、这个 IP 用了多少"""
    since = time.time() - WINDOW
    try:
        with _LOCK, _conn() as c:
            g = c.execute("SELECT COUNT(*) FROM usage WHERE ts>=?", (since,)).fetchone()[0]
    except Exception:
        g = 0
    return {"global_used": g, "global_left": max(0, daily_global - g)}
