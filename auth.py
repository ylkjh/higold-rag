# -*- coding: utf-8 -*-
"""访问控制：公司账号登录（OIDC）+ 邮箱白名单

主方案（B）：
    用 Streamlit 内置的 st.login() 接 Google / 微软（Entra ID）/ Auth0 等
    OIDC 提供方。用户点一次"用公司账号登录"，跳转到公司账号页，
    登录后回到应用，我们校验他的邮箱是否在白名单里。

    好处：身份是真实的，链接转发无效（对方得用自己账号登录），
    谁问了什么、用了多少次都能对上人。

    配置在 Secrets 里（[auth] 段由 Streamlit 自己读取）：
        [auth]
        redirect_uri = "https://你的应用/oauth2callback"
        cookie_secret = "随机字符串"
        client_id = "..."
        client_secret = "..."
        server_metadata_url = "https://login.microsoftonline.com/<租户ID>/v2.0/.well-known/openid-configuration"

        ALLOWED_EMAILS = "zhangsan@higold.com.cn, lisi@higold.com.cn, @higold.com.cn"

    白名单支持两种写法：完整邮箱、@域名（整个域名都放行）。

备用（A，已不推荐）：链接邀请码 ?k=xxx，见 parse_keys/resolve。
"""
import secrets
import sys


# ---------- 邮箱白名单 ----------
def parse_emails(raw) -> list:
    """把多行/逗号/分号分隔的白名单拆成列表（统一小写）"""
    if not raw:
        return []
    txt = str(raw).replace("，", ",").replace("；", ";").replace("\n", ",")
    txt = txt.replace(";", ",").replace(" ", ",")
    return [e.strip().lower() for e in txt.split(",") if e.strip()]


def email_allowed(email, raw) -> bool:
    """邮箱是否在白名单里；支持 @域名 通配整个域

    白名单为空 -> 一律拒绝（宁可进不来，也不能变成公开访问）。
    """
    email = (email or "").strip().lower()
    rules = parse_emails(raw)
    if not email or not rules:
        return False
    for r in rules:
        if r.startswith("@"):
            if email.endswith(r):
                return True
        elif email == r:
            return True
    return False


# ---------- 备用：链接邀请码 ----------
def parse_keys(raw) -> dict:
    """解析 ACCESS_KEYS 为 {key: 姓名}，支持 "姓名=key" 或只写 key"""
    out = {}
    if not raw:
        return out
    items = []
    for line in str(raw).replace("，", ",").replace("；", ";").splitlines():
        items.extend(part for part in line.replace(";", ",").split(",") if part.strip())
    for i, item in enumerate(items, 1):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            name, key = item.split("=", 1)
            name, key = name.strip(), key.strip()
        else:
            name, key = f"成员{i}", item
        if key:
            out[key] = name or f"成员{i}"
    return out


def resolve(query_value, raw):
    """用 URL 里的 k 换取姓名；无效返回 None"""
    if not query_value:
        return None
    return parse_keys(raw).get(str(query_value).strip())


# ---------- 小工具 ----------
def make_key(n: int = 14) -> str:
    """随机串（用于 cookie_secret 或备用邀请码）"""
    return secrets.token_urlsafe(n)


if __name__ == "__main__":
    print("cookie_secret（贴到 Secrets 的 [auth] 段）:")
    print(make_key(32))
    if len(sys.argv) > 1:
        print("\n备用邀请码：")
        for nm in sys.argv[1:]:
            print(f"{nm}={make_key(10)}")
