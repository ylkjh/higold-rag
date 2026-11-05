# -*- coding: utf-8 -*-
"""访问凭证：用「带 key 的链接」代替密码框

为什么不弹密码框：
    密码要口口相传，谁泄露了查不出来，而且每次访问都得输一遍。
    把随机 key 放进链接（?k=xxxx），点开即进，无感；
    每人一条独立 key，谁用超了、外传了，能定位、能单独吊销。

Secrets 里这样配（一行一个人）：

    ACCESS_KEYS = '''
    张三=7fK2pQ9x
    李四=m3R8sT1v
    '''

访问链接：https://你的应用地址/?k=7fK2pQ9x
删除某一行的 key 即吊销该人的访问权（下次刷新就进不来）。

生成新 key：python auth.py 张三 李四
"""
import secrets
import sys


def parse_keys(raw) -> dict:
    """把 Secrets 里的多行/逗号分隔文本解析成 {key: 姓名}

    支持 "姓名=key"，也支持只写 key（自动命名 成员1、成员2）。
    """
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
    keys = parse_keys(raw)
    return keys.get(str(query_value).strip())


def make_key(n: int = 10) -> str:
    """生成一个足够随机的短 key（64^10 种可能，猜不出来）"""
    return secrets.token_urlsafe(n).replace("-", "x").replace("_", "y")[:14]


if __name__ == "__main__":
    names = sys.argv[1:] or ["成员1"]
    print("\n把下面几行贴进 Secrets 的 ACCESS_KEYS（三引号里）：\n")
    for nm in names:
        print(f"{nm}={make_key()}")
    print("\n访问链接：https://你的应用地址/?k=<上面等号右边的key>")
