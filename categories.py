# -*- coding: utf-8 -*-
"""资料内容自动归类（build_kb.py 和 app.py 共用这个文件）
--------------------------------------------------------------------
作用：给每份 PDF 自动打上"内容大类"标签，网页侧边栏的类别列表就是从
      这里来的——以后往 docs/ 丢新文件，跑一次 build_kb.py，类别自动出现，
      不用改任何代码。

识别方式（纯本地关键词比对，不调用 API、不花钱、非常快）：
  1. 文件名里出现关键词   -> 一个 +8 分（文件名最能说明内容）
  2. 正文前 4000 字出现    -> 每出现一次 +1 分（同一关键词最多计 5 分）
  3. 得分最高的类别胜出，且必须 >= 3 分
  4. 一个都命中不了 -> 从文件名自动派生一个新类别
     （例：安全生产管理办法.pdf -> 自动新增"安全生产"类）

想调整或新增大类：直接改下面 CATEGORIES 字典即可，照格式加一段。
--------------------------------------------------------------------
"""

# 内容大类表：key 是内部代号，值 = (显示名, 关键词列表)
CATEGORIES = {
    "company": ("公司概况", [
        "公司简介", "集团简介", "企业简介", "品牌", "产品线", "生产基地",
        "发展历程", "成立于", "主营", "官网", "售后服务电话", "企业愿景",
    ]),
    "hr": ("人事与考勤", [
        "打卡", "考勤", "上班时间", "下班", "请假", "休假", "加班", "调休",
        "奖惩", "奖励", "处罚", "违纪", "薪酬", "工资", "绩效", "招聘",
        "入职", "离职", "试用期", "转正", "员工守则", "行为规范",
        "迟到", "早退", "旷工", "社保", "培训",
    ]),
    "admin": ("行政与后勤", [
        "宿舍", "住宿", "食堂", "用餐", "就餐", "接待", "来访", "参观",
        "车辆", "用车", "派车", "差旅", "出差", "办公用品", "门禁",
        "安保", "保洁", "会议室", "快递", "名片", "印章", "工装",
    ]),
    "finance": ("财务与内控", [
        "财务", "报销", "费用", "预算", "资金", "付款", "收款", "发票",
        "税务", "会计", "审计", "内控", "内部控制", "审批权限", "采购",
        "招标", "合同", "成本", "资产", "存货", "结算", "账务", "盘点",
    ]),
    "business": ("业务与售后", [
        "代理商", "经销商", "售后", "退换货", "退货", "换货", "保修",
        "维修", "客户投诉", "客诉", "订单", "销售", "发货", "物流",
        "品质", "安装", "加盟",
    ]),
}

AUTO_PREFIX = "auto:"      # 自动派生类别的内部前缀
OTHER_KEY = "other"        # 什么都识别不出来的兜底类别
OTHER_LABEL = "其他资料"

# 从文件名派生新类别时，需要去掉的"噪音词"（先长后短，顺序不能乱）
NOISE_WORDS = [
    "温馨提示", "管理规定", "管理办法", "管理制度", "实施细则", "暂行办法",
    "补充规定", "修订版", "规定", "办法", "制度", "流程", "通知", "细则",
    "手册", "暂行", "实施", "补充", "修订", "关于", "悍高", "集团", "公司",
    "的", "（", "）", "(", ")", "-", "_",
]


def label_of(key: str) -> str:
    """内部代号 -> 给人看的类别名（app.py 侧边栏显示用）"""
    if key in CATEGORIES:
        return CATEGORIES[key][0]
    if key.startswith(AUTO_PREFIX):
        return key[len(AUTO_PREFIX):]
    return OTHER_LABEL


def _score(text: str, keyword: str) -> int:
    """关键词在正文里的得分：每出现一次 1 分，最多 5 分"""
    return min(text.count(keyword), 5)


def classify(filename: str, text: str = "") -> str:
    """给一份 PDF 判定内容大类，返回内部代号（company/hr/auto:xxx……）"""
    stem = filename.rsplit(".", 1)[0]      # 去掉 .pdf
    head = (text or "")[:4000]             # 只看正文前 4000 字，够判断了

    best_key, best_score = OTHER_KEY, 0
    for key, (_label, keywords) in CATEGORIES.items():
        score = 0
        for kw in keywords:
            if kw in stem:                 # 文件名命中：权重最高
                score += 8
            score += _score(head, kw)      # 正文命中：按次数加分
        if score > best_score:             # 严格大于：平手时保留靠前的类别
            best_key, best_score = key, score

    if best_score >= 3:
        return best_key

    # 一个关键词都没命中 -> 从文件名派生新类别，实现"自动添加类别"
    name = stem
    for w in NOISE_WORDS:
        name = name.replace(w, " ")
    parts = [p.strip() for p in name.split() if len(p.strip()) >= 2]
    derived = "".join(parts)[:8]
    return AUTO_PREFIX + derived if derived else OTHER_KEY
