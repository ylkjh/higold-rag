# -*- coding: utf-8 -*-
"""内容分级：判断一段文字是"可公开的制度流程"还是"不能外传的实际数据"
--------------------------------------------------------------------
设计原则（按内容性质分，不按文件类型分）：
  可公开  -> 制度、流程、规范、职责、审批权限、管理要求、操作步骤
  不可公开 -> 报价单、单价、合同金额、客户/供应商名单、联系人、工资表、
             身份证/银行卡/税号、真实经营数据（营收、净利润、成本）

判定方式：把文本按句切开，逐句检查。只要有一个句子命中"实际数据"特征，
这一整块（500 字）就判为内部，宁严勿宽——宁可少公开一段，不可漏出去一条。

为什么"金额阈值"不算敏感：
  "采购金额 50 万元以上需总经理审批" 是制度条款，可以公开；
  "本次采购单价 35.6 元，总价 128,000 元" 是真实交易数据，不能公开。
  区分方式 = 敏感词（报价/单价/成本…）与金额出现在同一句才算数据。
--------------------------------------------------------------------
"""
import re

# ---------- 1. 表单字段类：带冒号 + 具体内容，说明填了真实信息 ----------
FIELD_DATA = re.compile(
    r"(客户|供应商|厂商|联系人|收款人|开户行|账号|税号|纳税人识别号|"
    r"合同编号|订单号|发票号|报价|单价|总价|金额)\s*[:：]\s*\S"
)

# ---------- 2. 明确不能外传的实体/台账 ----------
# 注意：写"身份证号"而不是"身份证"——制度里的"伪造假身份证"是违纪条款，
# 属于可公开内容，不能误判；真实号码由下面的 PERSONAL 正则兜住。
HARD_SENSITIVE = re.compile(
    r"身份证号|身份证复印件|银行卡号|银行账号|开户许可证|营业执照号|"
    r"工资表|薪资明细|社保基数|个人所得税|"
    r"客户名单|供应商名单|通讯录|报价单|价格表|成本明细|"
    r"投标底价|返点|佣金比例"
)

# ---------- 3. 个人隐私：手机号、身份证号、长数字账号 ----------
PERSONAL = re.compile(r"1[3-9]\d{9}|\d{17}[\dXx]|\b\d{16,19}\b")

# ---------- 4. 金额 + 交易语境：真实报价/成本/结算数据 ----------
AMOUNT = re.compile(r"\d[\d,\.]*\s*(?:万元|万|元|块)")
TRADE_CTX = re.compile(
    r"报价|单价|总价|含税|不含税|折扣|毛利|利润率|成本价|"
    r"结算价|佣金|返点|让利|底价|投标价|成交价"
)

# ---------- 4b. 收费语境：对内/对外的运营收费标准也是实际价格数据 ----------
# "冷水 5 元/吨""住宿 200 元/人"是真实定价，不是制度条款；
# "元/单位"用通用匹配（元/次、元/度、元/月……都算），避免新单位漏网。
# 与金额同句出现即判内部（"不收费""免收"这类句子一般不带金额，不会误伤）。
PRICE_CTX = re.compile(
    r"收费标准|水电费|水费|电费|燃气费|住宿费|房租|租金|押金|"
    r"物业费|管理费|伙食费|餐费|停车费|"
    r"元\s*/"
)

# ---------- 5. 真实经营数据（只认明确的财务指标，避免"成本控制"这类制度用语误伤） ----------
FIN_DATA = re.compile(
    r"(营业收入|营业额|净利润|利润总额|毛利率|资产负债率|"
    r"销售额|产值|年度营收)\s*(为|是|约|达到|共计)?\s*\d"
)


def is_sensitive_sentence(sentence: str) -> bool:
    """单句判定：命中任意一条实际数据特征 -> 不可公开"""
    if FIELD_DATA.search(sentence):
        return True
    if HARD_SENSITIVE.search(sentence):
        return True
    if PERSONAL.search(sentence):
        return True
    if TRADE_CTX.search(sentence) and AMOUNT.search(sentence):
        return True
    if PRICE_CTX.search(sentence) and AMOUNT.search(sentence):
        return True
    if FIN_DATA.search(sentence):
        return True
    return False


def _sentences(text: str) -> list:
    """判级前的标准化：压掉所有空白再拆句

    PDF / docx 抽出来的文字经常在数字中间换行（"100-500\\n元/次"），
    不压掉的话金额和语境词会被拆进两句，凑不齐"金额+语境"就漏判了。
    """
    return re.split(r"[。！？；]", re.sub(r"\s+", "", text))


def is_public_block(text: str) -> bool:
    """整块判定：所有句子都干净才算可公开"""
    for sentence in _sentences(text):
        if sentence.strip() and is_sensitive_sentence(sentence):
            return False
    return True


def why_sensitive(text: str) -> str:
    """给出敏感原因（建库时打印，方便人工复核）"""
    for sentence in _sentences(text):
        if not sentence.strip():
            continue
        if FIELD_DATA.search(sentence):
            return "表单字段：" + sentence.strip()[:28]
        if HARD_SENSITIVE.search(sentence):
            return "敏感信息：" + sentence.strip()[:28]
        if PERSONAL.search(sentence):
            return "个人隐私：" + sentence.strip()[:28]
        if TRADE_CTX.search(sentence) and AMOUNT.search(sentence):
            return "报价金额：" + sentence.strip()[:28]
        if PRICE_CTX.search(sentence) and AMOUNT.search(sentence):
            return "收费标准：" + sentence.strip()[:28]
        if FIN_DATA.search(sentence):
            return "经营数据：" + sentence.strip()[:28]
    return ""


# ---------- 6. 导出脱敏：人名不整块删，就地打码 ----------
# 制度文件页首页尾都有"制定/日期:张三  审核/日期:李四"这类签署栏。
# 人名不在正则的能力范围内（重名、姓氏组合太多），但整块删又会连带丢掉正文，
# 所以在导出阶段做定点替换：保留版面结构，只把人名换成 ×。
SIGN_NAME = re.compile(
    r"((?:制定|编制|审核|批准|核准|审定)\s*/\s*日期\s*[:：])"
    r"([ \t\u3000]*)"                       # 只吃同一行的空白，不能跨行
    r"([\u4e00-\u9fff]{2,4})"
    r"(?!\s*/\s*日期)"                      # 后面跟着"/日期"说明是下一个签署栏，不是人名
)
SIGN_NAME_2 = re.compile(
    r"((?:制定人|编制人|审核人|批准人)\s*[:：])([ \t\u3000]*)([\u4e00-\u9fff]{2,4})")


def mask_public(text: str) -> str:
    """导出前的定点脱敏：签署栏人名 -> ×××"""
    text = SIGN_NAME.sub(r"\1\2×××", text)
    text = SIGN_NAME_2.sub(r"\1\2×××", text)
    return text
