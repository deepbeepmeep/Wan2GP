import re
from opencc import OpenCC


t2s_converter = OpenCC("t2s")
s2t_converter = OpenCC("s2t")


EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # Emoticons
    "]+",
    flags=re.UNICODE,
)

# 创建一个翻译表，用于替换和移除字符
TRANSLATION_TABLE = str.maketrans(
    {
        "-": " ",  # 将 '-' 替换为空格
        ",": None,
        ".": None,
        "，": None,
        "。": None,
        "!": None,
        "！": None,
        "?": None,
        "？": None,
        "…": None,
        ";": None,
        "；": None,
        ":": None,
        "：": None,
        "\u3000": " ",  # 将全角空格替换为空格
    }
)

# 替换括号中的内容，包括中括号和小括号
BACKSLASH_PATTERN = re.compile(r"\(.*?\)|\[.*?\]")

SPACE_PATTERN = re.compile("(?<!^)\s+(?!$)")


def normalize_text(text, language, strip=True):
    """
    对文本进行标准化处理，去除标点符号，转为小写（如果适用）
    """
    # Step 1: 替换 '-' 为 ' ' 并移除标点符号
    text = text.translate(TRANSLATION_TABLE)

    # Step 2: 移除表情符号
    text = EMOJI_PATTERN.sub("", text)

    # Step 3: 连续空白字符替换为单个空格，首位除外
    text = SPACE_PATTERN.sub(" ", text)

    # Step 4: 去除首尾空白字符（如果需要）
    if strip:
        text = text.strip()

    # Step 5: 转为小写
    text = text.lower()

    # Step 6: 多语言转换
    if language == "zh":
        text = t2s_converter.convert(text)
    if language == "yue":
        text = s2t_converter.convert(text)
    # 其他语言根据需要添加
    return text
