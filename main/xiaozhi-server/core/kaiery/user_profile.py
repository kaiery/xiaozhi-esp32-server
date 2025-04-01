import urllib.parse

from config.logger import setup_logging

TAG = __name__
logger = setup_logging()


def get_user_info(headers):
    """获取用户信息并以自然语言组合"""
    # 从headers中提取用户信息，如果不存在则为空字符串
    user_id = headers.get("user_id", "")
    user_name = headers.get("user_name", "")
    user_age = headers.get("user_age", "")
    user_gender = headers.get("user_gender", "").lower()
    user_from = headers.get("user_from", "")

    user_id = urllib.parse.unquote(user_id, encoding='utf-8')
    user_name = urllib.parse.unquote(user_name, encoding='utf-8')
    user_age = urllib.parse.unquote(user_age, encoding='utf-8')
    user_gender = urllib.parse.unquote(user_gender, encoding='utf-8')
    user_from = urllib.parse.unquote(user_from, encoding='utf-8')

    # Build introduction parts
    parts = []

    # Start with name if available
    if user_name:
        parts.append(f"Hi, I'm {user_name}")
    else:
        parts.append("Hi there")

    # Add age and gender naturally
    if user_age and user_gender:
        parts.append(f"a {user_age}-year-old {user_gender}")
    elif user_age:
        parts.append(f"{user_age} years old")
    elif user_gender:
        parts.append(f"a {user_gender}")

    # Add from information
    if user_from:
        parts.append(f"from {user_from}")

    # Add ID only if no other personal info exists
    if user_id and not (user_name or user_age or user_gender or user_from):
        parts.append(f"My ID is {user_id}")

    if not (user_id or user_name or user_age or user_gender or user_from):
        introduction = None
    else:
        # Combine all parts into a sentence
        introduction = " ".join(parts) + "."

    return introduction
