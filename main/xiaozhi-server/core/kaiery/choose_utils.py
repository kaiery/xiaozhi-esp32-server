import datetime

from config.logger import setup_logging
from core.auth import AuthMiddleware, AuthenticationError

TAG = __name__
logger = setup_logging()


def choose_system_prompt(config, headers):
    """
    根据token，选择系统提示词
    """
    auth_header = headers.get("authorization", "")
    token = auth_header.split(" ")[1] if " " in auth_header else None
    # 选择 prompt（如果 token 无效则用默认 prompt）
    prompt = config['prompts'].get(token, config['prompt'])
    if "{date_time}" in prompt:
        prompt = prompt.format(date_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return prompt
