import os
from dotenv import load_dotenv
from prompt import IMG_TO_BBOX, IMG_TO_BBOX_NORM, BAIGE_PROMPT, DUMP_TEXT_JSON

# 加载.env文件中的环境变量
load_dotenv()

TONGYI_API_KEY = os.getenv('TONGYI_API_KEY')
ZHIPU_API_KEY = os.getenv('ZHIPU_API_KEY')

llm_configs = {
    "qwen-vl-max": {
        "api_key": TONGYI_API_KEY,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-max",
        "system_prompt": BAIGE_PROMPT,
    },
    "qwen3-next": {
        "api_key": TONGYI_API_KEY,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3-next-80b-a3b-instruct",
        "system_prompt": DUMP_TEXT_JSON,
    },
    "glm-4.5v": {
        "api_key": ZHIPU_API_KEY,
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "model": "glm-4.5v",
        "system_prompt": BAIGE_PROMPT,
    },
}