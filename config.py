import os
from dotenv import load_dotenv
from prompt import IMG_TO_BBOX, IMG_TO_BBOX_NORM, BAIGE_PROMPT, DUMP_TEXT_JSON

# 加载.env文件中的环境变量
load_dotenv()

TONGYI_API_KEY = os.getenv('TONGYI_API_KEY')
ZHIPU_API_KEY = os.getenv('ZHIPU_API_KEY')

# 检查必要环境变量是否存在
if not TONGYI_API_KEY:
    raise ValueError("请设置 TONGYI_API_KEY 环境变量")

if not ZHIPU_API_KEY:
    raise ValueError("请设置 ZHIPU_API_KEY 环境变量")

llm_configs = {
    "qwen3-vl-30b-instruct-4090": {
        "base_url": "http://192.168.1.192:18434/v1",
        "model": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
        "system_prompt": BAIGE_PROMPT,
    },
    "qwen3-vl-30b-thinking-4090": {
        "base_url": "http://192.168.1.192:19434/v1",
        "model": "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8",
        "system_prompt": BAIGE_PROMPT,
    },
    "qwen3-vl-30b-instruct": {
        "api_key": TONGYI_API_KEY,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3-vl-30b-a3b-instruct",
        "system_prompt": BAIGE_PROMPT,
    },
    "qwen3-vl-30b-thinking": {
        "api_key": TONGYI_API_KEY,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3-vl-30b-a3b-thinking",
        "system_prompt": BAIGE_PROMPT,
    },
    "qwen3-vl": {
        "api_key": TONGYI_API_KEY,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3-vl-235b-a22b-instruct",
        "system_prompt": BAIGE_PROMPT,
    },
    "qwen3-vl-thinking": {
        "api_key": TONGYI_API_KEY,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3-vl-235b-a22b-thinking",
        "system_prompt": BAIGE_PROMPT,
    },
    "qwen3-next": {
        "api_key": TONGYI_API_KEY,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3-next-80b-a3b-instruct",
        "system_prompt": DUMP_TEXT_JSON,
    },
    "qwen-vl-max": {
        "api_key": TONGYI_API_KEY,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-max",
        "system_prompt": BAIGE_PROMPT,
    },
    "glm-4.5v": {
        "api_key": ZHIPU_API_KEY,
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "model": "glm-4.5v",
        "system_prompt": BAIGE_PROMPT,
    },
}