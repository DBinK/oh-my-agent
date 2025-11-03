import os
from pathlib import Path
import dashscope
from rich import print

# 可配置参数
API_KEY = os.getenv("DASHSCOPE_API_KEY")
assert API_KEY, "请设置环境变量 DASHSCOPE_API_KEY"

MODEL_NAME = "qwen3-vl-flash"
IMAGE_PATHS = [
    r"tmp\flip_milk_train_data\flip_milk_1_success\color_0\color_0_0.jpg",
    r"tmp\flip_milk_train_data\flip_milk_1_success\color_0\color_0_133.jpg",
]

SESSION_STR = "把牛奶扶起来"
PROMPT_TEMPLATE = f"""
你是一个视觉语言模型（VLM），请根据以下信息判断机械臂执行任务是否成功：

- **任务描述**：{SESSION_STR}
- **执行前画面**：[图像1]
- **执行后画面**：[图像2]

请仔细对比两帧图像，并结合任务描述进行推理。最终仅输出一个 JSON 对象，格式如下：

{{ 
  "success": true 或 false,
  "reason": "简要中文说明判断依据"
}}

注意：
- 仅根据图像内容和任务描述进行客观判断；
- 不要假设图像中未呈现的信息；
- 如果任务目标在执行后画面中已达成，则 success 为 true；
- 否则为 false，并说明缺失或错误之处。
"""


# 辅助函数
def path_to_file_uri(local_path: str) -> str:
    """将本地路径转换为 file:// URI 格式，使用 pathlib.Path 处理路径"""
    path_obj = Path(local_path).resolve()
    return f"file://{path_obj.as_posix()}"


def prepare_content(image_paths: list, prompt: str) -> list:
    """准备发送给模型的内容"""
    content = [{"image": path_to_file_uri(path)} for path in image_paths]
    content.append({"text": prompt})
    return content


if __name__ == "__main__":
    # 打印API密钥用于调试
    # print(API_KEY)

    # 准备内容
    content = prepare_content(IMAGE_PATHS, PROMPT_TEMPLATE)

    # 构建消息
    messages = [{"role": "user", "content": content}]

    # 调用模型
    response = dashscope.MultiModalConversation.call(
        api_key=API_KEY,
        model=MODEL_NAME,
        messages=messages,
    )

    # 打印结果
    print(response["output"]["choices"][0]["message"].content[0]["text"])  # type: ignore
