import os
from pathlib import Path
from http import HTTPStatus
import time

import dashscope
from rich import print
from natsort import natsorted

# 可配置参数
API_KEY = os.getenv("DASHSCOPE_API_KEY")
assert API_KEY, "请设置环境变量 DASHSCOPE_API_KEY" 

MODEL_NAME = "qwen3-vl-flash"
# IMAGE_DIR = r"tmp\flip_milk_train_data\flip_milk_1_success\color_0"
# IMAGE_DIR = r"tmp\flip_milk_train_data\flip_milk_7_success\color_0"
IMAGE_DIR = r"tmp\flip_milk_train_data\flip_milk_5\color_0"
FPS = 1
SAMPLE_FRAMES = 8  # 要平均采样的帧数 >= 4

SESSION_STR = "把牛奶扶起来"
PROMPT_TEMPLATE = f"""
你是一个视觉语言模型（VLM），请根据以下信息判断机械臂执行任务是否成功：

- **任务描述**：{SESSION_STR}

请仔细观看视频，并结合任务描述进行推理。最终仅输出一个 JSON 对象，格式如下：

{{ 
  "success": true 或 false,
  "reason": "简要中文说明判断依据"
}}

注意：
- 仅根据视频内容和任务描述进行客观判断；
- 不要假设视频中未呈现的信息；
- 如果任务目标在视频中已达成，则 success 为 true；
- 否则为 false，并说明缺失或错误之处。
"""


# 辅助函数
def get_image_paths_from_dir(dir_path: str) -> list:
    """从指定目录获取所有图片路径，按文件名排序"""
    dir_obj = Path(dir_path)
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_paths = [str(p) for p in dir_obj.iterdir() 
                   if p.suffix.lower() in image_extensions]
    # 按文件名排序确保顺序正确
    image_paths = natsorted(image_paths)
    return image_paths

def sample_frames(image_paths: list, num_samples: int) -> list:
    """从图像路径列表中均匀采样指定数量的帧"""
    if len(image_paths) <= num_samples:
        return image_paths
    
    indices = [int(i * (len(image_paths) - 1) / (num_samples - 1)) for i in range(num_samples)]
    sampled_paths = [image_paths[i] for i in indices]
    return sampled_paths


def path_to_file_uri(local_path: str) -> str:
    """将本地路径转换为 file:// URI 格式，使用 pathlib.Path 处理路径"""
    path_obj = Path(local_path).resolve()
    return f"file://{path_obj.as_posix()}"


def prepare_content(image_paths: list, prompt: str, fps: int) -> list:
    """准备发送给模型的内容"""
    # 将本地路径转换为URI
    image_uris = [path_to_file_uri(path) for path in image_paths]
    
    # 构造视频内容，包含fps参数
    content = [{"video": image_uris, "fps": fps}]
    content.append({"text": prompt})
    return content



if __name__ == "__main__":
    
    # 获取目录下所有图片
    image_paths = get_image_paths_from_dir(IMAGE_DIR)
    print(f"找到 {len(image_paths)} 张图片")
    # print("图片路径：")
    # print(image_paths)
    
    # 均匀采样帧
    if SAMPLE_FRAMES and SAMPLE_FRAMES < len(image_paths):
        image_paths = sample_frames(image_paths, SAMPLE_FRAMES)
        print(f"采样后剩余 {len(image_paths)} 张图片")
        print("图片路径：")
        print(image_paths)

    # 准备内容
    content = prepare_content(image_paths, PROMPT_TEMPLATE, FPS)

    # 构建消息
    messages = [{"role": "user", "content": content}]

    # 调用模型
    print(f"正在理解视频任务: {SESSION_STR} ...")
    
    start = time.time()
    
    response = dashscope.MultiModalConversation.call(
        api_key=API_KEY,
        model=MODEL_NAME,
        messages=messages,
        stream=True,
        incremental_output=True,  # 关键：设置为True以获取增量输出，性能更佳。
    )

    # 打印结果
    # print(response["output"]["choices"][0]["message"].content[0]["text"]) # type: ignore
    
    # 3. 处理流式响应
    content_parts = []
    print("AI: ", end="", flush=True)

    for resp in response:
        if resp.status_code == HTTPStatus.OK:
            content = resp.output.choices[0].message.content
            if content:
                content_str: str = content[0].get('text') # type: ignore
                print(content_str, end="", flush=True)
                content_parts.append(content_str)

            # 检查是否是最后一个包
            if resp.output.choices[0].finish_reason == "stop":
                usage = resp.usage
                print("\n--- 请求用量 ---")
                print(f"输入 Tokens: {usage.input_tokens}")
                print(f"输出 Tokens: {usage.output_tokens}")
                print(f"总计 Tokens: {usage.total_tokens}")
        else:
            # 处理错误情况
            print(
                f"\n请求失败: request_id={resp.request_id}, code={resp.code}, message={resp.message}"
            )
            break
        
    # print("content_parts:", content_parts)
    full_response = "".join(content_parts)
    end = time.time()
    spend_time = end - start
    print(f"总耗时: {spend_time:.2f} 秒")
    
    # print("\n完整结果:")
    # print(full_response)
    
    