import base64
import json

from rich import print
from openai import OpenAI

from prompt import IMG_TO_BBOX, IMG_TO_BBOX_NORM


client = OpenAI(
    api_key="8dd3cb1ced784a0aaede31cb6c8595ce.URPzLkji9BeRvADu",
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)
model="glm-4.5v"

# client = OpenAI(
#     api_key="sk-4f2d9f61f54340a4a3568a1553be3fd9",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
# model="qwen-vl-max-latest"

def detect_objects(user_prompt, img_path):
    """
    检测图片中物体的位置

    Args:
        prompt (str): 提示词，描述需要识别的物体
        img_path (str): 图片路径

    Returns:
        dict: 包含识别结果的字典列表，每个字典包含物体名称和边界框坐标
    """
    with open(img_path, "rb") as img_file:
        img_base = base64.b64encode(img_file.read()).decode("utf-8")

    print("正在识别图片中物体的位置...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": IMG_TO_BBOX_NORM},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base}"}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
        response_format={
            "type": "json_object"
        }
    )
    print("结果:", response.choices[0].message.content)
    # result = json.loads(response.choices[0].message.content)
    result = response.choices[0].message.content
    return result


# 示例用法
if __name__ == "__main__":

    from vis import draw_yolo_style 

    prompt = "请从图片中识别出 所有水果 的位置"
    img_path = r"tmp\test.png"

    result_json = detect_objects(prompt, img_path)

    if result_json:
        result_dict = json.loads(result_json)

        draw_yolo_style(img_path, result_dict)
