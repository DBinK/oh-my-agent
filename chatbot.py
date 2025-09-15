import base64
from typing import Dict, Any

from openai import OpenAI


class ChatBot:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 ChatBot 实例

        Args:
            config: 包含配置信息的字典，可以包含以下键:
                - api_key (str): API 密钥
                - base_url (str, optional): API 基础 URL
                - system_prompt (str, optional): 系统提示词
        """
        api_key = config.get("api_key")
        base_url = config.get("base_url")
        system_prompt = config.get("system_prompt")

        self.model = config.get("model", "gpt-4o-mini")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.conversation = [{"role": "system", "content": system_prompt}]

    def encode_image(self, image_path: str) -> str:
        """将图像编码为 base64 字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def chat(
        self,
        user_input: str,                 # Prompt 输入
        img_base64: str | None = None,   # base64 编码的图像数据
        use_history: bool = False,       # 是否使用历史对话记录
        json_mode: bool | None = None,   # 是否使用 JSON 模式
    ) -> str:
        # 添加用户消息
        if img_base64:
            # 如果有图像，构造包含图像和文本的消息
            message_content = [
                {"type": "text", "text": user_input},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                },
            ]
        else:
            # 如果没有图像，只使用文本
            message_content = user_input

        # 构造消息列表
        if use_history:
            # 使用历史对话记录
            messages = self.conversation.copy()
            messages.append({"role": "user", "content": message_content})  # type: ignore
        else:
            # 不使用历史对话记录，只发送系统提示和当前消息
            messages = [
                self.conversation[0],
                {"role": "user", "content": message_content},
            ]

        # 调用 API
        api_params = {  # 构造参数字典，避免传递 None 值
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
        }
        
        if json_mode:  # 只在需要时添加 response_format 参数
            api_params["response_format"] = {"type": "json_object"}
            
        response = self.client.chat.completions.create(**api_params)

        # 获取 AI 回复
        ai_response = response.choices[0].message.content

        # 根据是否使用历史记录来决定如何添加到对话历史
        if use_history:
            self.conversation.append({"role": "user", "content": message_content})  # type: ignore
            self.conversation.append({"role": "assistant", "content": ai_response})  # type: ignore

        return ai_response  # type: ignore

    def clear_history(self):
        """清除对话历史，保留系统提示"""
        self.conversation = [self.conversation[0]]


if __name__ == "__main__":

    import json
    from rich import print

    from prompt import IMG_TO_BBOX, IMG_TO_BBOX_NORM
    from vis import draw_yolo_style

    glm45v_config = {
        "model": "glm-4.5v",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "system_prompt": IMG_TO_BBOX_NORM,
        "api_key": "8dd3cb1ced784a0aaede31cb6c8595ce.URPzLkji9BeRvADu",
    }

    bot = ChatBot(glm45v_config)

    img_path = r"tmp\test.png"
    img_base64 = bot.encode_image(img_path)

    result_text = bot.chat("找到图中的奶龙", json_mode=True)

    print(result_text)

    result_dict = json.loads(result_text)

    print(result_dict)

    draw_yolo_style(img_path, result_dict)
