import base64
import json
from typing import Dict, Any

from rich import print 
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

    def json_loads(self, response_text):
        """ 简单提取JSON部分的方法 """
        try:  # 首先尝试直接解析整个响应文本
            return json.loads(response_text)
        except json.JSONDecodeError:
            print("无法解析JSON，尝试提取文本中 JSON 部分")
        
        try:  # 查找第一个{和最后一个}的位置
            first_brace = response_text.find('{')
            last_brace = response_text.rfind('}')
            
            if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
                json_str = response_text[first_brace:last_brace+1]  # 提取JSON字符串
                return json.loads(json_str)  # 尝试解析
            
        except json.JSONDecodeError:
            print("提取文本中 JSON 部分失败")
        
        return None
    
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
            "stream": True,
        }
        
        if json_mode:  # 只在需要时添加 response_format 参数
            api_params["response_format"] = {"type": "json_object"}
        
        print(api_params) 
        
        response = self.client.chat.completions.create(**api_params)

        
        print("LLM 实时回复: ", end="", flush=True)
        content_parts = []
        for chunk in response:
            # 最后一个chunk不包含choices，但包含usage信息。
            if chunk.choices:
                # 关键：delta.content可能为None，使用`or ""`避免拼接时出错。
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                content_parts.append(content)
            
            else:
                print("Usage:")
                print(chunk.usage)

        full_response = "".join(content_parts)
        # print(f"content_parts: {full_response}")

        # 获取 AI 回复
        ai_response = full_response

        # ai_response = response.choices[0].message.content

        # 根据是否使用历史记录来决定如何添加到对话历史
        if use_history:
            self.conversation.append({"role": "user", "content": message_content})  # type: ignore
            self.conversation.append({"role": "assistant", "content": ai_response})  # type: ignore

        return ai_response  # type: ignore

    def clear_history(self):
        """清除对话历史，保留系统提示词"""
        self.conversation = [self.conversation[0]]




if __name__ == "__main__":

    import cv2
    import json
    import time

    from vis import draw_bbox
    from config import llm_configs

    bot = ChatBot(llm_configs["glm-4.5v"])
    # bot = ChatBot(llm_configs["qwen-vl-max"])
    # bot = ChatBot(llm_configs["qwen3-next"])

    prompt = "找到画面中的水果和饮料"
    img_path = "tmp/test1.png"
    img_base64 = bot.encode_image(img_path)

    print(f"prompt 输入: {prompt}")
    # result_text = bot.chat("画面里有什么", img_base64, json_mode=False)

    start_time = time.time()

    result_text = bot.chat(prompt, img_base64, json_mode=True)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"函数执行时间: {execution_time:.2f} 秒")
    # print("模型返回结果JSON:")
    # print(result_text)

    result_dict = json.loads(result_text)

    print("模型返回结果字典")
    print(result_dict)

    img = draw_bbox(img_path, result_dict)
    cv2.imshow("YOLO", img)

    img = draw_bbox(img_path, result_dict, None, 1000.0)
    cv2.imshow("YOLO_NORM", img)
    cv2.waitKey(0)
