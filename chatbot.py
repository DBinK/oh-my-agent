
import base64
from openai import OpenAI



class ChatBot:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )
        self.conversation = [
            {"role": "system", "content": "你是一个有用的 AI 助手"}
        ]
    
    def encode_image(self, image_path: str) -> str:
        """将图像编码为 base64 字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def chat(self, user_input: str, img_base64: str|None = None, use_history: bool = False) -> str:
        # 添加用户消息
        if img_base64:
            # 如果有图像，构造包含图像和文本的消息
            message_content = [
                {"type": "text", "text": user_input},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            ]
        else:
            # 如果没有图像，只使用文本
            message_content = user_input
        
        # 构造消息列表
        if use_history:
            # 使用历史对话记录
            messages = self.conversation.copy()
            messages.append({"role": "user", "content": message_content})
        else:
            # 不使用历史对话记录，只发送系统提示和当前消息
            messages = [self.conversation[0], {"role": "user", "content": message_content}]
        
        # 调用 API
        response = self.client.chat.completions.create(
            model="glm-4-air-250414",
            messages=messages,
            temperature=0.7
        )
        
        # 获取 AI 回复
        ai_response = response.choices[0].message.content
        
        # 根据是否使用历史记录来决定如何添加到对话历史
        if use_history:
            self.conversation.append({"role": "user", "content": message_content})
            self.conversation.append({"role": "assistant", "content": ai_response})
        # 如果不使用历史记录，则不添加到对话历史中
        
        return ai_response
    
    def clear_history(self):
        """清除对话历史，保留系统提示"""
        self.conversation = self.conversation[:1]


if __name__ == "__main__":
    from rich import print

    bot = ChatBot("8dd3cb1ced784a0aaede31cb6c8595ce.URPzLkji9BeRvADu")

    print(bot.chat("你好，请介绍一下自己"))
    print(bot.chat("你能帮我写代码吗？"))
    print(bot.chat("写一个 Python 的快速排序算法"))