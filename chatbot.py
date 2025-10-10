import base64
import json
from typing import Dict, Any

from rich import print 
from openai import OpenAI
import tiktoken


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
        api_key = config.get("api_key", "")
        base_url = config.get("base_url")
        system_prompt = config.get("system_prompt")

        self.model = config.get("model", "gpt-4o-mini")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.conversation = [{"role": "system", "content": system_prompt}]

    def encode_image(self, image_path: str) -> str:
        """将图像编码为 base64 字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def extract_json(self, response_text: str) -> str|None:
        """ 简单提取JSON部分的方法 """
        try:  # 首先尝试直接解析整个响应文本
            json.loads(response_text)  # 验证是否为有效的JSON
            return response_text  # 返回原始文本
        except json.JSONDecodeError:
            print("JSON解析失败, 尝试查找JSON部分")
        
        try:  # 查找第一个{和最后一个}的位置
            first_brace = response_text.find('{')
            last_brace = response_text.rfind('}')
            
            if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
                json_str = response_text[first_brace:last_brace+1]  # 提取JSON字符串
                json.loads(json_str)  # 验证是否为有效的JSON
                return json_str       # 只返回JSON字符串
            
        except json.JSONDecodeError:
            print("查找文本中 JSON 部分失败, 返回原始文本")
            return response_text
    
    def json_loads(self, response_text: str) -> dict|None:
        """ 从响应文本中提取 JSON 字符串并加载为字典 """
        json_str = self.extract_json(response_text)
        if json_str is not None:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                print("解析提取的JSON字符串失败")
                return None
        return None
    
        
    def count_tokens(self, text: str, model_name: str = "gpt-5"):
        """ 计算给定文本的token数量 """
        tokenizer = tiktoken.encoding_for_model(model_name)
        tokens = tokenizer.encode(text)
        tokens_count = len(tokens)
        
        return tokens_count, len(text), tokens  # 返回token数量 字符数 编码的token列表

    def chat(
        self,
        user_input: str,                 # Prompt 输入
        img_base64: str | None = None,   # base64 编码的图像数据
        use_history: bool = False,       # 是否使用历史对话记录
        json_mode: bool | None = False,   # 是否使用 JSON 模式
    ) -> str:
        
        if img_base64:            # 添加用户消息
            message_content = [   # 如果有图像，构造包含图像和文本的消息
                {"type": "text", "text": user_input},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                },
            ]
        else:
            message_content = user_input   # 如果没有图像，只使用文本
      
        if use_history:   # 构造消息列表
            messages = self.conversation.copy()   # 使用历史对话记录
            messages.append({"role": "user", "content": message_content})  # type: ignore
        else:
            messages = [   # 不使用历史对话记录，只发送系统提示和当前消息
                self.conversation[0],
                {"role": "user", "content": message_content},
            ]

        api_params = {  # 构造参数字典，避免传递 None 值
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            # "temperature": 1.0,
            "stream": True,
        }
        
        if json_mode:  # 只在需要时添加 response_format 参数
            api_params["response_format"] = {"type": "json_object"}

        # 调用 API
        response = self.client.chat.completions.create(**api_params)
        
        print("LLM 实时回复: ", flush=True)
        content_parts = []
        for chunk in response:
            if chunk.choices:   # 关键：delta.content可能为None，使用`or ""`避免拼接时出错。
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                content_parts.append(content)
        print("\n")

        if not content_parts:  # 如果没有内容，则返回空字符串
            full_response_str = ""
        else:
            full_response_str = "".join(content_parts)
            
        # full_response_str = response.choices[0].message.content  
       
        if use_history:  # 根据是否使用历史记录来决定如何添加到对话历史
            self.conversation.append({"role": "user", "content": message_content})  # type: ignore
            self.conversation.append({"role": "assistant", "content": full_response_str})  # type: ignore

        if json_mode:
            ai_response = self.extract_json(full_response_str)

        return ai_response  # type: ignore
    
    def chat_with_metrics(
        self,
        user_input: str,  # Prompt 输入
        img_base64: str | None = None,  # base64 编码的图像数据
        use_history: bool = False,  # 是否使用历史对话记录
        json_mode: bool | None = False,  # 是否使用 JSON 模式
    ) -> dict:
        
        if img_base64:            # 添加用户消息
            message_content = [   # 如果有图像，构造包含图像和文本的消息
                {"type": "text", "text": user_input},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                },
            ]
        else:
            message_content = user_input   # 如果没有图像，只使用文本
      
        if use_history:   # 构造消息列表
            messages = self.conversation.copy()   # 使用历史对话记录
            messages.append({"role": "user", "content": message_content})  # type: ignore
        else:
            messages = [   # 不使用历史对话记录，只发送系统提示和当前消息
                self.conversation[0],
                {"role": "user", "content": message_content},
            ]

        api_params = {  # 构造参数字典，避免传递 None 值
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            # "temperature": 1.0,
            "stream": True,
        }
        
        if json_mode:  # 只在需要时添加 response_format 参数
            api_params["response_format"] = {"type": "json_object"}

        # 初始化性能统计变量
        start_time = time.time()
        first_token_time = None

        # 调用 API
        response = self.client.chat.completions.create(**api_params)
        
        print("LLM 实时回复: ", flush=True)
        
        content_parts = []
        for chunk in response:
            if chunk.choices:   # 关键：delta.content可能为None，使用`or ""`避免拼接时出错。
                content = chunk.choices[0].delta.content or ""
                
                if first_token_time is None and content:  # 记录首字延迟
                    first_token_time = time.time()
                    
                print(content, end="", flush=True)
                content_parts.append(content)
                
        print("\n")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if not content_parts:  # 如果没有内容，则返回空字符串
            full_response_str = ""
        else:
            full_response_str = "".join(content_parts)
        
        if first_token_time is not None:  # 计算首字延迟
            first_token_delay = first_token_time - start_time
        else:
            first_token_delay = 0.0
        
        token_count, char_count, _ = self.count_tokens(full_response_str)  # 计算 token 指标
        if token_count > 0:
            tokens_per_second = token_count / total_time if total_time > 0 else 0
            char_per_second = char_count / total_time if total_time > 0 else 0
        else:
            tokens_per_second = 0.0
            char_per_second = 0.0
            
        print(f"首字延迟: {first_token_delay:.2f} 秒")
        print(f"tokens avg: {tokens_per_second:.2f} tok/s, char avg: {char_per_second:.2f} char/s")
        print(f"总token数: {token_count}")
            
        # full_response_str = response.choices[0].message.content  
       
        if use_history:  # 根据是否使用历史记录来决定如何添加到对话历史
            self.conversation.append({"role": "user", "content": message_content})  # type: ignore
            self.conversation.append({"role": "assistant", "content": full_response_str})  # type: ignore

        if json_mode:
            full_response_str = self.extract_json(full_response_str)

        return {
            "response": full_response_str,
            "metrics": {
                "first_token_delay": first_token_delay,
                "total_time": total_time,
                "token_count": token_count,
                "tokens_per_second": tokens_per_second,
                "char_count": char_count,
                "char_per_second": char_per_second
            }
        }

    def clear_history(self):
        """清除对话历史，保留系统提示词"""
        self.conversation = [self.conversation[0]]


if __name__ == "__main__":

    import cv2
    import json
    import time

    from vis import draw_bbox
    from config import llm_configs

    # bot = ChatBot(llm_configs["glm-4.5v"])
    # bot = ChatBot(llm_configs["qwen-vl-max"])
    # bot = ChatBot(llm_configs["qwen3-next"])
    bot = ChatBot(llm_configs["qwen3-vl-30b-instruct"])

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
