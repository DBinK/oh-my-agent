import base64
import json
import time
from typing import Dict, Any
from rich import print
from openai import OpenAI
import tiktoken


class ChatBot:
    def __init__(self, config: Dict[str, Any], silent: bool = False):
        """初始化 ChatBot 实例"""
        self.silent = silent # 控制是否打印流式输出
        self.model = config.get("model", "gpt-4o-mini")
        self.client = OpenAI(
            api_key=config.get("api_key", ""),
            base_url=config.get("base_url"),
        )
        system_prompt = config.get("system_prompt", "")
        self.conversation = [{"role": "system", "content": system_prompt}]

    # ===================== 工具函数 =====================
    def encode_image(self, image_path: str) -> str:
        """将图像编码为 base64 字符串"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def extract_json(self, response_text: str) -> str | None:
        """从文本中提取 JSON 内容"""
        try:
            json.loads(response_text)
            return response_text
        except json.JSONDecodeError:
            pass

        first, last = response_text.find("{"), response_text.rfind("}")
        if first != -1 and last != -1:
            json_str = response_text[first : last + 1]
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass
        return None

    def count_tokens(self, text: str, model_name: str = "gpt-5"):
        """统计 token 数"""
        tokenizer = tiktoken.encoding_for_model(model_name)
        tokens = tokenizer.encode(text)
        return len(tokens), len(text), tokens

    # ===================== 公共辅助函数 =====================
    def _build_message_content(self, text: str, img_base64: str | None):
        """ 如果需要图片, 则构造 message 内容为 [text, image_url] """
        if img_base64:
            return [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
            ]
        return text

    def _build_messages(self, content, use_history: bool):
        """ 如果需要历史对话, 则构造消息列表为 [system_prompt, user_input, assistant_output] """
        if use_history:
            msgs = self.conversation.copy()
            msgs.append({"role": "user", "content": content})
        else:
            msgs = [self.conversation[0], {"role": "user", "content": content}]
        return msgs

    def _build_api_params(self, messages, json_mode: bool):
        """ 统一构造 API 调用参数 """
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "stream": True,
        }
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        return params

    def _stream_response(self, response):
        """统一处理流式输出"""
        if not self.silent:
            print(f"模型: {self.model} 流式输出响应: ", flush=True)
        
        parts, first_token_time = [], None
        for chunk in response:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content or ""
            if first_token_time is None and content:
                first_token_time = time.time()
            parts.append(content)
            
            if not self.silent:
                print(content, end="", flush=True)
                
        print("\n")
        return "".join(parts), first_token_time

    # ===================== 核心功能函数 =====================
    def chat(self, user_input: str, img_base64: str | None = None, use_history=False, json_mode=False) -> str:
        """普通聊天接口"""
        msg_content = self._build_message_content(user_input, img_base64)
        messages = self._build_messages(msg_content, use_history)
        params = self._build_api_params(messages, json_mode)

        response = self.client.chat.completions.create(**params)
        text, _ = self._stream_response(response)

        if use_history:
            self.conversation += [
                {"role": "user", "content": msg_content},
                {"role": "assistant", "content": text},
            ]

        return self.extract_json(text) if json_mode else text

    def chat_with_metrics(self, user_input: str, img_base64: str | None = None, use_history=False, json_mode=False) -> dict:
        """带性能指标的聊天接口"""
        start = time.time()
        msg_content = self._build_message_content(user_input, img_base64)
        messages = self._build_messages(msg_content, use_history)
        params = self._build_api_params(messages, json_mode)

        response = self.client.chat.completions.create(**params)
        text, first_token_time = self._stream_response(response)
        end = time.time()

        token_count, char_count, _ = self.count_tokens(text)
        total_time = end - start
        first_delay = (first_token_time - start) if first_token_time else 0.0

        metrics = {
            "first_token_delay": first_delay,
            "total_time": total_time,
            "token_count": token_count,
            "tokens_per_second": token_count / total_time if total_time > 0 else 0,
            "char_count": char_count,
            "char_per_second": char_count / total_time if total_time > 0 else 0,
        }

        print(f"首字延迟: {metrics['first_token_delay']:.2f}s")
        print(f"平均速率: {metrics['tokens_per_second']:.2f} tok/s")

        if use_history:
            self.conversation += [
                {"role": "user", "content": msg_content},
                {"role": "assistant", "content": text},
            ]

        return {"response": self.extract_json(text) if json_mode else text, "metrics": metrics}

    def clear_history(self):
        """清除对话历史"""
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
