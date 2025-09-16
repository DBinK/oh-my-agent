
import time

import cv2
from looptick import LoopTick
from rich import print

from chatbot import ChatBot
from config import llm_configs
from vis import draw_bbox

stamp = LoopTick()

bot = ChatBot(llm_configs["glm-4.5v"])
# bot = ChatBot(llm_configs["qwen-vl-max"])
# bot = ChatBot(llm_configs["qwen3-next"])
json_dumper = ChatBot(llm_configs["qwen3-next"])

# prompt = "把奶龙放到白色托盘里"
# prompt = "把所有方块放到碗里里"
prompt = "把桌面上的水果放到键盘上"
# prompt = "把饮料瓶放到书本上"
img_path = "tmp/test1.png"
img_base64 = bot.encode_image(img_path)

print(f"prompt 输入: {prompt}")
# result_text = bot.chat("画面里有什么", img_base64, json_mode=False)

start = stamp.tick_sec()

result_text = bot.chat(prompt, img_base64, json_mode=True)

end = stamp.tick_sec()

print(f"bot执行时间: {end:.2f} 秒")
# print("模型返回结果 JSON:")
# print(result_text)

result_dict = bot.json_loads(result_text)

if result_dict is not None:

    print("模型返回结果字典")
    print(result_dict)

    objs = result_dict.get("objs", {})
    print("检测到的对象:")
    print(objs)

    img = draw_bbox(img_path, result_dict)
    cv2.imshow("YOLO", img)

    img = draw_bbox(img_path, result_dict, None, 1000.0)
    cv2.imshow("YOLO_NORM", img)
    cv2.waitKey(0)
