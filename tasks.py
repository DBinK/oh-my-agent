# tasks_blinker.py
from blinker import Signal
import threading
import time
import keyboard

# ================== 定义信号 ==================
object_detected = Signal()
hand_position = Signal()

# ================== 定义任务 ==================
def object_tracking():
    while True:
        print("[ObjectTracking] 检测物体...")
        object_detected.send({"name": "red_block"})
        time.sleep(1)

def hand_control():
    while True:
        print("[HandControl] 跟随手部移动...")
        hand_position.send({"x": 0.5, "y": 0.3})
        time.sleep(1)

def robot_control():
    def on_object(sender):
        print(f"[Robot] 收到物体事件: {sender['name']}, 执行抓取动作")

    def on_hand(sender):
        print(f"[Robot] 收到手部位置事件: {sender}, 控制机械臂跟随")

    object_detected.connect(on_object)
    hand_position.connect(on_hand)

# ================== 启动线程 ==================
threading.Thread(target=robot_control, daemon=True).start()

# 主循环按键事件触发任务
print("按键: 1=追踪模式, 2=手势模式, g=抓取, q=退出")
while True:
    if keyboard.is_pressed("1"):
        threading.Thread(target=object_tracking, daemon=True).start()
        time.sleep(0.3)

    elif keyboard.is_pressed("2"):
        threading.Thread(target=hand_control, daemon=True).start()
        time.sleep(0.3)

    elif keyboard.is_pressed("g"):
        object_detected.send({"name": "red_block"})
        time.sleep(0.3)

    elif keyboard.is_pressed("q"):
        break

    time.sleep(0.05)
