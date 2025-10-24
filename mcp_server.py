# mcp_sever.py

import time
from random import random
from typing import  Optional

from rich import print
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel


class ObjectsResult(BaseModel):
    """表示物体检测结果的数据模型"""
    image_path: str = "tmp/test1.png"  # 检测结果的图片路径
    objects: dict[str, tuple[float, float, float]]  # {name: (x, y, z)}


class RobotPose(BaseModel):
    """表示机器人位姿的数据模型"""
    position_mm: tuple[float, float, float]         # (x, y, z) in mm
    orientation: tuple[float, float, float, float]  # (x, y, z, w) in quaternion


class RobotResponse(BaseModel):
    """表示机器人操作响应的数据模型"""
    success: bool
    message: str


class SynaBotServer:
    """SynaBot MCP服务器类"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        self.mcp = FastMCP(name="SynaBot", host=host, port=port)
        
        # 初始化机器人状态
        self.current_pose = RobotPose(
            position_mm=(0.0, 0.0, 0.0),
            orientation=(0, 0, 0, 1)
        )
        self.gripper_status = False  # False: 开启, True: 闭合
        self.held_object: Optional[str] = None  # 当前抓取的物体名称        
        
        # 初始化检测结果
        self.detection_results: ObjectsResult = ObjectsResult(objects={})
        self.preset_result = ObjectsResult(
                objects= {
                    "milk dragon": (100.0, 150.0, 10.0),
                    "white tray": (200.0, 250.0, 10.0),
                    "lemon": (300.0, 350.0, 10.0),
                    "apple": (350.0, 400.0, 10.0),
                    "banana": (400.0, 450.0, 10.0),
                }
            )

        self._register_tools()
    
    def _register_tools(self):
        """注册所有工具函数"""
        self.mcp.tool()(self.object_detector)
        # self.mcp.tool()(self.show_detection_result)
        self.mcp.tool()(self.get_ee_pose)
        self.mcp.tool()(self.moveL)
        self.mcp.tool()(self.grip)
        self.mcp.tool()(self.release)
    
    def object_detector(self) -> ObjectsResult:
        """物体位置检测器"""
        if random() < 0.5:  # 模拟 50% 的失败
            self.detection_results = ObjectsResult(objects={})
            return self.detection_results
        else:
            self.detection_results = self.preset_result.model_copy()
            return self.detection_results

    # def show_detection_result(self) -> str:
    #     """ 显示物体位置检测结果图片 """
    #     image_path: str = "tmp/test1.png"  
    #     img_cv = cv2.imread(image_path)
    #     if img_cv is not None:
    #         cv2.imshow("Detection Result", img_cv)
    #         # cv2.waitKey(0)
    #         # cv2.destroyAllWindows()
    #     return image_path

    def get_ee_pose(self) -> RobotPose:
        """获取机器人当前位姿"""
        return self.current_pose

    def moveL(self, target_object: str) -> RobotResponse:
        """移动机器人到指定物体的位置"""
        time.sleep(1)  # 模拟移动时间
        
        target_pos = self.detection_results.objects[target_object]
        
        self.current_pose = RobotPose(
            position_mm=target_pos,
            orientation=(0, 0, 0, 1)
        )
        
        # 模拟概率失败
        if random() < 0.2:
            print("移动失败")
            print("detection_results:")
            print(self.detection_results)
            print("preset_result:")
            print(self.preset_result)
            return RobotResponse(success=False, message="移动失败")
        else:
            self.preset_result.objects[target_object] = target_pos  # 模拟物体位置更新
            print("移动成功")
            print("detection_results:")
            print(self.detection_results)
            print("preset_result:")
            print(self.preset_result)
            return RobotResponse(success=True, message=f"移动到 ({self.current_pose})")

    def grip(self) -> RobotResponse:
        """夹取物体"""
        time.sleep(1)  # 模拟夹取时间
        
        self.gripper_status = True
        
        if random() < 0.2:  # 模拟 50% 的失败
            return RobotResponse(success=False, message="夹取失败")
        else:
            return RobotResponse(success=True, message="夹取成功")

    def release(self) -> RobotResponse:
        """释放物体"""
        time.sleep(1)  # 模拟释放时间
        
        self.gripper_status = False
        
        if random() < 0.2:  # 模拟 50% 的失败
            return RobotResponse(success=False, message="释放失败")
        else:
            return RobotResponse(success=True, message="释放成功")

    def run(self, transport: str = "streamable-http"):
        """运行服务器"""
        self.mcp.run(transport=transport)


if __name__ == "__main__":
    server = SynaBotServer()
    server.run()
