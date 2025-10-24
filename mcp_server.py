# mcp_sever.py

import time
from random import random

import cv2
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel


# 创建 MCP 服务器
mcp = FastMCP(name="SynaBot", host="0.0.0.0", port=5000)

# 添加机器人相关工具
class ObjectPosition(BaseModel):
    name : str
    position_mm: tuple[float, float, float]  # (x, y, z) in mm

class ObjectsResult(BaseModel):
    image_path: str = "tmp/test1.png"  # 检测结果的图片路径
    objects: list[ObjectPosition]
    
class RobotPose(BaseModel):
    position_mm: tuple[float, float, float]         # (x, y, z) in mm
    orientation: tuple[float, float, float, float]  # (x, y, z, w) in quaternion
    
class RobotResponse(BaseModel):
    success: bool
    message: str


# @mcp.tool()
# def take_photo() -> Image:
#     """拍照工具"""
#     # 模拟拍照时间
#     # time.sleep(1)
    
#     # 打开原始图片
#     pil_image = PILImage.open("tmp/test1.png")
    
#     # 调整图片大小，使用Image.LANCZOS进行高质量缩放
#     resized_image = pil_image.resize(size=(500, 500))
    
#     # 将图片保存到内存缓冲区
#     buffer = io.BytesIO()
#     resized_image.save(buffer, format="JPEG", quality=80, optimize=True)
#     buffer.seek(0)
    
#     return Image(data=buffer.getvalue(), format="jpeg")


@mcp.tool()
def object_detector() -> ObjectsResult:
    """物体位置检测器"""
    if random() < 0.5:  # 模拟 50% 的失败
        return ObjectsResult(objects=[])
    else:
        objects: list[ObjectPosition] = [  # 随机生成一些物体位置
            ObjectPosition(name="milk dragon", position_mm=(1000.0, 1500.0, 10.0)),
            ObjectPosition(name="white tray", position_mm=(2000.0, 2500.0, 10.0)),
            ObjectPosition(name="lemon", position_mm=(3000.0, 3500.0, 10.0)),
        ]
        return ObjectsResult(objects=objects)


@mcp.tool()
def show_detection_result() -> str:
    """ 显示物体位置检测结果图片 """
    image_path: str = "tmp/test1.png"  
    img_cv = cv2.imread(image_path)
    if img_cv is not None:
        cv2.imshow("Detection Result", img_cv)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
    return image_path


@mcp.tool()
def get_ee_pose() -> RobotPose:
    """获取机器人当前位姿"""
    pose = RobotPose(
        position_mm=(1500.0, 2000.0, 500.0),
        orientation=(0, 0, 0, 1)
    )
    return pose

@mcp.tool()
def moveL(target_pose: RobotPose) -> RobotResponse:
    """移动机器人到指定位姿"""
    
    time.sleep(1)  # 模拟移动时间
    
    # 模拟 50% 的失败
    if random() < 0.5:
        return RobotResponse(success=False, message="移动失败")
    else:
        return RobotResponse(success=True, message=f"移动到 ({target_pose})")

@mcp.tool()
def grip() -> RobotResponse:
    """夹取物体"""
    
    time.sleep(1)  # 模拟夹取时间
    
    if random() < 0.5:  # 模拟 50% 的失败
        return RobotResponse(success=False, message="夹取失败")
    else:
        return RobotResponse(success=True, message="夹取成功")

@mcp.tool()
def release() -> RobotResponse:
    """释放物体"""
    
    time.sleep(1)  # 模拟释放时间
    
    if random() < 0.5:  # 模拟 50% 的失败
        return RobotResponse(success=False, message="释放失败")
    else:
        return RobotResponse(success=True, message="释放成功")


if __name__ == "__main__":
    mcp.run(transport="streamable-http")