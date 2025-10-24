
from random import random
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP

# 创建 MCP 服务器
mcp = FastMCP("SynaBot")

# 添加机器人相关工具
class ObjectPosition(BaseModel):
    name : str
    position_mm: tuple[float, float, float]  # (x, y, z) in mm

class ObjectsResult(BaseModel):
    objects: list[ObjectPosition]
    
class RobotPose(BaseModel):
    position_mm: tuple[float, float, float]         # (x, y, z) in mm
    orientation: tuple[float, float, float, float]  # (x, y, z, w) in quaternion
    
class RobotResponse(BaseModel):
    success: bool
    message: str
    
@mcp.tool()
def object_detector() -> ObjectsResult:
    """物体位置检测器"""
    # 随机生成一些物体位置
    objects: list[ObjectPosition] = [
        ObjectPosition(name="milk dragon", position_mm=(1000.0, 1500.0, 10.0)),
        ObjectPosition(name="white tray", position_mm=(2000.0, 2500.0, 10.0)),
        ObjectPosition(name="lemon", position_mm=(3000.0, 3500.0, 10.0)),
    ]
    return ObjectsResult(objects=objects)

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
    # 模拟 50% 的失败
    if random() < 0.5:
        return RobotResponse(success=False, message="移动失败")
    else:
        return RobotResponse(success=True, message=f"移动到 ({target_pose})")

@mcp.tool()
def grip() -> RobotResponse:
    """夹取物体"""
    if random() < 0.5:
        return RobotResponse(success=False, message="夹取失败")
    else:
        return RobotResponse(success=True, message="夹取成功")

@mcp.tool()
def release() -> RobotResponse:
    """释放物体"""
    if random() < 0.5:
        return RobotResponse(success=False, message="释放失败")
    else:
        return RobotResponse(success=True, message="释放成功")


# 添加动态问候资源
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """获取个性化问候"""
    return f"你好，{name}！"


# 添加提示词
@mcp.prompt()
def greet_user(name: str, style: str = "friendly") -> str:
    """生成问候提示词"""
    styles = {
        "friendly": "请写一个温暖友好的问候",
        "formal": "请写一个正式专业的问候",
        "casual": "请写一个随意轻松的问候",
    }

    return f"{styles.get(style, styles['friendly'])}，对象是名为 {name} 的人。"

if __name__ == "__main__":
    mcp.run()