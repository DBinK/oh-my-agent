import cv2
import random
import json


def draw_bbox(
    image_path: str,
    data: dict,
    output_path: str | None = None,
    normalized_range: float | None = None,
):
    """
    根据输入字典绘制边界框 (适配objs字典格式)

    Args:
        image_path (str): 图片路径
        data (dict): 输入数据，包含objs字段
        output_path (str | None): 输出图片路径，如果为None则不保存
        normalized_range (float | None): 如果传入，则表示输入坐标是归一化的
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    h, w = img.shape[:2]
    objs = data.get("objs", {})
    acts = data.get("acts")

    # 显示任务描述（如果存在）
    if acts:
        # 设置字体和大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # 计算所有动作文本的尺寸
        act_strings = [f"{act[0]}({act[1]})" if len(act) > 1 else act[0] for act in acts]
        line_height = 30  # 每行的高度
        start_y = 30      # 起始Y位置
        
        # 绘制每个动作
        for i, act_str in enumerate(act_strings):
            y_position = start_y + i * line_height
            # 确保不会绘制超出图像范围
            if y_position < h - 10:
                cv2.putText(img, act_str, (10, y_position), font, font_scale, (255, 255, 255), thickness)
            else:
                # 如果超出范围，显示省略号
                cv2.putText(img, "...", (10, h - 10), font, font_scale, (255, 255, 255), thickness)
                break

    # 给每个类别随机分配颜色
    colors = {
        name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for name in objs
    }

    # 绘制边界框和标签
    for name, box in objs.items():
        x1, y1, x2, y2 = box
        if normalized_range:
            x1 = int(x1 * w / normalized_range)
            x2 = int(x2 * w / normalized_range)
            y1 = int(y1 * h / normalized_range)
            y2 = int(y2 * h / normalized_range)
        else:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # 保证边界框在图片范围内
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        color = colors[name]
        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # 绘制文字标签
        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(
            img, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1
        )
        cv2.putText(
            img, name, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

    if output_path:
        cv2.imwrite(output_path, img)
        print(f"结果已保存到: {output_path}")

    return img


if __name__ == "__main__":
    image_path = "tmp/test1.png"

    # 紧凑数组格式
    result_json_fix = """
{
    "say": "目标是将桌面上的水果放到键盘上，桌面上有香蕉、苹果、猕猴桃、柠檬等水果",
    "task": "依次抓取香蕉、苹果、猕猴桃、柠檬，并将它们放置到键盘上",
    "acts": [
        ["moveTo", "banana"],
        ["grip", "banana"],
        ["moveTo", "keyboard"],
        ["release"],
        ["moveTo", "apple"],
        ["grip", "apple"],
        ["moveTo", "keyboard"],
        ["release"],
        ["moveTo", "kiwi"],
        ["grip", "kiwi"],
        ["moveTo", "keyboard"],
        ["release"],
        ["moveTo", "lemon"],
        ["grip", "lemon"],
        ["moveTo", "keyboard"],
        ["release"]
    ],
    "objs": {
        "banana": [154, 229, 218, 320],
        "apple": [321, 255, 358, 311],
        "kiwi": [298, 295, 351, 350],
        "lemon": [215, 309, 264, 370],
        "keyboard": [197, 138, 402, 214]
    }
}

    """

    result_list = json.loads(result_json_fix)

    img = draw_bbox(image_path, result_list)
    cv2.imshow("YOLO", img)

    img = draw_bbox(image_path, result_list, None, 1000.0)
    cv2.imshow("YOLO_NORM", img)
    cv2.waitKey(0)
