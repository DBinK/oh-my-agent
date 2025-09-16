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

    # 给每个类别随机分配颜色
    colors = {
        name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for name in objs
    }

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
  "say": "目标物体为奶龙和白色托盘，需要将奶龙放入白色托盘中",
  "task": "抓取奶龙，移动到白色托盘上方，放下奶龙",
  "acts": [
    ["moveTo", "milk_dragon"],
    ["grip", "milk_dragon"],
    ["moveTo", "white_tray"],
    ["release"]
  ],
  "objs": {
    "milk_dragon": [504, 269, 594, 365],
    "white_tray": [614, 148, 857, 269]
  }
}
    """

    result_list = json.loads(result_json_fix)

    img = draw_bbox(image_path, result_list)
    cv2.imshow("YOLO", img)

    img = draw_bbox(image_path, result_list, None, 1000.0)
    cv2.imshow("YOLO_NORM", img)
    cv2.waitKey(0)
