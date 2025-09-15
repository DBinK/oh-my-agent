import cv2
import random
def draw_yolo_style(image_path: str, detections: list, output_path: str | None = None, normalized_range: float | None = None):
    """
    以YOLO风格绘制检测结果

    Args:
        image_path (str): 原始图片路径
        detections (list): 检测结果列表
        output_path (str): 输出图片路径，默认为None则显示图片
        normalized_range (float | None): 归一化范围，如果为None则使用绝对坐标，否则使用归一化坐标

    Returns:
        None
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    img_height, img_width = img.shape[:2]
    
    # 生成随机颜色用于不同类别
    colors = {}
    for detection in detections:
        name = detection["name"]
        if name not in colors:
            colors[name] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )

    # 在图片上绘制边界框和标签
    for detection in detections:
        name = detection["name"]
        bbox = detection["bbox"]
        
        # 根据normalized_range参数决定是否进行坐标转换
        if normalized_range is not None:
            # 将归一化坐标转换为实际图像坐标
            x1_norm, y1_norm, x2_norm, y2_norm = bbox
            x1 = int(x1_norm * img_width / normalized_range)
            y1 = int(y1_norm * img_height / normalized_range)
            x2 = int(x2_norm * img_width / normalized_range)
            y2 = int(y2_norm * img_height / normalized_range)
        else:
            # 使用绝对坐标
            x1, y1, x2, y2 = bbox

        # 确保坐标在图片范围内
        x1 = max(0, min(x1, img_width))
        x2 = max(0, min(x2, img_width))
        y1 = max(0, min(y1, img_height))
        y2 = max(0, min(y2, img_height))

        # 绘制边界框
        color = colors[name]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # 绘制标签背景
        label_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(
            img,
            (int(x1), int(y1) - label_size[1] - 10),
            (int(x1) + label_size[0], int(y1)),
            color,
            -1,
        )

        # 绘制标签文字
        cv2.putText(
            img,
            name,
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    # 保存或显示结果
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"结果已保存到: {output_path}")
    else:
        cv2.imshow("YOLO Style Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":

    import json

    image_path = "tmp/test.png"

    # result_json_fix = """
    # [
    #     {
    #         "name": "apple",
    #         "bbox": [346, 622, 404, 663]
    #     },
    #     {
    #         "name": "strawberry",
    #         "bbox": [587, 273, 638, 314]
    #     }
    # ]
    # """
    result_json_fix = """
        [{"bbox":[150,180,350,380],"name":"milk"}]
    """

    result_dict = json.loads(result_json_fix)

    draw_yolo_style(image_path, result_dict)

    draw_yolo_style(image_path, result_dict, None, 1000.0)






