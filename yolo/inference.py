"""
YOLOv8 明火检测推理脚本

根据传入的图片路径，使用训练好的模型进行推理，返回是否检测到明火。

用法:
    from yolo.inference import FireDetector
    detector = FireDetector("yolo/runs/train/fire_detection/weights/best.pt")
    result = detector.predict("path/to/image.jpg")
    # result = {"is_safe": False, "has_fire": True, "confidence": 0.95, ...}
"""
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("请先安装ultralytics: pip install ultralytics")

import numpy as np
from PIL import Image


@dataclass
class DetectionResult:
    """检测结果数据类"""
    is_safe: bool              # 是否安全 (True=无火/安全, False=有火/危险)
    has_fire: bool             # 是否检测到明火
    confidence: float = 0.0    # 检测置信度 (0.0~1.0)
    num_detections: int = 0    # 检测到的火焰目标数量
    detections: list = field(default_factory=list)  # 详细检测列表 [bbox, conf]


class FireDetector:
    """明火检测器

    使用YOLOv8模型检测图片中是否存在明火。

    使用示例:
        detector = FireDetector("path/to/best.pt")
        result = detector.predict("path/to/photo.jpg")
        if not result.is_safe:
            print(f"检测到明火! 置信度: {result.confidence:.2%}")
    """

    # 默认权重路径
    DEFAULT_WEIGHTS = "yolo/runs/train/fire_detection/weights/best.pt"

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.25):
        """
        初始化检测器

        参数:
            model_path: 训练好的模型权重路径，为None时使用默认路径
            conf_threshold: 检测置信度阈值，低于此值的目标将被忽略
        """
        self.model_path = model_path or self.DEFAULT_WEIGHTS

        if not os.path.exists(self.model_path):
            # 若训练权重不存在，尝试直接使用预训练权重
            print(f"[警告] 未找到训练权重 {self.model_path}")
            print(f"[警告] 将使用默认的yolov8n.pt预训练权重进行推理(效果可能不佳)")
            self.model_path = "yolov8n.pt"

        self.conf_threshold = conf_threshold
        self._model: Optional[YOLO] = None

    @property
    def model(self) -> YOLO:
        """延迟加载模型(单例)"""
        if self._model is None:
            print(f"正在加载模型: {self.model_path}")
            self._model = YOLO(self.model_path)
            print("模型加载完成")
        return self._model

    def predict(self, image_path: str) -> DetectionResult:
        """对单张图片进行明火检测

        参数:
            image_path: 图片文件路径

        返回:
            DetectionResult: 包含is_safe和检测详情的结构体
        """
        if not os.path.exists(image_path):
            return DetectionResult(
                is_safe=True,
                has_fire=False,
                confidence=0.0,
                num_detections=0,
                detections=[]
            )

        # 验证图片可读
        try:
            img = Image.open(image_path)
            img.verify()
        except Exception:
            return DetectionResult(
                is_safe=True,
                has_fire=False,
                confidence=0.0,
                num_detections=0,
                detections=[]
            )

        # 执行推理
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            verbose=False,
        )

        detections = []
        max_conf = 0.0
        has_fire = False

        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                    # class_id=0 表示 fire
                    if cls_id == 0:
                        has_fire = True
                        max_conf = max(max_conf, conf)
                        detections.append({
                            "bbox": [round(x, 2) for x in xyxy],
                            "confidence": round(conf, 4),
                            "class": "fire"
                        })

        return DetectionResult(
            is_safe=not has_fire,
            has_fire=has_fire,
            confidence=round(max_conf, 4),
            num_detections=len(detections),
            detections=detections,
        )

    def predict_from_bytes(self, image_bytes: bytes) -> DetectionResult:
        """从图片字节流进行推理 (用于API接口)

        参数:
            image_bytes: 图片的字节流数据

        返回:
            DetectionResult
        """
        import tempfile

        # 写入临时文件
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            tmp_path = f.name

        try:
            result = self.predict(tmp_path)
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return result


# 全局检测器单例 (在应用启动时初始化)
_detector: Optional[FireDetector] = None


def get_detector(model_path: Optional[str] = None) -> FireDetector:
    """获取全局检测器单例"""
    global _detector
    if _detector is None:
        _detector = FireDetector(model_path)
    return _detector


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python -m yolo.inference <图片路径>")
        print("示例: python -m yolo.inference test_fire.jpg")
        sys.exit(1)

    detector = FireDetector()
    result = detector.predict(sys.argv[1])

    print(f"\n检测结果:")
    print(f"  安全状态: {'安全' if result.is_safe else '危险! 检测到明火'}")
    print(f"  是否检测到火: {result.has_fire}")
    print(f"  最高置信度: {result.confidence:.4f}")
    print(f"  检测目标数: {result.num_detections}")
    if result.detections:
        for i, det in enumerate(result.detections, 1):
            print(f"  目标{i}: bbox={det['bbox']}, conf={det['confidence']:.4f}")
