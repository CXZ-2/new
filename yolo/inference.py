"""
YOLOv8 明火检测推理脚本

根据传入的图片，使用训练好的模型进行推理，返回是否检测到明火。
当 ultralytics 未安装或训练权重不存在时，自动降级为基于颜色特征的火焰检测。

用法:
    from yolo.inference import FireDetector
    detector = FireDetector("path/to/best.pt")
    result = detector.predict("path/to/image.jpg")
"""
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

# 尝试导入 ultralytics (可选依赖)
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False


@dataclass
class DetectionResult:
    """检测结果数据类"""
    is_safe: bool              # 是否安全 (True=无火/安全, False=有火/危险)
    has_fire: bool             # 是否检测到明火
    confidence: float = 0.0    # 检测置信度 (0.0~1.0)
    num_detections: int = 0    # 检测到的火焰目标数量
    detections: list = field(default_factory=list)  # 详细检测列表 [bbox, conf]
    method: str = "yolo"       # 检测方法 (yolo / color_fallback)


def _color_based_fire_detect(image_path: str, conf_threshold: float = 0.3) -> DetectionResult:
    """
    基于颜色特征的火焰检测 (fallback方案)

    检测原理: 计算图片中红色/橙色像素的占比，
    若超过阈值则判定为有火焰。
    """
    if not os.path.exists(image_path):
        return DetectionResult(is_safe=True, has_fire=False, method="color_fallback")

    try:
        img = Image.open(image_path).convert("RGB")
        pixels = np.array(img, dtype=np.float32)

        # 红色火焰: R > 150, R > G*1.5, R > B*1.5
        r, g, b = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
        fire_mask = (r > 150) & (r > g * 1.5) & (r > b * 1.5)

        # 橙色火焰: R > 180, G > 60, B < 80
        orange_mask = (r > 180) & (g > 60) & (b < 80)

        combined_mask = fire_mask | orange_mask
        fire_ratio = combined_mask.sum() / combined_mask.size

        has_fire = fire_ratio > conf_threshold
        confidence = min(fire_ratio * 3, 1.0)  # 将比例映射到置信度

        # 找到火焰区域的包围盒
        detections = []
        if has_fire:
            fire_rows = np.any(combined_mask, axis=1)
            fire_cols = np.any(combined_mask, axis=0)
            if fire_rows.any() and fire_cols.any():
                y1, y2 = np.where(fire_rows)[0][[0, -1]]
                x1, x2 = np.where(fire_cols)[0][[0, -1]]
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": round(confidence, 4),
                    "class": "fire"
                })

        return DetectionResult(
            is_safe=not has_fire,
            has_fire=has_fire,
            confidence=round(confidence, 4),
            num_detections=len(detections),
            detections=detections,
            method="color_fallback",
        )
    except Exception:
        return DetectionResult(is_safe=True, has_fire=False, method="color_fallback")


class FireDetector:
    """明火检测器

    优先使用YOLOv8模型；若不可用则降级为颜色特征检测。

    使用示例:
        detector = FireDetector("path/to/best.pt")
        result = detector.predict("path/to/photo.jpg")
        if not result.is_safe:
            print(f"检测到明火! 置信度: {result.confidence:.2%}")
    """

    DEFAULT_WEIGHTS = "yolo/runs/train/fire_detection/weights/best.pt"

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.25):
        self.model_path = model_path or self.DEFAULT_WEIGHTS
        self.conf_threshold = conf_threshold
        self._model = None
        self._use_yolo = False

        # 检查是否可以使用 YOLO
        if _HAS_YOLO and os.path.exists(self.model_path):
            self._use_yolo = True
            print(f"[模型] 使用YOLOv8: {self.model_path}")
        elif _HAS_YOLO and os.path.exists("yolov8n.pt"):
            self.model_path = "yolov8n.pt"
            self._use_yolo = True
            print("[模型] 使用预训练yolov8n.pt (建议先运行训练脚本)")
        else:
            print("[模型] YOLO不可用，使用颜色特征检测 (安装ultralytics并训练后可替换)")

    @property
    def model(self):
        """延迟加载 YOLO 模型"""
        if self._model is None and self._use_yolo:
            print(f"正在加载模型: {self.model_path}")
            self._model = YOLO(self.model_path)
            print("模型加载完成")
        return self._model

    def predict(self, image_path: str) -> DetectionResult:
        """对单张图片进行明火检测"""
        if not os.path.exists(image_path):
            return DetectionResult(is_safe=True, has_fire=False)

        # 验证图片可读
        try:
            img = Image.open(image_path)
            img.verify()
        except Exception:
            return DetectionResult(is_safe=True, has_fire=False)

        # YOLO推理
        if self._use_yolo and self.model is not None:
            return self._yolo_predict(image_path)
        else:
            return _color_based_fire_detect(image_path, self.conf_threshold)

    def _yolo_predict(self, image_path: str) -> DetectionResult:
        """YOLOv8 推理"""
        results = self.model.predict(
            source=image_path, conf=self.conf_threshold, verbose=False,
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
                    xyxy = box.xyxy[0].tolist()
                    if cls_id == 0:  # class_id=0 = fire
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
            method="yolo",
        )

    def predict_from_bytes(self, image_bytes: bytes) -> DetectionResult:
        """从图片字节流进行推理 (用于API接口)"""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            tmp_path = f.name
        try:
            return self.predict(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# 全局检测器单例
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
        sys.exit(1)
    detector = FireDetector()
    result = detector.predict(sys.argv[1])
    print(f"检测方法: {result.method}")
    print(f"安全状态: {'安全' if result.is_safe else '危险! 检测到明火'}")
    print(f"是否检测到火: {result.has_fire}")
    print(f"置信度: {result.confidence:.4f}")
    print(f"检测目标数: {result.num_detections}")
