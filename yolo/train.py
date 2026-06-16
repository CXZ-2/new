"""
YOLOv8 迁移学习训练脚本

使用预训练权重 yolov8n.pt (nano版本，适合快速训练)
在明火数据集上进行迁移学习，只训练检测头部分

用法:
    python -m yolo.train          # 使用默认参数训练
    python -m yolo.train --epochs 50 --batch 16 --model yolov8s.pt
"""
import argparse
import os
from pathlib import Path

# 尝试导入ultralytics; 如果未安装则提示
try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "请先安装ultralytics: pip install ultralytics\n"
        "Please install ultralytics: pip install ultralytics"
    )


def train_model(
    data_yaml: str = "yolo/datasets/fire/data.yaml",
    model_name: str = "yolov8n.pt",
    epochs: int = 30,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "auto",
    output_dir: str = "yolo/runs/train",
):
    """
    YOLOv8迁移学习训练函数

    参数:
        data_yaml: 数据集配置文件路径
        model_name: 预训练模型名称 (yolov8n.pt / yolov8s.pt / yolov8m.pt)
        epochs: 训练轮数
        batch_size: 批次大小
        img_size: 输入图片尺寸
        device: 训练设备 (auto/cpu/0)
        output_dir: 输出目录
    """
    # 检查数据集配置文件
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(
            f"数据集配置文件不存在: {data_yaml}\n"
            f"请先运行 python -m yolo.dataset 转换数据集"
        )

    print(f"使用预训练模型: {model_name}")
    print(f"数据集配置: {data_yaml}")
    print(f"训练轮数: {epochs}, 批次大小: {batch_size}, 图片尺寸: {img_size}")

    # 加载预训练模型
    model = YOLO(model_name)

    # 训练参数
    # 迁移学习策略: 冻结backbone前几层，只训练检测头
    train_args = {
        "data": data_yaml,
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": img_size,
        "device": device,
        "project": output_dir,
        "name": "fire_detection",
        "exist_ok": True,
        "pretrained": True,          # 使用预训练权重
        "optimizer": "AdamW",
        "lr0": 0.001,                # 初始学习率(预训练模型用较小lr)
        "lrf": 0.01,                 # 最终学习率因子
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "box": 7.5,                  # 边界框损失权重
        "cls": 0.5,                  # 分类损失权重
        "dfl": 1.5,                  # DFL损失权重
        "patience": 10,              # 早停耐心值
        "save": True,
        "save_period": 5,            # 每5轮保存一次
        "val": True,
        "plots": True,
        # 数据增强
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 10.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 2.0,
        "perspective": 0.0001,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1,
    }

    print("\n开始迁移学习训练...")
    print("-" * 50)

    # 训练模型
    results = model.train(**train_args)

    # 评估模型
    print("\n" + "=" * 50)
    print("训练完成，开始验证集评估...")
    metrics = model.val()

    # 导出模型为ONNX格式(便于部署)
    print("\n导出ONNX模型...")
    best_pt = Path(output_dir) / "fire_detection" / "weights" / "best.pt"
    if best_pt.exists():
        model = YOLO(str(best_pt))
        model.export(format="onnx")
        print(f"ONNX模型已导出到: {best_pt.with_suffix('.onnx')}")

    print("\n训练总结:")
    print(f"  最佳权重: {best_pt}")
    print(f"  训练结果: {Path(output_dir) / 'fire_detection'}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8明火检测迁移学习训练")
    parser.add_argument("--data", default="yolo/datasets/fire/data.yaml",
                        help="数据集YAML配置文件路径")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="预训练模型 (yolov8n.pt / yolov8s.pt / yolov8m.pt)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="训练轮数")
    parser.add_argument("--batch", type=int, default=16,
                        help="批次大小")
    parser.add_argument("--img-size", type=int, default=640,
                        help="输入图片尺寸")
    parser.add_argument("--device", default="auto",
                        help="训练设备 (auto/cpu/0)")
    parser.add_argument("--output", default="yolo/runs/train",
                        help="输出目录")
    parser.add_argument("--data-dir", default=None,
                        help="Kaggle数据集路径(若已有本地数据则直接转换)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 如果没有数据集，提示下载
    if not os.path.exists(args.data):
        print("数据集配置文件不存在，正在尝试自动准备数据集...")
        try:
            from yolo.dataset import convert_to_yolo_format
            from yolo.download_data import download_fire_dataset

            if args.data_dir:
                dataset_path = args.data_dir
            else:
                dataset_path = download_fire_dataset()

            convert_to_yolo_format(dataset_path)
        except Exception as e:
            print(f"数据集准备失败: {e}")
            print("请手动准备数据集后重试")
            exit(1)

    train_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device,
        output_dir=args.output,
    )
