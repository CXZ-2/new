"""
数据集格式转换: 将Kaggle明火数据集转换为YOLOv8标准格式

YOLO格式目录结构:
datasets/fire/
├── images/
│   ├── train/
│   ├── val/
├── labels/
│   ├── train/
│   ├── val/

标签格式: class_id x_center y_center width height (归一化坐标)
class_id: 0=fire, 1=non-fire (但non-fire不含目标框，空标签文件即可)
"""
import os
import shutil
import random
from pathlib import Path
from PIL import Image

random.seed(42)


def convert_to_yolo_format(dataset_path: str, output_dir: str = "yolo/datasets/fire",
                           train_ratio: float = 0.8):
    """
    将Kaggle数据集转换为YOLOv8格式

    参数:
        dataset_path: Kaggle数据集原始路径
        output_dir: 输出的YOLO格式数据集路径
        train_ratio: 训练集比例
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 查找fire和non-fire图片目录
    dataset_path = Path(dataset_path)
    fire_images = list(dataset_path.rglob("fire/*.jpg")) + \
                   list(dataset_path.rglob("fire/*.jpeg")) + \
                   list(dataset_path.rglob("fire/*.png"))
    non_fire_images = list(dataset_path.rglob("non_fire/*.jpg")) + \
                      list(dataset_path.rglob("non_fire/*.jpeg")) + \
                      list(dataset_path.rglob("non_fire/*.png")) + \
                      list(dataset_path.rglob("non-fire/*.jpg")) + \
                      list(dataset_path.rglob("non-fire/*.jpeg")) + \
                      list(dataset_path.rglob("non-fire/*.png")) + \
                      list(dataset_path.rglob("no_fire/*.jpg")) + \
                      list(dataset_path.rglob("no_fire/*.jpeg")) + \
                      list(dataset_path.rglob("no_fire/*.png"))

    if not fire_images and not non_fire_images:
        # 尝试其他可能的目录结构
        all_images = list(dataset_path.rglob("*.jpg")) + \
                     list(dataset_path.rglob("*.jpeg")) + \
                     list(dataset_path.rglob("*.png"))
        print(f"未找到标准fire/non_fire目录，发现 {len(all_images)} 张图片")
        print("请手动确认数据集目录结构，或调整代码中的路径匹配规则")
        return

    print(f"发现明火图片: {len(fire_images)} 张")
    print(f"发现无火图片: {len(non_fire_images)} 张")

    # 随机打乱并划分训练/验证集
    random.shuffle(fire_images)
    random.shuffle(non_fire_images)

    fire_train = int(len(fire_images) * train_ratio)
    non_fire_train = int(len(non_fire_images) * train_ratio)

    splits = {
        "train": fire_images[:fire_train] + non_fire_images[:non_fire_train],
        "val": fire_images[fire_train:] + non_fire_images[non_fire_train:],
    }

    for split, images in splits.items():
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in images:
            # 复制图片
            new_name = img_path.name
            shutil.copy2(img_path, img_dir / new_name)

            # 生成标签文件
            label_path = lbl_dir / f"{img_path.stem}.txt"

            # 判断是否包含明火 (路径中带fire且不包含non的为明火)
            path_lower = str(img_path).lower()
            is_fire = "fire" in path_lower and "non" not in path_lower

            if is_fire:
                # 明火图片: 整个图片作为目标框 (class_id=0)
                # 对于整图标注，也可以选择不标注(因为YOLOv8分类也能工作)
                img = Image.open(img_path)
                w, h = img.size
                # 目标框占满整个图片的80%中心区域
                label = f"0 0.5 0.5 0.8 0.8"
                with open(label_path, "w") as f:
                    f.write(label + "\n")
            else:
                # 无火图片: 创建空标签文件
                with open(label_path, "w") as f:
                    pass

    print(f"数据集转换完成 -> {output_dir.absolute()}")
    print(f"  训练集: {len(splits['train'])} 张图片")
    print(f"  验证集: {len(splits['val'])} 张图片")

    # 生成 data.yaml 配置文件
    yaml_content = f"""
path: {output_dir.absolute()}
train: images/train
val: images/val
names:
  0: fire
nc: 1
"""
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content.strip())
    print(f"YAML配置文件: {yaml_path}")


if __name__ == "__main__":
    from download_data import download_fire_dataset
    dataset_path = download_fire_dataset()
    convert_to_yolo_format(dataset_path)
