import os
import kagglehub

def download_fire_dataset():
    """
    下载Kaggle明火数据集
    数据集来源: phylake1337/fire-dataset
    包含 fire(明火) 和 non-fire(无火) 两类图片
    """
    print("正在从Kaggle下载明火数据集...")
    path = kagglehub.dataset_download("phylake1337/fire-dataset")
    print(f"数据集下载路径: {path}")

    # 列出数据集目录结构
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        if level < 3:
            subindent = ' ' * 2 * (level + 1)
            # 只显示前5个文件
            for f in files[:5]:
                print(f"{subindent}{f}")
            if len(files) > 5:
                print(f"{subindent}... 共 {len(files)} 个文件")

    return path

if __name__ == "__main__":
    download_fire_dataset()
