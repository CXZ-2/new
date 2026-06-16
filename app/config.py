"""应用配置"""
import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 数据库配置 - 使用SQLite便于测试，生产环境替换为PostgreSQL
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/mine_safety.db")

# YOLO模型权重路径
YOLO_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    str(BASE_DIR / "yolo/runs/train/fire_detection/weights/best.pt")
)

# 图片上传目录
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# API配置
API_PREFIX = "/api/v1"
PAGE_SIZE_DEFAULT = 20
PAGE_SIZE_MAX = 100
