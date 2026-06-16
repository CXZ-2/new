# 矿井安全检查系统

基于 YOLOv8 的明火检测与安全检查记录管理系统。

## 功能

- **YOLOv8 迁移学习**: 在 Kaggle 明火数据集上进行迁移学习，识别图片中的违规行为(明火)
- **安全检查记录 API**: FastAPI 后端，支持照片上传、模型推理、结果存储与分页查询
- **数据库管理**: SQLModel 定义数据表，Service 层封装业务逻辑

## 项目结构

```
├── app/                    # FastAPI 后端
│   ├── main.py             # 应用入口
│   ├── config.py           # 配置
│   ├── database.py         # 数据库连接
│   ├── models/             # SQLModel 数据表定义
│   ├── schemas/            # Pydantic 请求/响应模型
│   ├── services/           # 业务逻辑层 (CRUD)
│   └── api/                # API 路由
├── yolo/                   # YOLOv8 模块
│   ├── download_data.py    # Kaggle 数据集下载
│   ├── dataset.py          # 数据集格式转换 (→ YOLO)
│   ├── train.py            # 迁移学习训练脚本
│   └── inference.py        # 推理脚本
├── static/uploads/         # 上传照片存储
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据集与训练模型 (可选)

```bash
# 下载训练数据并转换为YOLO格式
python -m yolo.dataset

# 迁移学习训练 (使用预训练yolov8n.pt)
python -m yolo.train --epochs 30 --batch 16
```

### 3. 启动 API 服务

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. 访问 API 文档

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 接口

### 施工队管理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/teams` | 创建施工队 |
| GET  | `/api/v1/teams` | 获取所有施工队 |

### 采区管理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/areas` | 创建采区 |
| GET  | `/api/v1/areas` | 获取所有采区 |

### 安全检查记录

| 方法 | 路径 | 说明 |
|------|------|------|
| POST   | `/api/v1/inspections`           | 上传照片并创建检查记录 |
| GET    | `/api/v1/inspections`           | 分页查询检查记录 |
| GET    | `/api/v1/inspections/{id}`      | 获取单条记录详情 |
| DELETE | `/api/v1/inspections/{id}`      | 删除检查记录 |

### 使用示例

```bash
# 1. 创建施工队
curl -X POST http://localhost:8000/api/v1/teams \
  -H "Content-Type: application/json" \
  -d '{"name": "一队", "team_code": "T001"}'

# 2. 创建采区
curl -X POST http://localhost:8000/api/v1/areas \
  -H "Content-Type: application/json" \
  -d '{"name": "A采区", "area_code": "A01"}'

# 3. 上传照片进行检查
curl -X POST http://localhost:8000/api/v1/inspections \
  -F "inspection_date=2025-01-15" \
  -F "team_id=1" \
  -F "area_id=1" \
  -F "shift=白班" \
  -F "photo=@test_fire.jpg"

# 4. 分页查询 (按施工队和采区筛选)
curl "http://localhost:8000/api/v1/inspections?page=1&page_size=20&team_id=1"

# 5. 筛选危险记录
curl "http://localhost:8000/api/v1/inspections?is_safe=false"
```

## 运行测试

```bash
# 安装测试依赖
pip install pytest httpx

# 运行测试
pytest -v

# 带覆盖率
pytest --cov=app --cov=yolo -v
```

## 技术栈

- **YOLOv8** (Ultralytics): 目标检测模型
- **FastAPI**: Web 框架
- **SQLModel**: ORM (SQLAlchemy + Pydantic)
- **SQLite**: 数据库 (可替换为 PostgreSQL)
