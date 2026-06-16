"""FastAPI 应用入口

启动命令:
    uvicorn app.main:app --reload

API文档:
    http://localhost:8000/docs     (Swagger UI)
    http://localhost:8000/redoc    (ReDoc)
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import API_PREFIX
from app.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时: 初始化数据库表
    print("正在初始化数据库...")
    init_db()
    print("数据库初始化完成")
    yield
    # 关闭时: 清理资源
    print("应用正在关闭...")


app = FastAPI(
    title="矿井安全检查系统",
    description="基于YOLOv8的明火检测与安全检查记录管理系统",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
from app.api.inspection import router as inspection_router
app.include_router(inspection_router, prefix=API_PREFIX, tags=["安全检查"])


@app.get("/", tags=["健康检查"])
def root():
    """服务健康检查"""
    return {
        "status": "running",
        "service": "矿井安全检查系统",
        "version": "1.0.0",
    }


@app.get("/health", tags=["健康检查"])
def health_check():
    """健康检查端点"""
    return {"status": "healthy"}
