"""数据库连接与Session管理"""
from sqlmodel import SQLModel, Session, create_engine
from app.config import DATABASE_URL

# 创建数据库引擎
# connect_args 仅用于SQLite; PostgreSQL下需移除
connect_args = {}
if "sqlite" in DATABASE_URL:
    connect_args["check_same_thread"] = False

engine = create_engine(
    DATABASE_URL,
    echo=False,          # 生产环境设为False
    connect_args=connect_args,
)


def init_db():
    """初始化数据库 - 创建所有表"""
    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    """获取数据库会话"""
    with Session(engine) as session:
        yield session
