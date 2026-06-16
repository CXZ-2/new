"""数据库模型定义 (SQLModel)"""
from datetime import datetime, date
from typing import Optional
from sqlmodel import SQLModel, Field, Relationship


class Team(SQLModel, table=True):
    """
    施工队表
    管理字段: created_at 用于记录创建时间
    """
    __tablename__ = "teams"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=100, index=True, description="施工队名称")
    team_code: str = Field(max_length=20, unique=True, index=True, description="施工队编号")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")

    # 关联关系
    inspections: list["InspectionRecord"] = Relationship(back_populates="team")


class Area(SQLModel, table=True):
    """
    采区表
    管理字段: created_at 用于记录创建时间
    """
    __tablename__ = "areas"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=100, index=True, description="采区名称")
    area_code: str = Field(max_length=20, unique=True, index=True, description="采区编号")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")

    # 关联关系
    inspections: list["InspectionRecord"] = Relationship(back_populates="area")


class InspectionRecord(SQLModel, table=True):
    """
    安全检查记录表

    业务字段:
        - inspection_date: 检查日期 (前端传入)
        - team_id: 施工队ID (外键关联)
        - area_id: 采区ID (外键关联)
        - shift: 班组
        - photo_path: 上传照片的存储路径
        - is_safe: 模型判断结果 (True=安全, False=有火灾风险)

    管理字段:
        - id: 自增主键
        - created_at: 创建时间
        - updated_at: 更新时间
        - model_confidence: 模型检测置信度
        - model_has_fire: 模型是否检测到明火
        - model_detections: 模型检测到的目标数量
    """
    __tablename__ = "inspection_records"

    # 主键
    id: Optional[int] = Field(default=None, primary_key=True)

    # 业务字段
    inspection_date: date = Field(index=True, description="检查日期")
    team_id: int = Field(foreign_key="teams.id", index=True, description="施工队ID")
    area_id: int = Field(foreign_key="areas.id", index=True, description="采区ID")
    shift: str = Field(max_length=20, index=True, description="班组")
    photo_path: str = Field(max_length=500, description="照片存储路径")
    is_safe: bool = Field(index=True, description="是否安全 (True=安全)")

    # 模型检测详情
    model_confidence: float = Field(default=0.0, description="模型检测置信度")
    model_has_fire: bool = Field(default=False, description="模型是否检测到明火")
    model_detections: int = Field(default=0, description="检测到的目标数量")

    # 管理字段
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新时间")

    # 关联关系
    team: Team = Relationship(back_populates="inspections")
    area: Area = Relationship(back_populates="inspections")
