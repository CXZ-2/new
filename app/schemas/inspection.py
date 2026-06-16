"""Pydantic请求/响应模型"""
from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel, Field


# ─────────────────── 施工队/采区 Schema ───────────────────

class TeamCreate(BaseModel):
    """创建施工队请求"""
    name: str = Field(..., max_length=100, description="施工队名称")
    team_code: str = Field(..., max_length=20, description="施工队编号")


class TeamResponse(BaseModel):
    """施工队响应"""
    id: int
    name: str
    team_code: str
    created_at: datetime

    class Config:
        from_attributes = True


class AreaCreate(BaseModel):
    """创建采区请求"""
    name: str = Field(..., max_length=100, description="采区名称")
    area_code: str = Field(..., max_length=20, description="采区编号")


class AreaResponse(BaseModel):
    """采区响应"""
    id: int
    name: str
    area_code: str
    created_at: datetime

    class Config:
        from_attributes = True


# ─────────────────── 检查记录 Schema ───────────────────

class InspectionCreate(BaseModel):
    """创建检查记录请求 (不含photo，photo通过FormData上传)"""
    inspection_date: date = Field(..., description="检查日期 (YYYY-MM-DD)")
    team_id: int = Field(..., gt=0, description="施工队ID")
    area_id: int = Field(..., gt=0, description="采区ID")
    shift: str = Field(..., max_length=20, description="班组")


class InspectionResponse(BaseModel):
    """单条检查记录响应"""
    id: int
    inspection_date: date
    team_id: int
    area_id: int
    shift: str
    photo_path: str
    is_safe: bool
    model_confidence: float
    model_has_fire: bool
    model_detections: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class InspectionListResponse(BaseModel):
    """分页查询响应"""
    total: int = Field(..., description="总记录数")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页条数")
    total_pages: int = Field(..., description="总页数")
    items: list[InspectionResponse] = Field(..., description="当前页记录列表")
