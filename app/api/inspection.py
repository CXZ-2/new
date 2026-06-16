"""API接口层 - 安全检查相关路由

所有请求通过Service层间接处理，不直接操作数据库Session。
"""
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException, Query
from sqlmodel import Session

from app.database import get_session
from app.schemas.inspection import (
    InspectionCreate,
    InspectionResponse,
    InspectionListResponse,
    TeamCreate,
    TeamResponse,
    AreaCreate,
    AreaResponse,
)
from app.services.inspection import InspectionService
from yolo.inference import get_detector

router = APIRouter()


# ───────── 施工队管理 ─────────

@router.post("/teams", response_model=TeamResponse, status_code=201)
def create_team(data: TeamCreate, db: Session = Depends(get_session)):
    """创建施工队"""
    service = InspectionService(db)
    return service.create_team(data.name, data.team_code)


@router.get("/teams", response_model=list[TeamResponse])
def list_teams(db: Session = Depends(get_session)):
    """获取所有施工队"""
    service = InspectionService(db)
    return service.list_teams()


# ───────── 采区管理 ─────────

@router.post("/areas", response_model=AreaResponse, status_code=201)
def create_area(data: AreaCreate, db: Session = Depends(get_session)):
    """创建采区"""
    service = InspectionService(db)
    return service.create_area(data.name, data.area_code)


@router.get("/areas", response_model=list[AreaResponse])
def list_areas(db: Session = Depends(get_session)):
    """获取所有采区"""
    service = InspectionService(db)
    return service.list_areas()


# ───────── 安全检查记录 ─────────

@router.post("/inspections", response_model=InspectionResponse, status_code=201)
async def create_inspection(
    inspection_date: str = Form(..., description="检查日期 (YYYY-MM-DD)"),
    team_id: int = Form(..., gt=0, description="施工队ID"),
    area_id: int = Form(..., gt=0, description="采区ID"),
    shift: str = Form(..., max_length=20, description="班组"),
    photo: UploadFile = File(..., description="现场照片"),
    db: Session = Depends(get_session),
):
    """
    创建安全检查记录

    接收表单数据 + 图片文件，调用YOLOv8模型进行明火检测，
    将检测结果和业务数据一并存入数据库。

    流程:
    1. 验证施工队/采区是否存在
    2. 保存上传的照片
    3. 调用YOLOv8进行明火检测
    4. 将结果写入数据库
    5. 返回完整记录
    """
    service = InspectionService(db)

    # 验证外键关联
    if not service.get_team(team_id):
        raise HTTPException(status_code=404, detail=f"施工队不存在: team_id={team_id}")
    if not service.get_area(area_id):
        raise HTTPException(status_code=404, detail=f"采区不存在: area_id={area_id}")

    # 解析日期
    try:
        parsed_date = date.fromisoformat(inspection_date)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"日期格式错误: {inspection_date}，应为 YYYY-MM-DD")

    # 保存照片
    photo_content = await photo.read()
    photo_path = service.save_photo(photo_content, photo.filename or "photo.jpg")

    # 调用模型进行明火检测
    detector = get_detector()
    result = detector.predict_from_bytes(photo_content)

    # 创建检查记录
    data = InspectionCreate(
        inspection_date=parsed_date,
        team_id=team_id,
        area_id=area_id,
        shift=shift,
    )
    record = service.create_inspection(
        data=data,
        photo_path=photo_path,
        is_safe=result.is_safe,
        has_fire=result.has_fire,
        confidence=result.confidence,
        detections=result.num_detections,
    )
    return record


@router.get("/inspections", response_model=InspectionListResponse)
def list_inspections(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页条数"),
    team_id: Optional[int] = Query(None, description="按施工队ID筛选"),
    area_id: Optional[int] = Query(None, description="按采区ID筛选"),
    is_safe: Optional[bool] = Query(None, description="按安全状态筛选"),
    start_date: Optional[date] = Query(None, description="开始日期 YYYY-MM-DD"),
    end_date: Optional[date] = Query(None, description="结束日期 YYYY-MM-DD"),
    db: Session = Depends(get_session),
):
    """
    分页查询安全检查记录

    支持筛选条件:
    - team_id: 施工队编号
    - area_id: 采区编号
    - is_safe: 安全状态
    - start_date / end_date: 日期范围
    """
    service = InspectionService(db)
    result = service.list_inspections(
        page=page,
        page_size=page_size,
        team_id=team_id,
        area_id=area_id,
        is_safe=is_safe,
        start_date=start_date,
        end_date=end_date,
    )
    return InspectionListResponse(**result)


@router.get("/inspections/{record_id}", response_model=InspectionResponse)
def get_inspection(record_id: int, db: Session = Depends(get_session)):
    """获取单条检查记录详情"""
    service = InspectionService(db)
    record = service.get_inspection(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"检查记录不存在: id={record_id}")
    return record


@router.delete("/inspections/{record_id}", status_code=204)
def delete_inspection(record_id: int, db: Session = Depends(get_session)):
    """删除检查记录"""
    service = InspectionService(db)
    if not service.delete_inspection(record_id):
        raise HTTPException(status_code=404, detail=f"检查记录不存在: id={record_id}")
