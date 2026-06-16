"""业务逻辑层 - 封装数据库CRUD操作，禁止在API接口中直接操作DBSession"""
import math
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from sqlmodel import Session, select, func

from app.models.inspection import InspectionRecord, Team, Area
from app.schemas.inspection import InspectionCreate
from app.config import UPLOAD_DIR, PAGE_SIZE_DEFAULT, PAGE_SIZE_MAX


class InspectionService:
    """安全检查记录业务服务

    所有数据库操作通过此Service完成，
    API层不直接操作DBSession。
    """

    def __init__(self, session: Session):
        self.session = session

    # ───────── 施工队管理 ─────────

    def create_team(self, name: str, team_code: str) -> Team:
        team = Team(name=name, team_code=team_code)
        self.session.add(team)
        self.session.commit()
        self.session.refresh(team)
        return team

    def get_team(self, team_id: int) -> Optional[Team]:
        return self.session.get(Team, team_id)

    def list_teams(self) -> list[Team]:
        return self.session.exec(select(Team)).all()

    # ───────── 采区管理 ─────────

    def create_area(self, name: str, area_code: str) -> Area:
        area = Area(name=name, area_code=area_code)
        self.session.add(area)
        self.session.commit()
        self.session.refresh(area)
        return area

    def get_area(self, area_id: int) -> Optional[Area]:
        return self.session.get(Area, area_id)

    def list_areas(self) -> list[Area]:
        return self.session.exec(select(Area)).all()

    # ───────── 检查记录管理 ─────────

    def save_photo(self, file_content: bytes, original_filename: str) -> str:
        """保存上传的照片到本地，返回存储路径"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        ext = Path(original_filename).suffix or ".jpg"
        save_name = f"{timestamp}{ext}"
        save_path = UPLOAD_DIR / save_name

        with open(save_path, "wb") as f:
            f.write(file_content)

        return str(save_path.relative_to(UPLOAD_DIR.parent))

    def create_inspection(
        self,
        data: InspectionCreate,
        photo_path: str,
        is_safe: bool,
        has_fire: bool = False,
        confidence: float = 0.0,
        detections: int = 0,
    ) -> InspectionRecord:
        """创建安全检查记录"""
        record = InspectionRecord(
            inspection_date=data.inspection_date,
            team_id=data.team_id,
            area_id=data.area_id,
            shift=data.shift,
            photo_path=photo_path,
            is_safe=is_safe,
            model_has_fire=has_fire,
            model_confidence=confidence,
            model_detections=detections,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def get_inspection(self, record_id: int) -> Optional[InspectionRecord]:
        """获取单条检查记录"""
        return self.session.get(InspectionRecord, record_id)

    def list_inspections(
        self,
        page: int = 1,
        page_size: int = PAGE_SIZE_DEFAULT,
        team_id: Optional[int] = None,
        area_id: Optional[int] = None,
        is_safe: Optional[bool] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> dict:
        """
        分页查询安全检查记录

        支持按施工队编号(team_id)、采区(area_id)筛选
        """
        page_size = min(page_size, PAGE_SIZE_MAX)

        # 构建查询条件
        stmt = select(InspectionRecord)
        count_stmt = select(func.count(InspectionRecord.id))

        if team_id is not None:
            stmt = stmt.where(InspectionRecord.team_id == team_id)
            count_stmt = count_stmt.where(InspectionRecord.team_id == team_id)

        if area_id is not None:
            stmt = stmt.where(InspectionRecord.area_id == area_id)
            count_stmt = count_stmt.where(InspectionRecord.area_id == area_id)

        if is_safe is not None:
            stmt = stmt.where(InspectionRecord.is_safe == is_safe)
            count_stmt = count_stmt.where(InspectionRecord.is_safe == is_safe)

        if start_date is not None:
            stmt = stmt.where(InspectionRecord.inspection_date >= start_date)
            count_stmt = count_stmt.where(InspectionRecord.inspection_date >= start_date)

        if end_date is not None:
            stmt = stmt.where(InspectionRecord.inspection_date <= end_date)
            count_stmt = count_stmt.where(InspectionRecord.inspection_date <= end_date)

        # 计算总记录数
        total = self.session.exec(count_stmt).one()

        # 分页
        total_pages = max(1, math.ceil(total / page_size))
        offset = (page - 1) * page_size
        stmt = stmt.order_by(InspectionRecord.created_at.desc()).offset(offset).limit(page_size)
        items = self.session.exec(stmt).all()

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "items": items,
        }

    def delete_inspection(self, record_id: int) -> bool:
        """删除检查记录及其关联的照片文件"""
        record = self.session.get(InspectionRecord, record_id)
        if record is None:
            return False

        # 删除照片文件
        photo_full_path = UPLOAD_DIR.parent / record.photo_path
        if photo_full_path.exists():
            photo_full_path.unlink()

        self.session.delete(record)
        self.session.commit()
        return True
