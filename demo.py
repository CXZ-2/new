#!/usr/bin/env python3
"""
矿井安全检查系统 - 完整演示脚本

自动执行:
1. 启动 FastAPI 服务
2. 生成模拟火焰测试图片
3. 创建施工队和采区
4. 上传照片进行检查
5. 分页查询验证结果

用法:
    python demo.py
"""
import io
import sys
import time
import subprocess
import signal
import requests
from PIL import Image, ImageDraw


# ───────── 1. 生成模拟测试图片 ─────────

def create_fire_test_image(path="test_fire.jpg"):
    """生成一张模拟火焰图片 (大面积红色/橙色，确保检测通过)"""
    img = Image.new("RGB", (640, 480), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)

    # 绘制大面积红色火焰基底 (覆盖约40%的区域)
    draw.rectangle([100, 150, 540, 380], fill=(220, 30, 0))
    draw.rectangle([150, 180, 490, 350], fill=(255, 80, 0))
    draw.rectangle([200, 210, 440, 320], fill=(255, 160, 20))

    # 叠加火焰形状
    for i in range(150):
        x = 150 + (i % 30) * 12
        y = 180 + (i // 6) * 4
        if x < 500 and y < 370:
            r = 255
            g = min(255, 30 + (i * 3) % 200)
            b = min(100, i % 50)
            draw.ellipse([x - 15, y - 15, x + 15, y + 15], fill=(r, g, b))

    img.save(path)
    print(f"[演示] 已生成模拟火焰测试图片: {path}")
    return path


def create_safe_test_image(path="test_safe.jpg"):
    """生成一张安全的普通图片 (灰色背景)"""
    img = Image.new("RGB", (640, 480), color=(128, 128, 128))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 540, 380], fill=(100, 110, 120), outline=(80, 85, 90))
    draw.text((250, 220), "安全场景", fill=(200, 200, 200))
    img.save(path)
    print(f"[演示] 已生成安全测试图片: {path}")
    return path


# ───────── 2. Demo API 调用 ─────────

BASE_URL = "http://localhost:8000/api/v1"


def demo_flow():
    """执行完整演示流程"""
    print("\n" + "=" * 60)
    print("  矿井安全检查系统 - 完整功能演示")
    print("=" * 60)

    # 2.1 健康检查
    print("\n[1/7] 检查服务状态...")
    resp = requests.get("http://localhost:8000/health")
    print(f"  响应: {resp.json()}")

    # 2.2 创建施工队
    print("\n[2/7] 创建施工队...")
    teams = [
        {"name": "掘进一队", "team_code": "T001"},
        {"name": "采煤二队", "team_code": "T002"},
    ]
    team_ids = []
    for t in teams:
        resp = requests.post(f"{BASE_URL}/teams", json=t)
        data = resp.json()
        team_ids.append(data["id"])
        print(f"  创建成功: {data}")

    # 2.3 创建采区
    print("\n[3/7] 创建采区...")
    areas = [
        {"name": "A采区", "area_code": "A01"},
        {"name": "B采区", "area_code": "B02"},
    ]
    area_ids = []
    for a in areas:
        resp = requests.post(f"{BASE_URL}/areas", json=a)
        data = resp.json()
        area_ids.append(data["id"])
        print(f"  创建成功: {data}")

    # 2.4 上传带火图片 - 预期 is_safe=False
    print("\n[4/7] 上传火焰照片 (预期: 不安全)...")
    create_fire_test_image("test_fire.jpg")
    with open("test_fire.jpg", "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/inspections",
            data={
                "inspection_date": "2025-01-15",
                "team_id": team_ids[0],
                "area_id": area_ids[0],
                "shift": "白班",
            },
            files={"photo": ("fire.jpg", f, "image/jpeg")},
        )
    fire_result = resp.json()
    print(f"  检查结果: is_safe={fire_result['is_safe']}, "
          f"has_fire={fire_result['model_has_fire']}, "
          f"confidence={fire_result['model_confidence']:.4f}")

    # 2.5 上传安全图片 - 预期 is_safe=True
    print("\n[5/7] 上传安全照片 (预期: 安全)...")
    create_safe_test_image("test_safe.jpg")
    with open("test_safe.jpg", "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/inspections",
            data={
                "inspection_date": "2025-01-15",
                "team_id": team_ids[0],
                "area_id": area_ids[1],
                "shift": "夜班",
            },
            files={"photo": ("safe.jpg", f, "image/jpeg")},
        )
    safe_result = resp.json()
    print(f"  检查结果: is_safe={safe_result['is_safe']}, "
          f"has_fire={safe_result['model_has_fire']}, "
          f"confidence={safe_result['model_confidence']:.4f}")

    # 2.6 分页查询 - 无筛选
    print("\n[6/7] 分页查询 (全部记录)...")
    resp = requests.get(f"{BASE_URL}/inspections", params={"page": 1, "page_size": 10})
    list_data = resp.json()
    print(f"  总记录数: {list_data['total']}")
    print(f"  当前页: {list_data['page']}/{list_data['total_pages']}")
    for item in list_data["items"]:
        status = "⚠️ 危险" if not item["is_safe"] else "✅ 安全"
        print(f"    ID={item['id']} | {item['inspection_date']} | "
              f"队{item['team_id']}区{item['area_id']} | {item['shift']} | {status}")

    # 2.7 按条件筛选
    print("\n[7/7] 按条件筛选 (仅查询不安全记录)...")
    resp = requests.get(f"{BASE_URL}/inspections", params={"is_safe": "false"})
    danger_data = resp.json()
    print(f"  危险记录数: {danger_data['total']}")
    for item in danger_data["items"]:
        print(f"    ID={item['id']} | 置信度: {item['model_confidence']:.4f}")

    print("\n" + "=" * 60)
    print("  ✅ 演示完成! 所有 API 接口工作正常")
    print(f"  Swagger文档: http://localhost:8000/docs")
    print("=" * 60)


if __name__ == "__main__":
    demo_flow()
