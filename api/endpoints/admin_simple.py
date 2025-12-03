# api/endpoints/admin_simple.py
from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
from typing import Optional

router = APIRouter()


# Простая заглушка для аутентификации
async def get_admin_user():
    """Заглушка для проверки прав администратора"""
    # В реальном приложении здесь будет проверка JWT токена
    return {
        "user_id": "admin_001",
        "username": "admin",
        "is_superuser": True,
        "permissions": ["read", "write", "delete"]
    }


@router.get("/dashboard")
async def admin_dashboard(admin_user: dict = Depends(get_admin_user)):
    """Панель управления администратора"""
    if not admin_user.get("is_superuser", False):
        raise HTTPException(status_code=403, detail="Admin access required")

    # Здесь будут реальные данные из БД
    return {
        "status": "ok",
        "user": admin_user["username"],
        "timestamp": datetime.now().isoformat(),
        "statistics": {
            "total_users": 1,
            "active_users": 1,
            "total_messages": 0,
            "total_feedbacks": 0,
            "average_response_time": 0.5,
            "system_uptime": "0 days"
        },
        "system_info": {
            "api_version": "2.0.0",
            "environment": "development",
            "debug_mode": True,
            "features_enabled": ["chat", "feedback", "admin"]
        }
    }


@router.get("/users")
async def list_users(admin_user: dict = Depends(get_admin_user)):
    """Список пользователей"""
    if not admin_user.get("is_superuser", False):
        raise HTTPException(status_code=403, detail="Admin access required")

    # Заглушка - в реальном приложении будет запрос к БД
    return {
        "users": [
            {
                "user_id": "admin_001",
                "username": "admin",
                "email": "admin@cnc-ai.local",
                "role": "admin",
                "created_at": datetime.now().isoformat(),
                "is_active": True
            },
            {
                "user_id": "operator_001",
                "username": "operator",
                "email": "operator@cnc-ai.local",
                "role": "operator",
                "created_at": datetime.now().isoformat(),
                "is_active": True
            }
        ],
        "total": 2,
        "page": 1,
        "per_page": 20
    }