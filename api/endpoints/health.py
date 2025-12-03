# api/endpoints/health.py
from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/health/detailed")
async def detailed_health():
    """Детальная проверка здоровья"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "operational",
            "websocket": "not_implemented",
            "background_tasks": "not_implemented",
            "file_storage": "basic"
        }
    }

@router.get("/health/ready")
async def readiness_probe():
    """Проверка готовности для Kubernetes/load balancers"""
    return {
        "status": "ready",
        "ready": True,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/health/live")
async def liveness_probe():
    """Проверка живучести"""
    return {
        "status": "live",
        "alive": True,
        "timestamp": datetime.now().isoformat()
    }