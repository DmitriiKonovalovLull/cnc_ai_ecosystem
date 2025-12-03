# api/main_clean.py - ПОЛНЫЙ КОД ЧИСТОЙ РАБОЧЕЙ ВЕРСИИ
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pathlib import Path
import uvicorn
import asyncio

# Загружаем переменные окружения
from dotenv import load_dotenv

load_dotenv()

# Настройки
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
PORT = int(os.getenv("PORT", 8000))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Создаем необходимые директории
Path("logs").mkdir(exist_ok=True)
Path("storage/uploads").mkdir(parents=True, exist_ok=True)
Path("storage/temp").mkdir(parents=True, exist_ok=True)
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    print("🚀 Starting CNC AI Ecosystem API")

    # Инициализируем состояние приложения
    app.state.start_time = datetime.now()
    app.state.requests_count = 0
    app.state.active_users = set()

    # Проверяем переменные окружения
    required_env_vars = ["JWT_SECRET_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        print(f"⚠️  Warning: Missing environment variables: {missing_vars}")
        print("   Using default values for development")

    yield

    # Завершение работы
    print("🛑 Stopping CNC AI Ecosystem API")

    # Сохраняем статистику
    uptime = datetime.now() - app.state.start_time
    print(f"📊 Uptime: {uptime}")
    print(f"📊 Total requests: {app.state.requests_count}")


# Создаем приложение FastAPI
app = FastAPI(
    title="CNC AI Ecosystem API",
    description="AI Assistant for CNC Operators and Engineers",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Android app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent.parent / "storage"
app.mount("/storage", StaticFiles(directory=str(static_dir)), name="storage")

# Подключаем простые эндпоинты
try:
    from api.endpoints.health import router as health_router

    app.include_router(health_router, tags=["health"])
    print("✅ Health endpoints loaded")
except ImportError:
    print("⚠️  Health endpoints not found, using defaults")

try:
    from api.endpoints.chat_simple import router as chat_router

    app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
    print("✅ Chat endpoints loaded")
except ImportError:
    print("⚠️  Chat endpoints not found")

try:
    from api.endpoints.feedback_simple import router as feedback_router

    app.include_router(feedback_router, prefix="/api/v1", tags=["feedback"])
    print("✅ Feedback endpoints loaded")
except ImportError:
    print("⚠️  Feedback endpoints not found")

try:
    from api.endpoints.admin_simple import router as admin_router

    app.include_router(admin_router, prefix="/api/v1/admin", tags=["admin"])
    print("✅ Admin endpoints loaded")
except ImportError:
    print("⚠️  Admin endpoints not found")


# Middleware для подсчета запросов
@app.middleware("http")
async def count_requests(request, call_next):
    app.state.requests_count += 1

    # Логируем запросы в режиме отладки
    if DEBUG:
        print(f"📥 {request.method} {request.url.path}")

    response = await call_next(request)
    return response


# Базовые эндпоинты
@app.get("/", tags=["health"])
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "CNC AI Ecosystem API",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health",
        "uptime": str(datetime.now() - app.state.start_time)
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Проверка здоровья системы"""
    try:
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime": str(datetime.now() - app.state.start_time),
            "requests_count": app.state.requests_count,
            "active_users": len(app.state.active_users),
            "services": {
                "api": "operational",
                "database": "not_configured",
                "redis": "not_configured",
                "ai_models": "not_configured"
            },
            "environment": {
                "debug": DEBUG,
                "log_level": LOG_LEVEL,
                "port": PORT
            }
        }

        # Проверяем наличие API ключей
        if os.getenv("OPENAI_API_KEY"):
            status["services"]["openai"] = "configured"
        else:
            status["services"]["openai"] = "not_configured"

        if os.getenv("SANDVIK_API_KEY"):
            status["services"]["sandvik"] = "configured"

        if os.getenv("ISCAR_API_KEY"):
            status["services"]["iscar"] = "configured"

        return status

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.get("/status", tags=["status"])
async def system_status():
    """Статус системы"""
    return {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "statistics": {
            "requests_total": app.state.requests_count,
            "uptime": str(datetime.now() - app.state.start_time),
            "active_users": len(app.state.active_users)
        },
        "features": {
            "chat": "basic",
            "feedback": "basic",
            "admin": "basic",
            "database": "stub",
            "ai_inference": "stub"
        }
    }


@app.get("/logs", tags=["admin"])
async def get_logs():
    """Получить список лог файлов"""
    import glob

    log_files = {
        "api": sorted(glob.glob("logs/api_*.log"), reverse=True),
        "errors": sorted(glob.glob("logs/error_*.log"), reverse=True)
    }

    return {
        "available_logs": {
            "api": [Path(f).name for f in log_files["api"][:5]],
            "errors": [Path(f).name for f in log_files["errors"][:5]]
        },
        "latest_api_log": log_files["api"][0] if log_files["api"] else None,
        "latest_error_log": log_files["errors"][0] if log_files["errors"] else None
    }


# Обработчики ошибок
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "path": request.url.path,
            "timestamp": datetime.now().isoformat(),
            "request_id": str(hash(request))
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    print(f"🔥 Unhandled exception: {exc}")

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if DEBUG else "Contact administrator",
            "path": request.url.path,
            "timestamp": datetime.now().isoformat(),
            "request_id": str(hash(request))
        }
    )


# Запуск сервера
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 CNC AI Ecosystem API")
    print("=" * 50)
    print(f"📡 Host: 0.0.0.0:{PORT}")
    print(f"🔧 Debug mode: {'ENABLED' if DEBUG else 'DISABLED'}")
    print(f"📝 Log level: {LOG_LEVEL}")
    print(f"📚 Documentation: http://localhost:{PORT}/docs")
    print(f"📊 Health check: http://localhost:{PORT}/health")
    print("=" * 50)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        reload=DEBUG,
        log_level=LOG_LEVEL.lower()
    )