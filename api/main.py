from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager
from pathlib import Path
import uvicorn
from loguru import logger
import asyncio

from config.settings import settings
from api.endpoints import chat, feedback, admin
from api.dependencies import get_current_user, get_db
from knowledge_base.feedback_loop import FeedbackLoop
from ml_core.training.data_collector import DataCollector
from tasks.celery_app import celery_app

# Configure logging
logger.add(
    "logs/api_{time}.log",
    rotation="500 MB",
    retention="10 days",
    level=settings.LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🚀 Starting CNC AI Ecosystem API")

    # Initialize components
    app.state.feedback_loop = FeedbackLoop()
    app.state.data_collector = DataCollector()

    # Start feedback processing
    app.state.feedback_task = asyncio.create_task(
        app.state.feedback_loop.process_feedback(
            processor_callback=app.state.data_collector.collect_from_operators
        )
    )

    yield

    # Cleanup
    logger.info("🛑 Stopping CNC AI Ecosystem API")

    # Stop feedback processing
    if hasattr(app.state, 'feedback_task'):
        await app.state.feedback_loop.stop_processing()
        app.state.feedback_task.cancel()
        try:
            await app.state.feedback_task
        except asyncio.CancelledError:
            pass


# Create FastAPI app
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
static_dir.mkdir(exist_ok=True)
app.mount("/storage", StaticFiles(directory=str(static_dir)), name="storage")

# Include routers
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(feedback.router, prefix="/api/v1", tags=["feedback"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])

# Security
security = HTTPBearer()


# Health check endpoint
@app.get("/", tags=["health"])
async def root():
    return {
        "message": "CNC AI Ecosystem API",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        from api.dependencies import get_db
        async for db in get_db():
            await db.execute("SELECT 1")

        # Check Redis connection (via Celery)
        celery_app.control.inspect().active()

        # Check vector database
        from ml_core.inference.answer_engine import AnswerEngine
        engine = AnswerEngine()
        test_search = engine.search_similar("test", limit=1)

        return {
            "status": "healthy",
            "database": "connected",
            "redis": "connected",
            "vector_db": "connected",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.get("/status", tags=["status"])
async def system_status(current_user: dict = Depends(get_current_user)):
    """System status endpoint"""
    try:
        # Get feedback loop status
        feedback_status = app.state.feedback_loop.get_queue_status()

        # Get knowledge base stats
        from knowledge_base.kb_manager import KnowledgeBaseManager
        kb_manager = KnowledgeBaseManager()
        kb_stats = kb_manager.get_statistics()

        # Get training data stats
        data_stats = app.state.data_collector.get_stats()

        return {
            "status": "operational",
            "user": current_user.get("user_id", "anonymous"),
            "feedback_loop": feedback_status,
            "knowledge_base": kb_stats,
            "training_data": data_stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/{log_type}", tags=["admin"])
async def get_logs(log_type: str = "api", current_user: dict = Depends(get_current_user)):
    """Get application logs"""
    if not current_user.get("is_superuser", False):
        raise HTTPException(status_code=403, detail="Admin access required")

    log_files = {
        "api": "logs/api_*.log",
        "celery": "logs/celery_*.log",
        "errors": "logs/error_*.log"
    }

    if log_type not in log_files:
        raise HTTPException(status_code=400, detail=f"Invalid log type. Available: {list(log_files.keys())}")

    import glob
    log_pattern = log_files[log_type]
    log_files = sorted(glob.glob(log_pattern), reverse=True)

    if not log_files:
        return {"message": "No log files found"}

    # Return latest log file
    latest_log = log_files[0]
    return FileResponse(
        latest_log,
        media_type="text/plain",
        filename=f"{log_type}_{datetime.now().strftime('%Y%m%d')}.log"
    )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "path": request.url.path,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "Contact administrator",
            "path": request.url.path,
            "timestamp": datetime.now().isoformat()
        }
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("📊 Starting up CNC AI Ecosystem")

    # Create necessary directories
    directories = [
        "logs",
        "data/raw",
        "data/processed",
        "data/vector_db",
        "data/training",
        "data/feedback",
        "storage/uploads",
        "storage/temp"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    logger.info("✅ Startup completed")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🔴 Shutting down CNC AI Ecosystem")

    # Save any pending data
    if hasattr(app.state, 'data_collector'):
        app.state.data_collector.save_batch("shutdown")

    logger.info("✅ Shutdown completed")


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )
