from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio

from api.dependencies import get_admin_user, get_current_user
from tasks.celery_app import celery_app
from knowledge_base.kb_manager import KnowledgeBaseManager
from ml_core.training.data_collector import DataCollector
from ml_core.training.trainer import ModelTrainer

router = APIRouter()


# Request/Response Models
class ScrapingTaskRequest(BaseModel):
    urls: List[str] = Field(..., min_items=1, description="URLs to scrape")
    priority: int = Field(1, ge=1, le=10, description="Task priority")
    force_refresh: bool = Field(False, description="Ignore cache")


class TrainingTaskRequest(BaseModel):
    model_type: str = Field(..., description="Model type to train")
    data_source: Optional[str] = Field(None, description="Data source for training")
    epochs: int = Field(10, ge=1, le=100, description="Training epochs")


class SystemCommandRequest(BaseModel):
    command: str = Field(..., description="System command to execute")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Command parameters")


@router.get("/dashboard")
async def admin_dashboard(current_user: dict = Depends(get_admin_user)):
    """
    Admin dashboard with system overview
    """
    try:
        # Get system statistics from various components
        from api.main import app

        # Knowledge Base stats
        kb_manager = KnowledgeBaseManager()
        kb_stats = kb_manager.get_statistics()

        # Feedback loop stats
        feedback_loop = app.state.feedback_loop
        feedback_stats = feedback_loop.get_queue_status()

        # Data collector stats
        data_collector = app.state.data_collector
        data_stats = data_collector.get_stats()

        # Celery worker stats
        try:
            celery_stats = celery_app.control.inspect().stats() or {}
        except:
            celery_stats = {"error": "Celery not available"}

        # System info
        import psutil
        import platform

        system_info = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage("/").percent,
            "uptime": psutil.boot_time()
        }

        return {
            "system": system_info,
            "knowledge_base": kb_stats,
            "feedback_loop": feedback_stats,
            "training_data": data_stats,
            "celery_workers": celery_stats,
            "timestamp": datetime.now().isoformat(),
            "user": current_user.get("user_id")
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading dashboard: {str(e)}"
        )


@router.post("/scraping/start")
async def start_scraping_task(
        request: ScrapingTaskRequest,
        background_tasks: BackgroundTasks,
        current_user: dict = Depends(get_admin_user)
):
    """
    Start scraping task for given URLs
    """
    try:
        from tasks.scraping_tasks import scrape_url_task

        task_ids = []

        for url in request.urls:
            task = scrape_url_task.delay(
                url=url,
                force_refresh=request.force_refresh,
                priority=request.priority
            )
            task_ids.append(task.id)

        return {
            "action": "scraping_started",
            "urls_count": len(request.urls),
            "task_ids": task_ids,
            "priority": request.priority,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error starting scraping: {str(e)}"
        )


@router.get("/scraping/status")
async def get_scraping_status(
        task_id: Optional[str] = None,
        current_user: dict = Depends(get_admin_user)
):
    """
    Get status of scraping tasks
    """
    try:
        if task_id:
            # Get specific task status
            task = celery_app.AsyncResult(task_id)
            return {
                "task_id": task_id,
                "status": task.status,
                "result": task.result if task.ready() else None,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Get all active tasks
            inspector = celery_app.control.inspect()
            active = inspector.active() or {}
            scheduled = inspector.scheduled() or {}
            reserved = inspector.reserved() or {}

            return {
                "active_tasks": active,
                "scheduled_tasks": scheduled,
                "reserved_tasks": reserved,
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting scraping status: {str(e)}"
        )


@router.post("/training/start")
async def start_training_task(
        request: TrainingTaskRequest,
        background_tasks: BackgroundTasks,
        current_user: dict = Depends(get_admin_user)
):
    """
    Start model training task
    """
    try:
        from tasks.training_tasks import train_model_task

        task = train_model_task.delay(
            model_type=request.model_type,
            data_source=request.data_source,
            epochs=request.epochs
        )

        return {
            "action": "training_started",
            "model_type": request.model_type,
            "task_id": task.id,
            "epochs": request.epochs,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error starting training: {str(e)}"
        )


@router.get("/training/status")
async def get_training_status(
        task_id: Optional[str] = None,
        current_user: dict = Depends(get_admin_user)
):
    """
    Get status of training tasks
    """
    try:
        if task_id:
            task = celery_app.AsyncResult(task_id)

            return {
                "task_id": task_id,
                "status": task.status,
                "progress": task.info if task.info else {},
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Get all training tasks
            inspector = celery_app.control.inspect()
            active = inspector.active() or {}

            training_tasks = {}
            for worker, tasks in active.items():
                training_tasks[worker] = [
                    task for task in tasks
                    if task.get('name', '').startswith('tasks.training')
                ]

            return {
                "training_tasks": training_tasks,
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting training status: {str(e)}"
        )


@router.post("/system/command")
async def execute_system_command(
        request: SystemCommandRequest,
        current_user: dict = Depends(get_admin_user)
):
    """
    Execute system command (admin only)
    """
    try:
        # Security check - only allow specific commands
        allowed_commands = {
            "clear_cache": lambda: clear_system_cache(),
            "rebuild_index": lambda: rebuild_knowledge_index(),
            "export_data": lambda params: export_system_data(params),
            "import_data": lambda params: import_system_data(params),
            "backup": lambda params: create_system_backup(params)
        }

        if request.command not in allowed_commands:
            raise HTTPException(
                status_code=400,
                detail=f"Command not allowed: {request.command}"
            )

        # Execute command
        result = allowed_commands[request.command](request.parameters)

        return {
            "command": request.command,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing command: {str(e)}"
        )


@router.get("/users")
async def get_users(
        limit: int = 50,
        offset: int = 0,
        current_user: dict = Depends(get_admin_user)
):
    """
    Get user list (admin only)
    """
    try:
        # This would typically query a database
        # For now, return mock data
        users = []

        return {
            "users": users,
            "total": len(users),
            "limit": limit,
            "offset": offset,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting users: {str(e)}"
        )


@router.post("/users/{user_id}/toggle")
async def toggle_user_status(
        user_id: str,
        current_user: dict = Depends(get_admin_user)
):
    """
    Toggle user active status (admin only)
    """
    try:
        # This would typically update database
        return {
            "action": "user_status_toggled",
            "user_id": user_id,
            "message": "User status updated",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error toggling user status: {str(e)}"
        )


@router.get("/logs")
async def get_system_logs(
        log_type: str = "all",
        lines: int = 100,
        current_user: dict = Depends(get_admin_user)
):
    """
    Get system logs (admin only)
    """
    try:
        import glob
        import os

        log_files = {
            "api": "logs/api_*.log",
            "celery": "logs/celery_*.log",
            "errors": "logs/error_*.log",
            "all": "logs/*.log"
        }

        if log_type not in log_files:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid log type. Available: {list(log_files.keys())}"
            )

        log_pattern = log_files[log_type]
        log_files_list = sorted(glob.glob(log_pattern), reverse=True)

        logs_content = []

        for log_file in log_files_list[:3]:  # Last 3 files
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    file_lines = f.readlines()[-lines:]
                    logs_content.append({
                        "file": os.path.basename(log_file),
                        "lines": file_lines,
                        "size": os.path.getsize(log_file)
                    })
            except Exception as e:
                logs_content.append({
                    "file": os.path.basename(log_file),
                    "error": str(e)
                })

        return {
            "log_type": log_type,
            "files_found": len(log_files_list),
            "files_read": len(logs_content),
            "lines_per_file": lines,
            "logs": logs_content,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading logs: {str(e)}"
        )


@router.post("/knowledge/import")
async def import_knowledge_base(
        file_path: str,
        current_user: dict = Depends(get_admin_user)
):
    """
    Import knowledge base from file
    """
    try:
        kb_manager = KnowledgeBaseManager()

        with open(file_path, 'r', encoding='utf-8') as f:
            import_data = f.read()

        result = await kb_manager.import_knowledge(import_data, format='json')

        return {
            "action": "knowledge_imported",
            "result": result,
            "file": file_path,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error importing knowledge: {str(e)}"
        )


@router.post("/knowledge/export")
async def export_knowledge_base(
        format: str = "json",
        current_user: dict = Depends(get_admin_user)
):
    """
    Export knowledge base
    """
    try:
        kb_manager = KnowledgeBaseManager()

        export_data = await kb_manager.export_knowledge(format)

        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"knowledge_export_{timestamp}.{format}"
        filepath = f"exports/{filename}"

        os.makedirs("exports", exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(export_data)

        return {
            "action": "knowledge_exported",
            "format": format,
            "file": filepath,
            "size": len(export_data),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error exporting knowledge: {str(e)}"
        )


# Helper functions for system commands
def clear_system_cache():
    """Clear system cache"""
    import shutil
    import os

    cache_dirs = [
        "data/vector_db/chroma",
        "data/raw",
        "data/temp"
    ]

    results = {}

    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir)
                results[cache_dir] = "cleared"
            except Exception as e:
                results[cache_dir] = f"error: {str(e)}"
        else:
            results[cache_dir] = "not_found"

    return results


def rebuild_knowledge_index():
    """Rebuild knowledge base index"""
    kb_manager = KnowledgeBaseManager()
    # The index is rebuilt automatically in init
    return {"status": "index_rebuilt", "documents": len(kb_manager.documents)}


def export_system_data(params: Optional[Dict[str, Any]] = None):
    """Export system data"""
    # Implementation depends on what data to export
    return {"action": "export", "params": params, "status": "completed"}


def import_system_data(params: Optional[Dict[str, Any]] = None):
    """Import system data"""
    # Implementation depends on what data to import
    return {"action": "import", "params": params, "status": "completed"}


def create_system_backup(params: Optional[Dict[str, Any]] = None):
    """Create system backup"""
    import shutil
    import tarfile
    from datetime import datetime

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f"backup_{timestamp}"

    # Create backup directory
    os.makedirs("backups", exist_ok=True)

    # Files and directories to backup
    backup_items = [
        "data",
        "config",
        "knowledge_base",
        "logs"
    ]

    # Create tar archive
    backup_file = f"backups/{backup_name}.tar.gz"

    with tarfile.open(backup_file, "w:gz") as tar:
        for item in backup_items:
            if os.path.exists(item):
                tar.add(item)

    return {
        "backup_file": backup_file,
        "size": os.path.getsize(backup_file),
        "items": backup_items,
        "timestamp": timestamp
    }