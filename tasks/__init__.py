from .celery_app import celery_app
from .scraping_tasks import scrape_url_task, schedule_daily_scraping
from .training_tasks import train_model_task, schedule_weekly_training
from .embedding_tasks import update_embeddings_task, schedule_embedding_updates

__all__ = [
    "celery_app",
    "scrape_url_task",
    "schedule_daily_scraping",
    "train_model_task",
    "schedule_weekly_training",
    "update_embeddings_task",
    "schedule_embedding_updates"
]