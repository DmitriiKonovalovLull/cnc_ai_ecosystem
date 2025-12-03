from celery import Celery
from kombu import Queue, Exchange
import os

from config.settings import settings

# Create Celery app
celery_app = Celery(
    "cnc_tasks",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

# Configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Europe/Moscow",
    enable_utc=True,

    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    worker_concurrency=4,

    # Task routing
    task_routes={
        'tasks.scraping_tasks.*': {'queue': 'scraping'},
        'tasks.training_tasks.*': {'queue': 'training'},
        'tasks.embedding_tasks.*': {'queue': 'embedding'},
        'tasks.maintenance_tasks.*': {'queue': 'maintenance'}
    },

    # Queue configuration
    task_queues=(
        Queue('scraping', Exchange('scraping'), routing_key='scraping'),
        Queue('training', Exchange('training'), routing_key='training'),
        Queue('embedding', Exchange('embedding'), routing_key='embedding'),
        Queue('maintenance', Exchange('maintenance'), routing_key='maintenance'),
        Queue('default', Exchange('default'), routing_key='default'),
    ),

    # Beat schedule
    beat_schedule={
        'daily-scraping': {
            'task': 'tasks.scraping_tasks.schedule_daily_scraping',
            'schedule': 86400.0,  # 24 hours
        },
        'weekly-training': {
            'task': 'tasks.training_tasks.schedule_weekly_training',
            'schedule': 604800.0,  # 7 days
        },
        'hourly-embeddings': {
            'task': 'tasks.embedding_tasks.schedule_embedding_updates',
            'schedule': 3600.0,  # 1 hour
        }
    },

    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,

    # Security
    worker_disable_rate_limits=False,
    broker_connection_retry_on_startup=True,
)

# Optional: Add custom task classes
if __name__ == "__main__":
    celery_app.start()