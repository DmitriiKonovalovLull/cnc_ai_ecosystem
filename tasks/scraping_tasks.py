from celery import Task
import asyncio
import yaml
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import logging

from tasks.celery_app import celery_app
from data_pipeline.crawlers.web_crawler import WebCrawler
from data_pipeline.crawlers.api_crawler import ApiCrawler
from data_pipeline.crawlers.pdf_crawler import PDFCrawler
from knowledge_base.kb_manager import KnowledgeBaseManager
from config.settings import settings

logger = logging.getLogger(__name__)


class ScrapingTask(Task):
    """Base class for scraping tasks with error handling"""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Scraping task {task_id} failed: {exc}")
        super().on_failure(exc, task_id, args, kwargs, einfo)

    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"Scraping task {task_id} completed successfully")
        super().on_success(retval, task_id, args, kwargs)


@celery_app.task(base=ScrapingTask, bind=True, max_retries=3)
def scrape_url_task(self, url: str, force_refresh: bool = False,
                    priority: int = 1) -> Dict[str, Any]:
    """
    Scrape single URL and add to knowledge base
    """
    try:
        logger.info(f"Starting scrape task for URL: {url}")

        # Determine crawler type based on URL
        crawler = _select_crawler(url)

        # Run async crawl
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Perform crawl
            result = loop.run_until_complete(crawler.crawl(url, use_cache=not force_refresh))

            if not result.success:
                error_msg = f"Failed to crawl {url}: {result.error}"
                logger.error(error_msg)
                raise self.retry(exc=Exception(error_msg), countdown=60)

            # Parse content
            parsed_data = crawler.parse(result.content)
            parsed_data['source'] = url
            parsed_data['crawled_at'] = datetime.now().isoformat()

            # Add to knowledge base
            kb_manager = KnowledgeBaseManager()
            doc_id = loop.run_until_complete(
                kb_manager.add_document(
                    content=result.content,
                    metadata=parsed_data,
                    source=url
                )
            )

            # Cleanup
            loop.run_until_complete(crawler.cleanup())
            loop.close()

            return {
                'status': 'success',
                'url': url,
                'document_id': doc_id,
                'content_length': len(result.content),
                'entities_found': len(parsed_data.get('entities', [])),
                'processing_time': result.duration
            }

        except Exception as e:
            loop.close()
            raise

    except Exception as e:
        logger.error(f"Scraping task failed for {url}: {e}")
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))


@celery_app.task(base=ScrapingTask)
def scrape_batch_task(urls: List[str], batch_size: int = 5) -> Dict[str, Any]:
    """
    Scrape batch of URLs
    """
    results = []
    errors = []

    for i in range(0, len(urls), batch_size):
        batch = urls[i:i + batch_size]

        for url in batch:
            try:
                # Use apply_async for parallel processing
                task = scrape_url_task.apply_async(
                    args=[url],
                    kwargs={'force_refresh': False},
                    queue='scraping'
                )
                results.append(task.id)

                # Rate limiting
                import time
                time.sleep(1)  # 1 second delay between requests

            except Exception as e:
                errors.append({'url': url, 'error': str(e)})

    return {
        'batch_size': len(urls),
        'tasks_created': len(results),
        'errors': len(errors),
        'task_ids': results,
        'error_details': errors,
        'timestamp': datetime.now().isoformat()
    }


@celery_app.task
def schedule_daily_scraping():
    """
    Schedule daily scraping of configured sources
    """
    try:
        # Load scraping sources from config
        config_path = Path(__file__).parent.parent / "config" / "scraping_sources.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            sources_config = yaml.safe_load(f)

        scheduled_tasks = []

        # Schedule scraping for each source category
        for category, sources in sources_config.get('sources', {}).items():
            logger.info(f"Scheduling {len(sources)} sources from category: {category}")

            for source in sources:
                try:
                    url = source.get('url')
                    if not url:
                        continue

                    # Schedule scraping task
                    task = scrape_url_task.apply_async(
                        args=[url],
                        kwargs={
                            'force_refresh': False,
                            'priority': source.get('priority', 5)
                        },
                        queue='scraping',
                        countdown=source.get('rate_limit_delay', 10)
                    )

                    scheduled_tasks.append({
                        'category': category,
                        'source': source.get('name'),
                        'url': url,
                        'task_id': task.id
                    })

                except Exception as e:
                    logger.error(f"Failed to schedule {source.get('name')}: {e}")
                    continue

        return {
            'status': 'scheduled',
            'total_sources': sum(len(s) for s in sources_config.get('sources', {}).values()),
            'scheduled_tasks': len(scheduled_tasks),
            'tasks': scheduled_tasks,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to schedule daily scraping: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task
def monitor_scraping_progress():
    """
    Monitor scraping progress and send alerts if needed
    """
    try:
        from tasks.celery_app import celery_app

        inspector = celery_app.control.inspect()

        # Get active and scheduled tasks
        active = inspector.active() or {}
        scheduled = inspector.scheduled() or {}

        total_active = sum(len(tasks) for tasks in active.values())
        total_scheduled = sum(len(tasks) for tasks in scheduled.values())

        # Check for stuck tasks
        stuck_tasks = []
        for worker, tasks in active.items():
            for task in tasks:
                # Check if task has been running for too long
                if task.get('time_start', 0):
                    import time
                    running_time = time.time() - task['time_start']
                    if running_time > 3600:  # 1 hour
                        stuck_tasks.append({
                            'worker': worker,
                            'task_id': task.get('id'),
                            'name': task.get('name'),
                            'running_time': running_time
                        })

        return {
            'status': 'monitoring',
            'active_tasks': total_active,
            'scheduled_tasks': total_scheduled,
            'stuck_tasks': len(stuck_tasks),
            'stuck_details': stuck_tasks,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def _select_crawler(url: str):
    """Select appropriate crawler based on URL"""
    if url.endswith('.pdf'):
        return PDFCrawler()
    elif 'api' in url or url.endswith('.json'):
        return ApiCrawler()
    else:
        return WebCrawler()