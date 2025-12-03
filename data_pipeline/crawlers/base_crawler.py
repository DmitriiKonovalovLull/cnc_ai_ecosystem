import asyncio
import random
import logging
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import aiohttp
from fake_useragent import UserAgent
from playwright.async_api import async_playwright
from redis.asyncio import Redis

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class CrawlResult:
    url: str
    content: str
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    duration: float = 0.0


class BaseCrawler(ABC):
    def __init__(self):
        self.user_agent = UserAgent()
        self.session: Optional[aiohttp.ClientSession] = None
        self.playwright_browser = None
        self.redis_client: Optional[Redis] = None
        self.cache_ttl = 3600  # 1 hour cache

    async def init_redis(self):
        """Initialize Redis connection for caching"""
        if not self.redis_client:
            self.redis_client = Redis.from_url(settings.REDIS_URL)

    def generate_cache_key(self, url: str) -> str:
        """Generate cache key for URL"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return f"crawl:{url_hash}"

    async def get_cached_content(self, url: str) -> Optional[str]:
        """Get cached content from Redis"""
        try:
            await self.init_redis()
            cache_key = self.generate_cache_key(url)
            content = await self.redis_client.get(cache_key)
            return content.decode() if content else None
        except Exception as e:
            logger.debug(f"Cache read failed: {e}")
            return None

    async def set_cached_content(self, url: str, content: str):
        """Cache content in Redis"""
        try:
            await self.init_redis()
            cache_key = self.generate_cache_key(url)
            await self.redis_client.setex(cache_key, self.cache_ttl, content)
        except Exception as e:
            logger.debug(f"Cache write failed: {e}")

    async def crawl(self, url: str, use_cache: bool = True) -> CrawlResult:
        """Main crawl method with caching and retry logic"""
        start_time = datetime.now()

        # Check cache first
        if use_cache:
            cached = await self.get_cached_content(url)
            if cached:
                logger.info(f"Cache hit for: {url}")
                return CrawlResult(
                    url=url,
                    content=cached,
                    metadata={"cached": True, "method": "cache"},
                    success=True,
                    duration=(datetime.now() - start_time).total_seconds()
                )

        # Try different strategies
        strategies = [
            self._crawl_with_playwright,
            self._crawl_with_aiohttp,
            self._crawl_with_selenium
        ]

        result = None
        last_error = None

        for strategy in strategies:
            try:
                # Random delay between attempts
                await asyncio.sleep(random.uniform(*settings.REQUEST_DELAY))

                result = await strategy(url)
                if result.success:
                    # Cache successful result
                    await self.set_cached_content(url, result.content)
                    break

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Strategy {strategy.__name__} failed: {e}")
                continue

        if not result or not result.success:
            result = CrawlResult(
                url=url,
                content="",
                metadata={},
                success=False,
                error=last_error or "All strategies failed"
            )

        result.duration = (datetime.now() - start_time).total_seconds()
        return result

    async def _crawl_with_playwright(self, url: str) -> CrawlResult:
        """Crawl using Playwright (handles JavaScript)"""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--disable-dev-shm-usage",
                        "--no-sandbox",
                        f"--user-agent={random.choice(settings.USER_AGENTS)}"
                    ]
                )

                context = await browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    locale="ru-RU"
                )

                page = await context.new_page()

                # Add random delays to mimic human
                await asyncio.sleep(random.uniform(0.5, 2.0))

                # Navigate with timeout
                await page.goto(
                    url,
                    wait_until="networkidle",
                    timeout=60000
                )

                # Random scrolling
                for _ in range(random.randint(1, 4)):
                    scroll_amount = random.randint(300, 1000)
                    await page.mouse.wheel(0, scroll_amount)
                    await asyncio.sleep(random.uniform(0.3, 1.2))

                # Wait for content to load
                await page.wait_for_load_state("networkidle")

                # Get content
                content = await page.content()

                # Take screenshot for debugging
                screenshot = await page.screenshot(full_page=True)

                await browser.close()

                return CrawlResult(
                    url=url,
                    content=content,
                    metadata={
                        "method": "playwright",
                        "screenshot": screenshot[:100] if screenshot else None
                    },
                    success=True
                )

        except Exception as e:
            logger.error(f"Playwright crawl failed: {e}")
            return CrawlResult(
                url=url,
                content="",
                metadata={},
                success=False,
                error=str(e)
            )

    async def _crawl_with_aiohttp(self, url: str) -> CrawlResult:
        """Crawl using aiohttp (fast for simple pages)"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            headers = {
                "User-Agent": random.choice(settings.USER_AGENTS),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }

            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    content = await response.text()

                    return CrawlResult(
                        url=url,
                        content=content,
                        metadata={
                            "method": "aiohttp",
                            "status": response.status,
                            "headers": dict(response.headers)
                        },
                        success=True
                    )
                else:
                    return CrawlResult(
                        url=url,
                        content="",
                        metadata={"status": response.status},
                        success=False,
                        error=f"HTTP {response.status}"
                    )

        except Exception as e:
            logger.error(f"aiohttp crawl failed: {e}")
            return CrawlResult(
                url=url,
                content="",
                metadata={},
                success=False,
                error=str(e)
            )

    async def _crawl_with_selenium(self, url: str) -> CrawlResult:
        """Fallback strategy with Selenium"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument(f"user-agent={random.choice(settings.USER_AGENTS)}")

            driver = webdriver.Chrome(options=chrome_options)
            driver.get(url)

            # Wait for page load
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Random delays
            import time
            time.sleep(random.uniform(1.0, 3.0))

            content = driver.page_source
            driver.quit()

            return CrawlResult(
                url=url,
                content=content,
                metadata={"method": "selenium"},
                success=True
            )

        except Exception as e:
            logger.error(f"Selenium crawl failed: {e}")
            return CrawlResult(
                url=url,
                content="",
                metadata={},
                success=False,
                error=str(e)
            )

    async def cleanup(self):
        """Cleanup resources"""
        if self.session and not self.session.closed:
            await self.session.close()

        if self.redis_client:
            await self.redis_client.close()

    @abstractmethod
    def parse(self, content: str) -> Dict[str, Any]:
        """Abstract method for parsing content"""
        pass

    @abstractmethod
    def extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract entities from content"""
        pass