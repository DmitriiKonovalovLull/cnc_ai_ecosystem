import aiohttp
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd

from .base_crawler import BaseCrawler, CrawlResult
from config.settings import settings

logger = logging.getLogger(__name__)


class ApiCrawler(BaseCrawler):
    def __init__(self, api_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.api_config = api_config or {}
        self.auth_tokens = {}
        self.rate_limit_info = {}

    async def authenticate(self, api_name: str) -> Optional[str]:
        """Authenticate with API and get token"""
        if api_name in self.auth_tokens:
            token_data = self.auth_tokens[api_name]
            # Check if token is still valid
            if datetime.now() < token_data.get('expires_at', datetime.now()):
                return token_data['token']

        # Get API credentials
        api_key_env = self.api_config.get('api_key_env')
        if not api_key_env:
            logger.error(f"No API key environment variable configured for {api_name}")
            return None

        api_key = getattr(settings, api_key_env, None)
        if not api_key:
            logger.error(f"API key not found in settings: {api_key_env}")
            return None

        # Different authentication strategies
        auth_url = self.api_config.get('auth_url')
        if auth_url:
            # OAuth or custom auth
            auth_payload = self.api_config.get('auth_payload', {})
            auth_payload['api_key'] = api_key

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(auth_url, json=auth_payload) as response:
                        if response.status == 200:
                            auth_data = await response.json()
                            token = auth_data.get('access_token') or auth_data.get('token')

                            if token:
                                # Store token with expiration
                                expires_in = auth_data.get('expires_in', 3600)
                                self.auth_tokens[api_name] = {
                                    'token': token,
                                    'expires_at': datetime.now() + timedelta(seconds=expires_in)
                                }
                                return token
            except Exception as e:
                logger.error(f"Authentication failed for {api_name}: {e}")
        else:
            # Simple API key
            return api_key

        return None

    async def crawl(self, url: str, use_cache: bool = True) -> CrawlResult:
        """Crawl API endpoint"""
        start_time = datetime.now()

        # Check cache first
        if use_cache:
            cached = await self.get_cached_content(url)
            if cached:
                logger.info(f"API cache hit for: {url}")
                return CrawlResult(
                    url=url,
                    content=cached,
                    metadata={"cached": True, "method": "api_cache"},
                    success=True,
                    duration=(datetime.now() - start_time).total_seconds()
                )

        # Get authentication token if needed
        api_name = self.api_config.get('name', 'unknown')
        auth_token = await self.authenticate(api_name)

        # Check rate limiting
        await self._check_rate_limit(api_name)

        # Make API request
        try:
            headers = self._prepare_headers(auth_token)

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    duration = (datetime.now() - start_time).total_seconds()

                    if response.status == 200:
                        content = await response.text()

                        # Update rate limit info
                        self._update_rate_limit_info(response.headers, api_name)

                        # Cache successful response
                        await self.set_cached_content(url, content)

                        return CrawlResult(
                            url=url,
                            content=content,
                            metadata={
                                "method": "api",
                                "status": response.status,
                                "rate_limit": self.rate_limit_info.get(api_name, {}),
                                "headers": dict(response.headers)
                            },
                            success=True,
                            duration=duration
                        )
                    else:
                        error_msg = f"API request failed: {response.status}"
                        logger.warning(f"{error_msg} for {url}")

                        return CrawlResult(
                            url=url,
                            content="",
                            metadata={
                                "status": response.status,
                                "rate_limit": self.rate_limit_info.get(api_name, {})
                            },
                            success=False,
                            error=error_msg,
                            duration=duration
                        )

        except Exception as e:
            logger.error(f"API crawl failed: {e}")
            return CrawlResult(
                url=url,
                content="",
                metadata={},
                success=False,
                error=str(e),
                duration=(datetime.now() - start_time).total_seconds()
            )

    def _prepare_headers(self, auth_token: Optional[str] = None) -> Dict[str, str]:
        """Prepare headers for API request"""
        headers = {
            "User-Agent": random.choice(settings.USER_AGENTS),
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate, br"
        }

        if auth_token:
            auth_type = self.api_config.get('auth_type', 'Bearer')
            if auth_type == 'Bearer':
                headers['Authorization'] = f'Bearer {auth_token}'
            elif auth_type == 'API-Key':
                headers['X-API-Key'] = auth_token
            elif auth_type == 'Basic':
                headers['Authorization'] = f'Basic {auth_token}'

        # Add custom headers from config
        custom_headers = self.api_config.get('headers', {})
        headers.update(custom_headers)

        return headers

    async def _check_rate_limit(self, api_name: str):
        """Check and respect rate limits"""
        rate_info = self.rate_limit_info.get(api_name, {})

        if rate_info.get('remaining', 1) <= 0:
            reset_time = rate_info.get('reset_time')
            if reset_time and datetime.now() < reset_time:
                wait_seconds = (reset_time - datetime.now()).total_seconds()
                logger.info(f"Rate limit reached for {api_name}, waiting {wait_seconds:.1f}s")
                await asyncio.sleep(wait_seconds + 1)

    def _update_rate_limit_info(self, headers: Dict[str, str], api_name: str):
        """Update rate limit information from response headers"""
        rate_info = {}

        # Common rate limit headers
        rate_limit_headers = {
            'x-ratelimit-limit': 'limit',
            'x-ratelimit-remaining': 'remaining',
            'x-ratelimit-reset': 'reset_time',
            'retry-after': 'retry_after'
        }

        for header_key, info_key in rate_limit_headers.items():
            if header_key in headers:
                value = headers[header_key]

                if info_key == 'reset_time' or info_key == 'retry_after':
                    # Parse timestamp or seconds
                    try:
                        if value.isdigit():
                            # Seconds until reset
                            reset_seconds = int(value)
                            rate_info[info_key] = datetime.now() + timedelta(seconds=reset_seconds)
                        else:
                            # Try to parse as timestamp
                            rate_info[info_key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    except:
                        pass
                else:
                    try:
                        rate_info[info_key] = int(value)
                    except:
                        rate_info[info_key] = value

        self.rate_limit_info[api_name] = rate_info

    def parse(self, content: str) -> Dict[str, Any]:
        """Parse API response"""
        try:
            data = json.loads(content)

            # Standardize response format
            parsed_data = {
                'data': data.get('data', data),
                'metadata': {
                    'source': self.api_config.get('name', 'unknown_api'),
                    'response_format': 'json',
                    'record_count': len(data.get('data', [])) if isinstance(data.get('data'), list) else 1
                }
            }

            # Extract entities based on API type
            api_type = self.api_config.get('type', 'unknown')
            if api_type == 'tool_catalog':
                parsed_data['entities'] = self._extract_tool_entities(data)
            elif api_type == 'machine_catalog':
                parsed_data['entities'] = self._extract_machine_entities(data)

            return parsed_data

        except json.JSONDecodeError:
            # Try CSV or other formats
            try:
                # Try to parse as CSV
                import io
                df = pd.read_csv(io.StringIO(content))
                return {
                    'data': df.to_dict('records'),
                    'metadata': {
                        'source': self.api_config.get('name', 'unknown_api'),
                        'response_format': 'csv',
                        'record_count': len(df)
                    }
                }
            except:
                # Return as plain text
                return {
                    'data': content,
                    'metadata': {
                        'source': self.api_config.get('name', 'unknown_api'),
                        'response_format': 'text'
                    }
                }

    def _extract_tool_entities(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool entities from API response"""
        entities = []

        if isinstance(data, dict):
            # Single tool record
            tool_data = data.get('data', data)

            # Extract tool information
            tool_info = {
                'type': 'tool',
                'name': tool_data.get('name') or tool_data.get('tool_name'),
                'manufacturer': tool_data.get('manufacturer') or self.api_config.get('name'),
                'category': tool_data.get('category') or tool_data.get('type'),
                'parameters': {}
            }

            # Extract cutting parameters if available
            cutting_params = tool_data.get('cutting_parameters') or tool_data.get('parameters') or {}
            if cutting_params:
                tool_info['parameters'] = {
                    'cutting_speed': cutting_params.get('cutting_speed'),
                    'feed_rate': cutting_params.get('feed_rate'),
                    'depth_of_cut': cutting_params.get('depth_of_cut'),
                    'material': cutting_params.get('material')
                }

            entities.append(tool_info)

        elif isinstance(data, list):
            # Multiple tool records
            for tool in data[:100]:  # Limit to first 100
                if isinstance(tool, dict):
                    tool_info = {
                        'type': 'tool',
                        'name': tool.get('name') or tool.get('tool_name'),
                        'manufacturer': tool.get('manufacturer') or self.api_config.get('name'),
                        'category': tool.get('category') or tool.get('type')
                    }
                    entities.append(tool_info)

        return entities

    def _extract_machine_entities(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract machine entities from API response"""
        entities = []

        if isinstance(data, dict):
            machine_data = data.get('data', data)

            machine_info = {
                'type': 'machine',
                'name': machine_data.get('name') or machine_data.get('model'),
                'manufacturer': machine_data.get('manufacturer') or self.api_config.get('name'),
                'type': machine_data.get('machine_type') or machine_data.get('type'),
                'specifications': machine_data.get('specifications') or {}
            }

            entities.append(machine_info)

        return entities

    def extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract entities from API response"""
        try:
            data = json.loads(content)
            api_type = self.api_config.get('type', 'unknown')

            if api_type == 'tool_catalog':
                return self._extract_tool_entities(data)
            elif api_type == 'machine_catalog':
                return self._extract_machine_entities(data)
            else:
                return []

        except:
            return []