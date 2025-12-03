import re
import json
from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup
import pandas as pd
from loguru import logger

from .base_crawler import BaseCrawler, CrawlResult


class WebCrawler(BaseCrawler):
    def __init__(self, parser_type: str = "auto"):
        super().__init__()
        self.parser_type = parser_type
        self.gost_patterns = [
            r'ГОСТ\s+[\d\-\.]+',
            r'ГОСТ\s+[ИСО\d\-\.]+',
            r'ОСТ\s+[\d\-\.]+',
            r'СТ\s+[А-Я\d\-\.]+',
            r'ISO\s+[\d\-\.]+',
            r'DIN\s+[\d\-\.]+',
            r'ANSI\s+[A-Z]+\s+[\d\-\.]+',
            r'JIS\s+[A-Z]+\s+[\d\-\.]+'
        ]

        self.material_patterns = [
            r'сталь\s+[0-9Хх]+',
            r'алюминий\s+[А-Я0-9]+',
            r'титан\s+[А-Я0-9]+',
            r'чугун\s+[А-Я0-9]+',
            r'сплав\s+[А-Я0-9]+',
            r'Steel\s+[A-Z0-9]+',
            r'Aluminum\s+[A-Z0-9]+'
        ]

        self.tool_patterns = [
            r'фреза\s+[А-Яа-яA-Z0-9\-\s]+',
            r'сверло\s+[А-Яа-яA-Z0-9\-\s]+',
            r'резец\s+[А-Яа-яA-Z0-9\-\s]+',
            r'пластина\s+[A-Z0-9\-]+',
            r'insert\s+[A-Z0-9\-]+',
            r'cutter\s+[A-Z0-9\-]+',
            r'drill\s+[A-Z0-9\-]+'
        ]

    async def crawl(self, url: str, use_cache: bool = True) -> CrawlResult:
        """Extended crawl with domain-specific logic"""
        result = await super().crawl(url, use_cache)

        if result.success:
            # Additional processing based on domain
            if 'cntd.ru' in url:
                result.metadata['source_type'] = 'gost'
            elif 'sandvik' in url:
                result.metadata['source_type'] = 'tool_catalog'
            elif 'practicalmachinist' in url:
                result.metadata['source_type'] = 'forum'

        return result

    def parse(self, content: str) -> Dict[str, Any]:
        """Parse HTML content and extract structured data"""
        soup = BeautifulSoup(content, 'lxml')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'noscript']):
            element.decompose()

        data = {
            'url': '',
            'title': self._extract_title(soup),
            'description': self._extract_description(soup),
            'content': self._extract_content(soup),
            'tables': self._extract_tables(soup),
            'images': self._extract_images(soup),
            'links': self._extract_links(soup),
            'metadata': self._extract_metadata(soup),
            'entities': self.extract_entities(content),
            'processed_at': pd.Timestamp.now().isoformat()
        }

        return data

    def extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract entities from content"""
        entities = []

        # Extract GOST codes
        for pattern in self.gost_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'type': 'gost_code',
                    'value': match.group(0),
                    'start': match.start(),
                    'end': match.end()
                })

        # Extract materials
        for pattern in self.material_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'type': 'material',
                    'value': match.group(0),
                    'start': match.start(),
                    'end': match.end()
                })

        # Extract tools
        for pattern in self.tool_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'type': 'tool',
                    'value': match.group(0),
                    'start': match.start(),
                    'end': match.end()
                })

        # Extract numeric parameters
        param_patterns = [
            r'скорость\s+резания\s*[:=]?\s*[\d\.]+\s*(?:м/мин|m/min)',
            r'подача\s*[:=]?\s*[\d\.]+\s*(?:мм/об|mm/rev)',
            r'глубина\s+резания\s*[:=]?\s*[\d\.]+\s*(?:мм|mm)',
            r'стойкость\s*[:=]?\s*[\d\.]+\s*(?:мин|min)'
        ]

        for pattern in param_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'type': 'parameter',
                    'value': match.group(0),
                    'start': match.start(),
                    'end': match.end()
                })

        return entities

    def _extract_title(self, soup) -> str:
        """Extract page title"""
        title_selectors = [
            'h1',
            '.title',
            '.document-title',
            'meta[property="og:title"]',
            'meta[name="twitter:title"]',
            'title'
        ]

        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                if selector.startswith('meta'):
                    return element.get('content', '').strip()
                return element.get_text().strip()

        return ''

    def _extract_description(self, soup) -> str:
        """Extract page description"""
        desc_selectors = [
            'meta[name="description"]',
            'meta[property="og:description"]',
            '.description',
            '.summary',
            'article > p:first-of-type'
        ]

        for selector in desc_selectors:
            element = soup.select_one(selector)
            if element:
                if selector.startswith('meta'):
                    return element.get('content', '').strip()
                return element.get_text().strip()

        return ''

    def _extract_content(self, soup) -> str:
        """Extract main content"""
        content_selectors = [
            'main',
            'article',
            '.content',
            '.document-content',
            '.post-content',
            '.article-content',
            '#content'
        ]

        main_element = None
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                main_element = element
                break

        if not main_element:
            # Fallback to body
            main_element = soup.find('body')

        if main_element:
            # Get text with paragraph preservation
            paragraphs = main_element.find_all(['p', 'h2', 'h3', 'h4', 'li'])
            if paragraphs:
                content_parts = []
                for p in paragraphs:
                    text = p.get_text().strip()
                    if text and len(text) > 10:  # Filter very short texts
                        content_parts.append(text)

                if content_parts:
                    return '\n\n'.join(content_parts)

            # Fallback to all text
            text = main_element.get_text(separator='\n', strip=True)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            return '\n'.join(lines[:500])  # Limit to 500 lines

        return ''

    def _extract_tables(self, soup) -> List[Dict[str, Any]]:
        """Extract tables with data"""
        tables = []

        for table in soup.find_all('table'):
            try:
                table_data = []
                headers = []

                # Extract headers
                header_row = table.find('thead') or table.find('tr')
                if header_row:
                    headers = [
                        th.get_text(strip=True)
                        for th in header_row.find_all(['th', 'td'])
                    ]

                # Extract rows
                rows = table.find_all('tr')[1:] if header_row else table.find_all('tr')
                for row in rows:
                    cells = [
                        cell.get_text(strip=True)
                        for cell in row.find_all(['td', 'th'])
                    ]
                    if cells:
                        table_data.append(cells)

                if table_data:
                    tables.append({
                        'headers': headers,
                        'data': table_data,
                        'row_count': len(table_data),
                        'col_count': len(headers) if headers else len(table_data[0]) if table_data else 0
                    })

            except Exception as e:
                logger.warning(f"Failed to parse table: {e}")
                continue

        return tables

    def _extract_images(self, soup) -> List[Dict[str, str]]:
        """Extract images with metadata"""
        images = []

        for img in soup.find_all('img'):
            try:
                img_data = {
                    'src': img.get('src', ''),
                    'alt': img.get('alt', ''),
                    'title': img.get('title', ''),
                    'width': img.get('width', ''),
                    'height': img.get('height', '')
                }

                # Filter out tracking pixels and icons
                if (img_data['src'] and
                        not any(x in img_data['src'].lower() for x in ['pixel', 'track', 'icon', 'logo']) and
                        (img_data['width'] and int(img_data['width'] or 0) > 50) or
                        (img_data['height'] and int(img_data['height'] or 0) > 50)):
                    images.append(img_data)

            except Exception as e:
                continue

        return images

    def _extract_links(self, soup) -> List[Dict[str, str]]:
        """Extract links with context"""
        links = []

        for a in soup.find_all('a', href=True):
            try:
                link_data = {
                    'href': a.get('href', ''),
                    'text': a.get_text(strip=True)[:200],
                    'title': a.get('title', '')
                }

                # Filter out common non-content links
                if (link_data['href'] and
                        not any(x in link_data['href'].lower() for x in ['javascript:', 'mailto:', 'tel:', '#']) and
                        len(link_data['text']) > 2):
                    links.append(link_data)

            except Exception as e:
                continue

        return links

    def _extract_metadata(self, soup) -> Dict[str, str]:
        """Extract meta tags"""
        metadata = {}

        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property') or meta.get('itemprop')
            content = meta.get('content', '')

            if name and content:
                # Clean up common meta names
                clean_name = name.replace('og:', '').replace('twitter:', '').strip()
                if clean_name and content.strip():
                    metadata[clean_name] = content.strip()

        return metadata