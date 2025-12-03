import pdfplumber
import PyPDF2
import io
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from .base_crawler import BaseCrawler, CrawlResult

logger = logging.getLogger(__name__)


class PDFCrawler(BaseCrawler):
    def __init__(self):
        super().__init__()
        self.text_extractors = [
            self._extract_with_pdfplumber,
            self._extract_with_pypdf2,
            self._extract_with_ocr  # Would require additional dependencies
        ]

    async def crawl(self, url: str, use_cache: bool = True) -> CrawlResult:
        """Download and extract PDF content"""
        start_time = datetime.now()

        # Check cache first
        if use_cache:
            cached = await self.get_cached_content(url)
            if cached:
                logger.info(f"PDF cache hit for: {url}")
                return CrawlResult(
                    url=url,
                    content=cached,
                    metadata={"cached": True, "method": "pdf_cache"},
                    success=True,
                    duration=(datetime.now() - start_time).total_seconds()
                )

        try:
            # Download PDF
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        pdf_content = await response.read()

                        # Try different extraction methods
                        extracted_text = ""
                        for extractor in self.text_extractors:
                            try:
                                extracted_text = extractor(pdf_content)
                                if extracted_text and len(extracted_text) > 100:  # Valid extraction
                                    break
                            except Exception as e:
                                logger.debug(f"PDF extractor failed: {e}")
                                continue

                        if not extracted_text or len(extracted_text) < 100:
                            logger.warning(f"PDF text extraction failed for: {url}")
                            return CrawlResult(
                                url=url,
                                content="",
                                metadata={},
                                success=False,
                                error="Text extraction failed",
                                duration=(datetime.now() - start_time).total_seconds()
                            )

                        # Cache extracted text
                        await self.set_cached_content(url, extracted_text)

                        return CrawlResult(
                            url=url,
                            content=extracted_text,
                            metadata={
                                "method": "pdf",
                                "pages": self._count_pages(pdf_content),
                                "file_size": len(pdf_content)
                            },
                            success=True,
                            duration=(datetime.now() - start_time).total_seconds()
                        )
                    else:
                        return CrawlResult(
                            url=url,
                            content="",
                            metadata={"status": response.status},
                            success=False,
                            error=f"HTTP {response.status}",
                            duration=(datetime.now() - start_time).total_seconds()
                        )

        except Exception as e:
            logger.error(f"PDF crawl failed: {e}")
            return CrawlResult(
                url=url,
                content="",
                metadata={},
                success=False,
                error=str(e),
                duration=(datetime.now() - start_time).total_seconds()
            )

    def _extract_with_pdfplumber(self, pdf_content: bytes) -> str:
        """Extract text using pdfplumber"""
        text_parts = []

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

                # Extract tables if present
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        for row in table:
                            if row:
                                row_text = ' | '.join([str(cell).strip() for cell in row if cell])
                                text_parts.append(row_text)

        return '\n\n'.join(text_parts)

    def _extract_with_pypdf2(self, pdf_content: bytes) -> str:
        """Extract text using PyPDF2"""
        text_parts = []

        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        return '\n\n'.join(text_parts)

    def _extract_with_ocr(self, pdf_content: bytes) -> str:
        """Extract text using OCR (requires pytesseract and PIL)"""
        # This is a placeholder - would need additional dependencies
        try:
            import pytesseract
            from PIL import Image
            import fitz  # PyMuPDF

            text_parts = []
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")

            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Extract text with OCR
                page_text = pytesseract.image_to_string(img, lang='rus+eng')
                if page_text:
                    text_parts.append(page_text)

            return '\n\n'.join(text_parts)
        except ImportError:
            logger.warning("OCR dependencies not installed")
            return ""

    def _count_pages(self, pdf_content: bytes) -> int:
        """Count pages in PDF"""
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            return len(pdf_reader.pages)
        except:
            return 0

    def parse(self, content: str) -> Dict[str, Any]:
        """Parse extracted PDF text"""
        # Split by pages or sections
        sections = content.split('\n\n')

        # Find GOST codes
        gost_patterns = [
            r'ГОСТ\s+[\d\-\.]+',
            r'ГОСТ\s+[ИСО\d\-\.]+',
            r'ОСТ\s+[\d\-\.]+'
        ]

        gost_codes = []
        for pattern in gost_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            gost_codes.extend(matches)

        # Extract tables (simple regex-based)
        tables = []
        table_pattern = r'(\|?.+?\|\n?){3,}'  # Simple pattern for markdown-like tables
        for match in re.finditer(table_pattern, content, re.MULTILINE):
            table_text = match.group(0)
            rows = [row.strip() for row in table_text.split('\n') if row.strip()]
            if len(rows) >= 2:
                tables.append(rows)

        # Find sections with headers
        sections_data = []
        for i, section in enumerate(sections):
            if len(section.strip()) > 50:  # Non-empty sections
                lines = section.split('\n')
                if lines:
                    # First line might be a header
                    header = lines[0].strip() if len(lines[0]) < 100 else ""
                    content_text = '\n'.join(lines[1:] if header else lines)

                    sections_data.append({
                        'index': i,
                        'header': header,
                        'content': content_text[:1000],  # Limit content length
                        'length': len(content_text)
                    })

        return {
            'sections': sections_data,
            'gost_codes': list(set(gost_codes)),
            'tables': tables[:10],  # Limit number of tables
            'total_length': len(content),
            'section_count': len(sections_data)
        }

    def extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract entities from PDF text"""
        entities = []

        # GOST codes
        gost_patterns = [
            r'ГОСТ\s+[\d\-\.]+',
            r'ГОСТ\s+[ИСО\d\-\.]+',
            r'ОСТ\s+[\d\-\.]+',
            r'ISO\s+[\d\-\.]+',
            r'DIN\s+[\d\-\.]+'
        ]

        for pattern in gost_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'type': 'gost_code',
                    'value': match.group(0),
                    'start': match.start(),
                    'end': match.end()
                })

        # Material specifications
        material_patterns = [
            r'Материал:\s*(.+)',
            r'Material:\s*(.+)',
            r'Сталь\s+[0-9Хх]+',
            r'Steel\s+[A-Z0-9]+'
        ]

        for pattern in material_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'type': 'material',
                    'value': match.group(0),
                    'start': match.start(),
                    'end': match.end()
                })

        # Technical parameters
        param_patterns = [
            r'Толщина[:\s]*[\d\.]+\s*(?:мм|mm)',
            r'Диаметр[:\s]*[\d\.]+\s*(?:мм|mm)',
            r'Длина[:\s]*[\d\.]+\s*(?:мм|mm)',
            r'Thickness[:\s]*[\d\.]+\s*(?:mm)',
            r'Diameter[:\s]*[\d\.]+\s*(?:mm)',
            r'Length[:\s]*[\d\.]+\s*(?:mm)'
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