import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from .base_transformer import BaseTransformer


class GOSTTransformer(BaseTransformer):
    def __init__(self):
        super().__init__()
        self.gost_pattern = re.compile(
            r'(ГОСТ|ОСТ|СТ)\s+([\w\d\-\.]+)(?:\s+\(([^)]+)\))?',
            re.IGNORECASE
        )

        self.status_patterns = {
            'действующий': ['действ', 'действующий', 'активен', 'active', 'current'],
            'отменен': ['отменен', 'заменен', 'отмененный', 'obsolete', 'withdrawn'],
            'пересмотрен': ['пересмотр', 'пересмотренный', 'revised']
        }

    def transform(self, data: Any) -> Dict[str, Any]:
        """Transform GOST document data"""
        if isinstance(data, str):
            # Parse from text
            return self._transform_from_text(data)
        elif isinstance(data, dict):
            # Already structured
            return self._transform_from_dict(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _transform_from_text(self, text: str) -> Dict[str, Any]:
        """Transform from raw text"""
        # Find GOST reference
        gost_match = self.gost_pattern.search(text)

        if not gost_match:
            return {"error": "No GOST code found in text"}

        gost_type = gost_match.group(1).upper()
        gost_number = gost_match.group(2)
        gost_title = gost_match.group(3) or ""

        # Extract metadata
        metadata = self._extract_metadata(text)

        # Determine status
        status = self._determine_status(text)

        # Extract sections
        sections = self._extract_sections(text)

        # Extract tables
        tables = self._extract_tables_from_text(text)

        # Extract parameters
        parameters = self._extract_parameters(text)

        return {
            "gost_type": gost_type,
            "gost_number": gost_number,
            "gost_title": gost_title,
            "status": status,
            "metadata": metadata,
            "sections": sections,
            "tables": tables,
            "parameters": parameters,
            "full_text": text[:5000],  # Limit text length
            "processed_at": datetime.now().isoformat()
        }

    def _transform_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform from structured dict"""
        # Ensure required fields
        required = ['gost_number', 'title']
        for field in required:
            if field not in data:
                data[field] = ""

        # Extract GOST type and number
        gost_match = self.gost_pattern.search(data.get('gost_number', ''))
        if gost_match:
            data['gost_type'] = gost_match.group(1).upper()
            data['gost_number_clean'] = gost_match.group(2)
        else:
            data['gost_type'] = 'ГОСТ'
            data['gost_number_clean'] = data['gost_number']

        # Determine status
        if 'status' not in data:
            data['status'] = self._determine_status(data.get('content', ''))

        # Extract parameters if not present
        if 'parameters' not in data and 'content' in data:
            data['parameters'] = self._extract_parameters(data['content'])

        # Add processing timestamp
        data['processed_at'] = datetime.now().isoformat()

        return data

    def _extract_metadata(self, text: str) -> Dict[str, str]:
        """Extract metadata from text"""
        metadata = {}

        # Extract dates
        date_patterns = [
            r'Дата\s+введения[:\s]*(\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4})',
            r'Дата\s+издания[:\s]*(\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4})',
            r'Введен\s+в\s+действие[:\s]*(\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4})',
            r'(\d{4})\s+год'
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata['date'] = match.group(1)
                break

        # Extract organization
        org_patterns = [
            r'Разработан\s+(?:[А-Яа-я\s]+)\s+(.+)',
            r'Утвержден\s+(?:[А-Яа-я\s]+)\s+(.+)',
            r'Принят\s+(?:[А-Яа-я\s]+)\s+(.+)'
        ]

        for pattern in org_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata['organization'] = match.group(1).strip()
                break

        # Extract keywords
        if len(text) > 100:
            # Simple keyword extraction (first 1000 chars)
            sample = text[:1000].lower()
            keywords = []

            tech_terms = [
                'материал', 'допуск', 'размер', 'толщина', 'диаметр',
                'прочность', 'твердость', 'шероховатость', 'точность'
            ]

            for term in tech_terms:
                if term in sample:
                    keywords.append(term)

            if keywords:
                metadata['keywords'] = ', '.join(keywords)

        return metadata

    def _determine_status(self, text: str) -> str:
        """Determine GOST status"""
        text_lower = text.lower()

        for status, patterns in self.status_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return status

        return 'неизвестен'

    def _extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extract document sections"""
        sections = []

        # Common section headers in GOST documents
        section_headers = [
            r'^\d+\.\s+([А-Я].+)',  # 1. SECTION TITLE
            r'^[А-Я]+\.\s+([А-Я].+)',  # A. SECTION TITLE
            r'^\s*([А-Я][А-ЯА-Я\s]{5,})\s*$',  # ALL CAPS HEADER
            r'^\s*\d+\.\d+\.\s+([А-Я].+)'  # 1.1. SECTION TITLE
        ]

        lines = text.split('\n')
        current_section = None

        for i, line in enumerate(lines):
            line = line.strip()

            # Check if line is a section header
            is_header = False
            header_text = ""

            for pattern in section_headers:
                match = re.match(pattern, line)
                if match:
                    is_header = True
                    header_text = match.group(1).strip()
                    break

            if is_header:
                # Save previous section if exists
                if current_section:
                    sections.append(current_section)

                # Start new section
                current_section = {
                    'header': header_text,
                    'content': '',
                    'line_start': i
                }
            elif current_section:
                # Add line to current section content
                if line and not line.startswith('---') and len(line) > 3:
                    current_section['content'] += line + '\n'

        # Add the last section
        if current_section:
            sections.append(current_section)

        # Clean up sections
        for section in sections:
            section['content'] = section['content'].strip()
            section['length'] = len(section['content'])

        return sections

    def _extract_tables_from_text(self, text: str) -> List[List[List[str]]]:
        """Extract tables from text"""
        tables = []

        # Look for table-like structures
        table_pattern = r'(\+[-]+\+[\n\r])?([^\n\r]*\|[^\n\r]*[\n\r])+'

        for match in re.finditer(table_pattern, text):
            table_text = match.group(0)
            rows = []

            for line in table_text.split('\n'):
                line = line.strip()
                if '|' in line:
                    # Split by pipe, clean cells
                    cells = [cell.strip() for cell in line.split('|')]
                    rows.append(cells)

            if len(rows) >= 2:  # At least header and one data row
                tables.append(rows)

        return tables

    def _extract_parameters(self, text: str) -> Dict[str, Any]:
        """Extract technical parameters"""
        parameters = {}

        # Common parameter patterns
        param_patterns = {
            'tolerance': [
                r'допуск[:\s]*([\d\.]+)\s*(?:мм|mm|мкм|µ)',
                r'tolerance[:\s]*([\d\.]+)\s*(?:mm|µm)'
            ],
            'roughness': [
                r'шероховатость[:\s]*Ra\s*([\d\.]+)',
                r'roughness[:\s]*Ra\s*([\d\.]+)'
            ],
            'thickness': [
                r'толщина[:\s]*([\d\.]+)\s*(?:мм|mm)',
                r'thickness[:\s]*([\d\.]+)\s*(?:mm)'
            ],
            'diameter': [
                r'диаметр[:\s]*([\d\.]+)\s*(?:мм|mm)',
                r'diameter[:\s]*([\d\.]+)\s*(?:mm)'
            ]
        }

        for param_name, patterns in param_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    parameters[param_name] = {
                        'value': float(match.group(1)),
                        'unit': self._extract_unit(match.group(0))
                    }
                    break

        # Extract material if mentioned
        material_match = re.search(
            r'Материал[:\s]*([^\n\.]+)',
            text, re.IGNORECASE
        )
        if material_match:
            parameters['material'] = material_match.group(1).strip()

        return parameters

    def _extract_unit(self, text: str) -> str:
        """Extract measurement unit from text"""
        units = {
            'мм': 'mm',
            'mm': 'mm',
            'мкм': 'µm',
            'µm': 'µm',
            'µ': 'µm',
            'см': 'cm',
            'cm': 'cm',
            'м': 'm',
            'm': 'm'
        }

        for unit_ru, unit_en in units.items():
            if unit_ru in text or unit_en in text:
                return unit_en

        return ''

    def get_required_fields(self) -> List[str]:
        return ['gost_type', 'gost_number', 'status', 'processed_at']