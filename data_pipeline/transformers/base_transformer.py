from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
import json


class BaseTransformer(ABC):
    def __init__(self):
        self.supported_formats = ['json', 'dict', 'dataframe']

    @abstractmethod
    def transform(self, data: Any) -> Dict[str, Any]:
        """Transform raw data into standardized format"""
        pass

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate transformed data"""
        required_fields = self.get_required_fields()
        return all(field in data for field in required_fields)

    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """Get list of required fields"""
        pass

    def to_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Convert transformed data to DataFrame"""
        # Flatten nested structures
        flat_data = self._flatten_dict(data)
        return pd.DataFrame([flat_data])

    def to_json(self, data: Dict[str, Any]) -> str:
        """Convert transformed data to JSON"""
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _flatten_dict(self, data: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to strings for DataFrame compatibility
                items.append((new_key, json.dumps(v, ensure_ascii=False)))
            else:
                items.append((new_key, v))

        return dict(items)

    def normalize_text(self, text: str) -> str:
        """Normalize text (remove extra spaces, normalize encoding)"""
        if not text:
            return ""

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Normalize quotes and dashes
        replacements = {
            '«': '"',
            '»': '"',
            '„': '"',
            '“': '"',
            '”': '"',
            '–': '-',
            '—': '-',
            ' ': ' ',  # non-breaking space
            '\xa0': ' '  # another non-breaking space
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text.strip()