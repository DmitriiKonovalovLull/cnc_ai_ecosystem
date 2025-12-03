import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from .base_transformer import BaseTransformer


class ToolDataTransformer(BaseTransformer):
    def __init__(self):
        super().__init__()

        # Tool classification patterns
        self.tool_types = {
            'end_mill': ['фреза концевая', 'end mill', 'концевая фреза'],
            'face_mill': ['фреза торцевая', 'face mill', 'торцевая фреза'],
            'drill': ['сверло', 'drill', 'сверло спиральное'],
            'tap': ['метчик', 'tap', 'резьбонарезной'],
            'insert': ['пластина', 'insert', 'сменная пластина'],
            'holder': ['державка', 'holder', 'оправка'],
            'boring_bar': ['расточной резец', 'boring bar', 'расточная оправка']
        }

        # Material patterns
        self.material_types = {
            'hss': ['быстрорежущая сталь', 'hss', 'high speed steel'],
            'carbide': ['твердый сплав', 'carbide', 'tungsten carbide'],
            'cermet': ['керамика', 'cermet', 'керамический'],
            'diamond': ['алмаз', 'diamond', 'pcd'],
            'cbn': ['кубический нитрид бора', 'cbn', 'cubic boron nitride']
        }

    def transform(self, data: Any) -> Dict[str, Any]:
        """Transform tool data"""
        if isinstance(data, dict):
            return self._transform_from_dict(data)
        elif isinstance(data, list):
            return self._transform_batch(data)
        elif isinstance(data, str):
            return self._transform_from_text(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _transform_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform single tool record"""
        # Extract basic info
        tool_info = {
            "tool_id": data.get('id') or data.get('tool_id') or self._generate_tool_id(data),
            "name": data.get('name') or data.get('tool_name') or "",
            "manufacturer": data.get('manufacturer') or data.get('brand') or "",
            "catalog_number": data.get('catalog_number') or data.get('part_number') or "",
            "type": self._classify_tool_type(data),
            "material": self._classify_material(data),
            "specifications": self._extract_specifications(data),
            "cutting_parameters": self._extract_cutting_parameters(data),
            "geometries": self._extract_geometries(data),
            "coatings": self._extract_coatings(data),
            "applications": self._extract_applications(data),
            "metadata": {
                "source": data.get('source', 'unknown'),
                "processed_at": datetime.now().isoformat(),
                "data_quality": self._assess_data_quality(data)
            }
        }

        # Calculate derived fields
        tool_info["price_range"] = self._estimate_price_range(data)
        tool_info["lifespan"] = self._estimate_lifespan(data)
        tool_info["compatibility"] = self._determine_compatibility(data)

        return tool_info

    def _transform_batch(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Transform batch of tool records"""
        tools = []

        for item in data:
            try:
                tool = self._transform_from_dict(item)
                tools.append(tool)
            except Exception as e:
                # Skip invalid records but log
                continue

        return {
            "batch_info": {
                "total_records": len(data),
                "valid_records": len(tools),
                "processed_at": datetime.now().isoformat()
            },
            "tools": tools
        }

    def _transform_from_text(self, text: str) -> Dict[str, Any]:
        """Transform from unstructured text"""
        # Extract tool information using patterns
        extracted = {}

        # Look for tool name
        name_patterns = [
            r'Наименование[:\s]*([^\n]+)',
            r'Name[:\s]*([^\n]+)',
            r'Tool[:\s]*([^\n]+)'
        ]

        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted['name'] = match.group(1).strip()
                break

        # Look for manufacturer
        manufacturer_patterns = [
            r'Производитель[:\s]*([^\n]+)',
            r'Manufacturer[:\s]*([^\n]+)',
            r'Brand[:\s]*([^\n]+)'
        ]

        for pattern in manufacturer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted['manufacturer'] = match.group(1).strip()
                break

        # Look for catalog number
        catalog_patterns = [
            r'Артикул[:\s]*([^\n]+)',
            r'Catalog[:\s#]*([^\n]+)',
            r'Part[:\s#]*([^\n]+)'
        ]

        for pattern in catalog_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted['catalog_number'] = match.group(1).strip()
                break

        # Add the text as source
        extracted['source_text'] = text[:2000]  # Limit text length

        return self._transform_from_dict(extracted)

    def _generate_tool_id(self, data: Dict[str, Any]) -> str:
        """Generate unique tool ID"""
        import hashlib

        id_parts = [
            data.get('manufacturer', ''),
            data.get('catalog_number', ''),
            data.get('name', '')
        ]

        id_string = '_'.join(filter(None, id_parts))
        if not id_string:
            id_string = str(datetime.now().timestamp())

        return f"tool_{hashlib.md5(id_string.encode()).hexdigest()[:12]}"

    def _classify_tool_type(self, data: Dict[str, Any]) -> str:
        """Classify tool type"""
        # Check explicit type field
        explicit_type = data.get('type') or data.get('tool_type')
        if explicit_type:
            for type_name, patterns in self.tool_types.items():
                for pattern in patterns:
                    if pattern.lower() in str(explicit_type).lower():
                        return type_name

        # Infer from name or description
        text_to_check = ''
        for field in ['name', 'description', 'tool_name']:
            if field in data and data[field]:
                text_to_check += ' ' + str(data[field]).lower()

        for type_name, patterns in self.tool_types.items():
            for pattern in patterns:
                if pattern.lower() in text_to_check:
                    return type_name

        return 'unknown'

    def _classify_material(self, data: Dict[str, Any]) -> str:
        """Classify tool material"""
        # Check explicit material field
        explicit_material = data.get('material') or data.get('tool_material')
        if explicit_material:
            for material_name, patterns in self.material_types.items():
                for pattern in patterns:
                    if pattern.lower() in str(explicit_material).lower():
                        return material_name

        # Infer from name or description
        text_to_check = ''
        for field in ['name', 'description', 'specifications']:
            if field in data and data[field]:
                text_to_check += ' ' + str(data[field]).lower()

        for material_name, patterns in self.material_types.items():
            for pattern in patterns:
                if pattern.lower() in text_to_check:
                    return material_name

        return 'unknown'

    def _extract_specifications(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract tool specifications"""
        specs = {}

        # Common specification fields
        spec_fields = {
            'diameter': ['диаметр', 'diameter', 'd', 'Ø'],
            'length': ['длина', 'length', 'l'],
            'flute_count': ['число зубьев', 'flute count', 'z'],
            'shank_diameter': ['диаметр хвостовика', 'shank diameter', 'sd'],
            'overall_length': ['общая длина', 'overall length', 'oal'],
            'cutting_length': ['длина режущей части', 'cutting length', 'cl']
        }

        # Try to find specifications in data
        for spec_name, patterns in spec_fields.items():
            value = None
            unit = None

            # Check direct fields
            for pattern in patterns:
                if pattern in data:
                    value = data[pattern]
                    unit = self._infer_unit(pattern, str(value))
                    break

            # Check in nested structures
            if not value and 'specifications' in data and isinstance(data['specifications'], dict):
                for pattern in patterns:
                    if pattern in data['specifications']:
                        value = data['specifications'][pattern]
                        unit = self._infer_unit(pattern, str(value))
                        break

            if value is not None:
                specs[spec_name] = {
                    'value': self._parse_numeric(value),
                    'unit': unit or 'mm'
                }

        return specs

    def _extract_cutting_parameters(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract cutting parameters"""
        params = {}

        # Common cutting parameter fields
        param_fields = {
            'cutting_speed': ['скорость резания', 'cutting speed', 'vc', 'v_c'],
            'feed_rate': ['подача', 'feed rate', 'f', 'f_z'],
            'depth_of_cut': ['глубина резания', 'depth of cut', 'ap', 'a_p'],
            'feed_per_tooth': ['подача на зуб', 'feed per tooth', 'fz', 'f_z'],
            'spindle_speed': ['частота вращения', 'spindle speed', 'n', 'rpm']
        }

        for param_name, patterns in param_fields.items():
            value = None
            unit = None

            # Check various data locations
            check_locations = [
                data,
                data.get('cutting_parameters', {}),
                data.get('parameters', {}),
                data.get('recommendations', {})
            ]

            for location in check_locations:
                if isinstance(location, dict):
                    for pattern in patterns:
                        if pattern in location:
                            value = location[pattern]
                            unit = self._infer_cutting_unit(param_name, str(value))
                            break
                    if value is not None:
                        break

            if value is not None:
                params[param_name] = {
                    'value': self._parse_numeric(value),
                    'unit': unit or self._get_default_unit(param_name),
                    'material': data.get('workpiece_material')
                }

        return params

    def _extract_geometries(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool geometries"""
        geometries = []

        # Check for geometry information
        geometry_data = data.get('geometry') or data.get('geometries')

        if isinstance(geometry_data, dict):
            # Single geometry
            geometries.append({
                'type': geometry_data.get('type', 'unknown'),
                'angles': self._extract_angles(geometry_data),
                'radii': self._extract_radii(geometry_data),
                'clearances': self._extract_clearances(geometry_data)
            })
        elif isinstance(geometry_data, list):
            # Multiple geometries
            for geom in geometry_data:
                if isinstance(geom, dict):
                    geometries.append({
                        'type': geom.get('type', 'unknown'),
                        'angles': self._extract_angles(geom),
                        'radii': self._extract_radii(geom),
                        'clearances': self._extract_clearances(geom)
                    })

        return geometries

    def _extract_angles(self, geometry: Dict[str, Any]) -> Dict[str, float]:
        """Extract angles from geometry data"""
        angles = {}

        angle_fields = {
            'rake_angle': ['передний угол', 'rake angle', 'γ'],
            'clearance_angle': ['задний угол', 'clearance angle', 'α'],
            'wedge_angle': ['угол заострения', 'wedge angle', 'β'],
            'cutting_edge_angle': ['главный угол в плане', 'cutting edge angle', 'κ']
        }

        for angle_name, patterns in angle_fields.items():
            for pattern in patterns:
                if pattern in geometry:
                    value = geometry[pattern]
                    numeric = self._parse_numeric(value)
                    if numeric is not None:
                        angles[angle_name] = numeric
                        break

        return angles

    def _extract_radii(self, geometry: Dict[str, Any]) -> Dict[str, float]:
        """Extract radii from geometry data"""
        radii = {}

        radius_fields = {
            'corner_radius': ['радиус при вершине', 'corner radius', 'r_ε'],
            'nose_radius': ['радиус закругления', 'nose radius', 'r_n'],
            'cutting_edge_radius': ['радиус режущей кромки', 'cutting edge radius']
        }

        for radius_name, patterns in radius_fields.items():
            for pattern in patterns:
                if pattern in geometry:
                    value = geometry[pattern]
                    numeric = self._parse_numeric(value)
                    if numeric is not None:
                        radii[radius_name] = numeric
                        break

        return radii

    def _extract_clearances(self, geometry: Dict[str, Any]) -> Dict[str, float]:
        """Extract clearance information"""
        clearances = {}

        clearance_fields = {
            'primary_clearance': ['первичный зазор', 'primary clearance'],
            'secondary_clearance': ['вторичный зазор', 'secondary clearance'],
            'land_width': ['ширина ленточки', 'land width']
        }

        for clearance_name, patterns in clearance_fields.items():
            for pattern in patterns:
                if pattern in geometry:
                    value = geometry[pattern]
                    numeric = self._parse_numeric(value)
                    if numeric is not None:
                        clearances[clearance_name] = numeric
                        break

        return clearances

    def _extract_coatings(self, data: Dict[str, Any]) -> List[str]:
        """Extract tool coatings"""
        coatings = []

        # Check various coating fields
        coating_sources = [
            data.get('coating'),
            data.get('coatings'),
            data.get('surface_treatment')
        ]

        for source in coating_sources:
            if isinstance(source, str):
                # Split comma-separated list
                parts = [part.strip() for part in source.split(',')]
                coatings.extend(parts)
            elif isinstance(source, list):
                coatings.extend([str(item).strip() for item in source])

        # Remove duplicates and empty strings
        return list(set(filter(None, coatings)))

    def _extract_applications(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool applications"""
        applications = []

        app_data = data.get('applications') or data.get('recommended_applications')

        if isinstance(app_data, list):
            for app in app_data:
                if isinstance(app, dict):
                    applications.append({
                        'material': app.get('material'),
                        'operation': app.get('operation'),
                        'conditions': app.get('conditions'),
                        'efficiency': app.get('efficiency')
                    })

        return applications

    def _assess_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality of tool data"""
        quality_score = 0
        max_score = 10
        missing_fields = []

        # Check required fields
        required_fields = ['name', 'manufacturer', 'type']
        for field in required_fields:
            if field in data and data[field]:
                quality_score += 2
            else:
                missing_fields.append(field)

        # Check specifications
        if 'specifications' in data and data['specifications']:
            quality_score += 2

        # Check cutting parameters
        if 'cutting_parameters' in data and data['cutting_parameters']:
            quality_score += 2

        # Check additional data
        additional_fields = ['catalog_number', 'material', 'coatings']
        for field in additional_fields:
            if field in data and data[field]:
                quality_score += 1

        return {
            'score': quality_score,
            'percentage': (quality_score / max_score) * 100,
            'missing_fields': missing_fields,
            'completeness': len(data.keys()) / 20  # Rough measure
        }

    def _estimate_price_range(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Estimate price range based on tool specifications"""
        # Simple estimation based on tool type and material
        base_prices = {
            'end_mill': {'carbide': 50, 'hss': 20},
            'drill': {'carbide': 30, 'hss': 10},
            'insert': {'carbide': 5, 'cermet': 8},
            'tap': {'hss': 15, 'carbide': 40}
        }

        tool_type = self._classify_tool_type(data)
        material = self._classify_material(data)

        base_price = base_prices.get(tool_type, {}).get(material, 10)

        # Adjust based on specifications
        if 'specifications' in data and isinstance(data['specifications'], dict):
            specs = data['specifications']
            if 'diameter' in specs:
                diameter = specs['diameter'].get('value', 0)
                if diameter > 0:
                    # Larger tools are more expensive
                    base_price *= (diameter / 10)

        return {
            'low': base_price * 0.7,
            'high': base_price * 1.3,
            'currency': 'USD'
        }

    def _estimate_lifespan(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate tool lifespan"""
        # Very rough estimation
        material_factors = {
            'carbide': 1.0,
            'cermet': 1.2,
            'diamond': 3.0,
            'cbn': 2.5,
            'hss': 0.3
        }

        material = self._classify_material(data)
        base_lifespan = material_factors.get(material, 0.5) * 60  # minutes

        # Adjust based on coatings
        coatings = self._extract_coatings(data)
        coating_factor = 1.0
        if coatings:
            coating_factor = 1.2  # Coatings typically improve lifespan

        return {
            'estimated_minutes': base_lifespan * coating_factor,
            'confidence': 'low',
            'factors': {
                'material': material,
                'coatings': coatings,
                'coating_factor': coating_factor
            }
        }

    def _determine_compatibility(self, data: Dict[str, Any]) -> List[str]:
        """Determine machine compatibility"""
        compatibilities = []

        # Check shank type
        shank_info = data.get('shank') or data.get('shank_type')
        if shank_info:
            shank_type = str(shank_info).lower()

            if 'bt' in shank_type or 'iso' in shank_type:
                compatibilities.append('bt_flange')
            if 'cat' in shank_type:
                compatibilities.append('cat_flange')
            if 'hs' in shank_type:
                compatibilities.append('hs_flange')
            if 'sk' in shank_type:
                compatibilities.append('sk_flange')

        # Check holder type
        holder_info = data.get('holder') or data.get('holder_type')
        if holder_info:
            holder_type = str(holder_info).lower()

            if 'er' in holder_type:
                compatibilities.append('er_collet')
            if 'tg' in holder_type:
                compatibilities.append('tg_collet')
            if 'hydraulic' in holder_type:
                compatibilities.append('hydraulic_holder')
            if 'shrink' in holder_type:
                compatibilities.append('shrink_fit')

        return list(set(compatibilities))

    def _parse_numeric(self, value: Any) -> Optional[float]:
        """Parse numeric value from various formats"""
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            # Remove non-numeric characters except dots and minus
            clean_str = re.sub(r'[^\d\.\-]', '', value)
            try:
                return float(clean_str) if clean_str else None
            except ValueError:
                return None

        return None

    def _infer_unit(self, field_name: str, value: str) -> Optional[str]:
        """Infer unit from field name and value"""
        if 'diameter' in field_name.lower() or 'радиус' in field_name.lower():
            return 'mm'
        elif 'length' in field_name.lower() or 'длина' in field_name.lower():
            return 'mm'
        elif 'angle' in field_name.lower() or 'угол' in field_name.lower():
            return 'deg'

        # Check value for unit indicators
        if 'mm' in value.lower():
            return 'mm'
        elif 'inch' in value.lower() or '"' in value:
            return 'inch'
        elif 'deg' in value.lower() or '°' in value:
            return 'deg'

        return None

    def _infer_cutting_unit(self, param_name: str, value: str) -> Optional[str]:
        """Infer unit for cutting parameters"""
        if 'speed' in param_name:
            if 'm/min' in value or 'м/мин' in value:
                return 'm/min'
            elif 'sfm' in value:
                return 'sfm'
        elif 'feed' in param_name:
            if 'mm/rev' in value or 'мм/об' in value:
                return 'mm/rev'
            elif 'mm/min' in value or 'мм/мин' in value:
                return 'mm/min'
            elif 'ipr' in value:
                return 'ipr'
        elif 'rpm' in param_name or 'speed' in param_name:
            return 'rpm'

        return None

    def _get_default_unit(self, param_name: str) -> str:
        """Get default unit for parameter"""
        defaults = {
            'cutting_speed': 'm/min',
            'feed_rate': 'mm/rev',
            'depth_of_cut': 'mm',
            'feed_per_tooth': 'mm/tooth',
            'spindle_speed': 'rpm'
        }

        return defaults.get(param_name, '')

    def get_required_fields(self) -> List[str]:
        return ['tool_id', 'name', 'type', 'material']