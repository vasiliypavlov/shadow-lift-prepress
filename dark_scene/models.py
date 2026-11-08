# dark_scene/models.py
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
import numpy as np

@dataclass
class SpatialAnalysisResult:
    """Результаты пространственного анализа изображения."""
    darkness_map: np.ndarray       # L-канал OKLAB [0,1]
    variance_map: np.ndarray       # Локальная дисперсия
    edge_density_map: np.ndarray   # Плотность границ
    chroma_map: np.ndarray         # Насыщенность (a,b)
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "darkness": self.darkness_map,
            "variance": self.variance_map,
            "edges": self.edge_density_map,
            "chroma": self.chroma_map
        }

@dataclass
class DarkRegion:
    """Описание отдельного темного региона."""
    region_id: int
    mask: np.ndarray               # Бинарная маска региона
    bbox: Tuple[int, int, int, int] # x, y, w, h
    area_pct: float                # % площади от всего изображения
    
    # Усредненные характеристики по региону
    mean_lum: float = 0.0
    mean_variance: float = 0.0
    mean_edge_density: float = 0.0
    mean_chroma: float = 0.0
    
    # Для будущего (Этап 2)
    darkness_intent: str = "UNKNOWN"
    confidence: float = 0.0

@dataclass
class RegionalAnalysisResult:
    """Общий результат регионального анализа."""
    regions: List[DarkRegion]
    global_dark_mask: np.ndarray
    total_dark_area_pct: float
    
@dataclass
class DarkSceneMasks:
    """Плавные управляющие маски для интеграции в движок."""
    preservation_mask: np.ndarray   # 1.0 = сохранить черноту, 0.0 = можно трогать
    recovery_mask: np.ndarray       # 1.0 = разрешено восстановление
    uncertainty_mask: np.ndarray    # 1.0 = высокая неопределенность