# dark_scene/regional_analyzer.py
import cv2
import numpy as np
from typing import List
from .models import SpatialAnalysisResult, DarkRegion, RegionalAnalysisResult

class RegionalAnalyzer:
    def __init__(self, dark_threshold: float = 0.20, min_area_pct: float = 0.005):
        self.dark_threshold = dark_threshold
        self.min_area_pct = min_area_pct

    def analyze(self, spatial: SpatialAnalysisResult) -> RegionalAnalysisResult:
        H, W = spatial.darkness_map.shape
        total_pixels = H * W
        
        # 1. Бинарная маска темноты
        dark_mask = (spatial.darkness_map < self.dark_threshold).astype(np.uint8)
        
        # Очистка от шума
        kernel = np.ones((3,3), np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 2. Connected Components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
        
        regions = []
        region_id_counter = 1
        
        for i in range(1, num_labels): # 0 - это фон (светлые области)
            area = stats[i, cv2.CC_STAT_AREA]
            area_pct = area / total_pixels
            
            # Игнорируем слишком мелкие пятна
            if area_pct < self.min_area_pct:
                continue
                
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            
            # Создаем локальную маску для этого региона
            region_mask = (labels == i)
            
            # Сбор статистики по региону
            mean_lum = float(np.mean(spatial.darkness_map[region_mask]))
            mean_var = float(np.mean(spatial.variance_map[region_mask]))
            mean_edge = float(np.mean(spatial.edge_density_map[region_mask]))
            mean_chroma = float(np.mean(spatial.chroma_map[region_mask]))
            
            region = DarkRegion(
                region_id=region_id_counter,
                mask=region_mask,
                bbox=(x, y, w, h),
                area_pct=area_pct,
                mean_lum=mean_lum,
                mean_variance=mean_var,
                mean_edge_density=mean_edge,
                mean_chroma=mean_chroma
            )
            regions.append(region)
            region_id_counter += 1
            
        total_dark_area_pct = float(np.mean(dark_mask))
        
        return RegionalAnalysisResult(
            regions=regions,
            global_dark_mask=dark_mask.astype(bool),
            total_dark_area_pct=total_dark_area_pct
        )