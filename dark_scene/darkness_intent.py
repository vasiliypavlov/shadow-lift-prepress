# dark_scene/darkness_intent.py
# VERSION 2.2 — Sky Prior & Geometric Spatial Awareness Edition
import numpy as np
from .models import DarkRegion

class DarknessIntentAnalyzer:
    """
    Анализирует характеристики региона и присваивает ему статус:
    PRESERVE, RECOVER или TRANSITION (согласно разделу 8 ТЗ).
    v2.2: Добавлен Sky Prior (пространственный приоритет верхнего полушария кадра)
    для корректного разделения ночного неба и боковых стен зданий.
    """
    def __init__(self):
        self.edge_thresh = 0.08
        self.variance_thresh = 0.002

    def analyze_region(self, region: DarkRegion, image_height: int = 1000) -> DarkRegion:
        area = region.area_pct
        var = region.mean_variance
        edge = region.mean_edge_density
        x, y, w, h = region.bbox

        # Оценка вертикального положения региона (нормированная от 0.0 до 1.0)
        y_center_norm = (y + h / 2.0) / max(image_height, 1)

        # 1. Абсолютная пустота (Космос, глубокий черный)
        if var < 0.0005 and edge < 0.06:
            region.darkness_intent = "PRESERVE"
            region.confidence = 0.95
            return region

        # 2. Крупные регионы (Более 15% изображения)
        if area > 0.15:
            # 2.1 Sky Prior: Если регион находится в верхних 35% кадра и достаточно гладкий, 
            # это гарантированно ночное небо — защищаем (PRESERVE).
            if y_center_norm < 0.35 and var < 0.008:
                region.darkness_intent = "PRESERVE"
                region.confidence = 0.95
                return region

            # 2.2 Крупные боковые объекты (Стены зданий, индустриальные элементы по бокам)
            # Даже если они темные и относительно гладкие, они находятся ниже/сбоку (y_center_norm >= 0.35).
            # Отправляем их в RECOVER/TRANSITION, чтобы убрать резкие ступенчатые обрывы масок.
            if var < 0.005 and y_center_norm < 0.6:
                region.darkness_intent = "TRANSITION"
                region.confidence = 0.75
                return region
            else:
                region.darkness_intent = "RECOVER"
                region.confidence = 0.80
                return region

        # 3. Объекты и структуры (Передний план, мелкие детали)
        if edge > self.edge_thresh or var > self.variance_thresh:
            region.darkness_intent = "RECOVER"
            region.confidence = 0.85
            return region

        # 4. По умолчанию (Промежуточные состояния)
        region.darkness_intent = "TRANSITION"
        region.confidence = 0.5
        return region

    def analyze_all(self, regions: list[DarkRegion], image_height: int = 1000) -> list[DarkRegion]:
        return [self.analyze_region(r, image_height) for r in regions]