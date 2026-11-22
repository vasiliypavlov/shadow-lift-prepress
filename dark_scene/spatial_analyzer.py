# dark_scene/spatial_analyzer.py
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from .models import SpatialAnalysisResult
from .utils import rgb_to_oklab

class SpatialAnalyzer:
    def __init__(self):
        pass

    def analyze(self, image_rgb: np.ndarray) -> SpatialAnalysisResult:
        H, W = image_rgb.shape[:2]
        
        # 1. OKLAB
        oklab = rgb_to_oklab(image_rgb)
        L = np.clip(oklab[..., 0], 0.0, 1.0)
        a = oklab[..., 1]
        b = oklab[..., 2]

        # 2. Локальная дисперсия (многомасштабная)
        # Используем формулу дисперсии через средние значения
        L_blur_1 = cv2.blur(L, (5, 5))
        L_sq_blur_1 = cv2.blur(L**2, (5, 5))
        var_scale1 = np.maximum(0, L_sq_blur_1 - L_blur_1**2)

        L_blur_2 = cv2.blur(L, (15, 15))
        L_sq_blur_2 = cv2.blur(L**2, (15, 15))
        var_scale2 = np.maximum(0, L_sq_blur_2 - L_blur_2**2)
        
        # Нормализуем и объединяем
        variance_map = (var_scale1 * 0.5 + var_scale2 * 0.5)
        variance_map = np.clip(variance_map * 10.0, 0.0, 1.0) # Усиливаем для видимости

        # 3. Плотность границ (Sobel)
        grad_x = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        # Сглаживаем для получения карты плотности
        edge_density_map = gaussian_filter(gradient_mag, sigma=3.0)
        edge_density_map = np.clip(edge_density_map * 5.0, 0.0, 1.0)

        # 4. Карта насыщенности (Chroma)
        chroma_map = np.sqrt(a**2 + b**2)
        chroma_map = np.clip(chroma_map * 2.0, 0.0, 1.0) # Нормализация

        return SpatialAnalysisResult(
            darkness_map=L,
            variance_map=variance_map,
            edge_density_map=edge_density_map,
            chroma_map=chroma_map
        )