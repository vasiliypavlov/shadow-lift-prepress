# dark_scene/mask_generator.py
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from .models import RegionalAnalysisResult, SpatialAnalysisResult, DarkSceneMasks
from .utils import fast_guided_filter

class MaskGenerator:
    def __init__(self, feather_radius: int = 25):
        self.feather_radius = feather_radius

    def generate(self, spatial: SpatialAnalysisResult, regional: RegionalAnalysisResult) -> DarkSceneMasks:
        H, W = spatial.darkness_map.shape
        
        preservation = np.zeros((H, W), dtype=np.float32)
        recovery = np.zeros((H, W), dtype=np.float32)
        uncertainty = np.zeros((H, W), dtype=np.float32)

        for region in regional.regions:
            mask = region.mask.astype(np.float32)
            
            if region.darkness_intent == "PRESERVE":
                preservation = np.maximum(preservation, mask)
                uncertainty = np.maximum(uncertainty, mask * (1.0 - region.confidence))
                
            elif region.darkness_intent == "RECOVER":
                recovery = np.maximum(recovery, mask)
                uncertainty = np.maximum(uncertainty, mask * (1.0 - region.confidence))
                
            elif region.darkness_intent == "TRANSITION":
                # Переходная зона получает мягкий промежуточный вес
                preservation = np.maximum(preservation, mask * 0.3)
                recovery = np.maximum(recovery, mask * 0.7)
                uncertainty = np.maximum(uncertainty, mask * (1.0 - region.confidence))

        guide = spatial.darkness_map
        
        # Многопроходный размывающий шлюз (Guided Filter + Gaussian Blur)
        preservation = fast_guided_filter(guide, preservation, r=self.feather_radius, eps=0.01)
        preservation = gaussian_filter(preservation, sigma=min(H, W) * 0.03)

        recovery = fast_guided_filter(guide, recovery, r=self.feather_radius, eps=0.01)
        uncertainty = gaussian_filter(uncertainty, sigma=5.0)

        return DarkSceneMasks(
            preservation_mask=np.clip(preservation, 0.0, 1.0),
            recovery_mask=np.clip(recovery, 0.0, 1.0),
            uncertainty_mask=np.clip(uncertainty, 0.0, 1.0)
        )