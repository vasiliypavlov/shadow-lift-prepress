# dark_scene/diagnostic_report.py
import os
import json
import numpy as np
from PIL import Image
from .models import SpatialAnalysisResult, RegionalAnalysisResult, DarkSceneMasks

class DiagnosticReport:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _save_map(self, arr: np.ndarray, name: str, stem: str):
        path = os.path.join(self.output_dir, f"{stem}_{name}.png")
        vis = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(vis).save(path)

    def save(self, stem: str, spatial: SpatialAnalysisResult, regional: RegionalAnalysisResult, masks: DarkSceneMasks):
        # Сохранение базовых карт
        self._save_map(spatial.darkness_map, "_dbg_darkness_map", stem)
        self._save_map(spatial.variance_map, "_dbg_darkness_variance", stem)
        self._save_map(spatial.edge_density_map, "_dbg_edge_density", stem)
        self._save_map(spatial.chroma_map, "_dbg_chroma", stem)
        self._save_map(masks.preservation_mask, "_dbg_dark_preservation", stem)
        self._save_map(masks.recovery_mask, "_dbg_recovery_permission", stem)
        self._save_map(masks.uncertainty_mask, "_dbg_dark_uncertainty", stem)
        
        # --- НОВОЕ: Darkness Fusion Map (Раздел 12.2) ---
        fusion_vis = np.zeros((spatial.darkness_map.shape[0], spatial.darkness_map.shape[1], 3), dtype=np.uint8)
        
        for r in regional.regions:
            if r.darkness_intent == "PRESERVE":
                color = [255, 50, 50]    # Красный (сохранить)
            elif r.darkness_intent == "RECOVER":
                color = [50, 255, 50]    # Зеленый (восстановить)
            else:
                color = [255, 255, 50]   # Желтый (transition)
                
            fusion_vis[r.mask] = color
            
        path_fusion = os.path.join(self.output_dir, f"{stem}_dbg_dark_fusion.png")
        Image.fromarray(fusion_vis).save(path_fusion)

        # Сохранение JSON отчета (добавлены intent и confidence)
        report_data = {
            "image_stem": stem,
            "total_dark_area_pct": regional.total_dark_area_pct,
            "regions_count": len(regional.regions),
            "regions": [
                {
                    "id": r.region_id,
                    "area_pct": round(r.area_pct * 100, 2),
                    "mean_lum": round(r.mean_lum, 4),
                    "mean_variance": round(r.mean_variance, 4),
                    "mean_edge_density": round(r.mean_edge_density, 4),
                    "darkness_intent": r.darkness_intent,
                    "confidence": r.confidence
                }
                for r in regional.regions
            ]
        }
        
        json_path = os.path.join(self.output_dir, f"{stem}_dbg_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)