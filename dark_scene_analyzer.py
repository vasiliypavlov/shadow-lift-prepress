#!/usr/bin/env python3
# dark_scene_analyzer.py
"""
Dark Scene Analyzer v1.6.0 (Stage 1: Diagnostic mode)
"""
import argparse
import os
import sys
import glob
import numpy as np
from PIL import Image
from dark_scene.darkness_intent import DarknessIntentAnalyzer
from dark_scene.mask_generator import MaskGenerator

# Импорт наших новых модулей
from dark_scene.spatial_analyzer import SpatialAnalyzer
from dark_scene.regional_analyzer import RegionalAnalyzer
from dark_scene.diagnostic_report import DiagnosticReport

def main():
    parser = argparse.ArgumentParser(description="Dark Scene Analyzer v1.6.0 (Diagnostic)")
    parser.add_argument("-i", "--input", required=True, 
                        help="Path to an input image OR a directory with images")
    parser.add_argument("-o", "--output-dir", required=True, 
                        help="Directory to save diagnostic files")
    parser.add_argument("--dark-threshold", type=float, default=0.20, 
                        help="Luminance threshold (OKLAB) for dark areas [0.0-1.0]. Default 0.20")
    args = parser.parse_args()

    # 1. Сбор файлов (файл или директория)
    is_single_file = os.path.isfile(args.input)
    
    if is_single_file:
        files = [args.input]
    elif os.path.isdir(args.input):
        exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.PNG", "*.JPG", "*.JPEG")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(args.input, ext)))
        files = sorted(list(set(files))) # Убираем дубликаты и сортируем
    else:
        sys.exit(f"Error: Input path not found: {args.input}")

    if not files:
        sys.exit(f"Error: No matching images found in: {args.input}")

    print("=" * 60)
    print(f" Dark Scene Analyzer (v1.6.0 Stage 1) - {len(files)} file(s)")
    print("=" * 60)
    
    # Создаем папку для результатов
    os.makedirs(args.output_dir, exist_ok=True)

    # Инициализируем анализаторы
    spatial_analyzer = SpatialAnalyzer()
    regional_analyzer = RegionalAnalyzer(dark_threshold=args.dark_threshold)
    intent_analyzer = DarknessIntentAnalyzer()
    mask_generator = MaskGenerator() # <-- НОВОЕ
    report = DiagnosticReport(args.output_dir)

    # 2. Цикл по файлам
    for idx, fpath in enumerate(files, 1):
        filename = os.path.basename(fpath)
        print(f"\n[{idx}/{len(files)}] Processing: {filename}")
        
        try:
            # Загрузка
            img_pil = Image.open(fpath).convert("RGB")
            img_rgb = np.asarray(img_pil, dtype=np.float32)
            stem = os.path.splitext(filename)[0]

            # Spatial Analysis
            spatial_res = spatial_analyzer.analyze(img_rgb)

            # Regional Analysis
            regional_res = regional_analyzer.analyze(spatial_res)
            
            # Darkness Intent Analysis
            regional_res.regions = intent_analyzer.analyze_all(regional_res.regions)
            
            # --- НОВОЕ: Mask Generation ---
            masks = mask_generator.generate(spatial_res, regional_res)
            
            # Diagnostic Report
            report.save(stem, spatial_res, regional_res, masks)

        except Exception as e:
            print(f"       ERROR processing {filename}: {e}", file=sys.stderr)

    print("\n" + "=" * 60)
    print(f" DONE. Processed {len(files)} file(s).")
    print(f" Check diagnostic files in: {args.output_dir}")

if __name__ == "__main__":
    main()