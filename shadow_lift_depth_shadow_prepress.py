"""
Shadow-Lift Depth/Shadow Prepress Engine
========================================

VERSION 1.0.9 (GitHub Release Edition)

ЦЕЛЬ ПРОЕКТА
------------
Пакетная предпечатная подготовка сложных Low-Key изображений для типографии (The Game Crafter).
Задача — вытянуть детали геометрии из глубины теней, скомпенсировать физическое растискивание 
точки и "проклятый черный фильтр" TGC, сохранив атмосферу, цвет (Colored Black) и контраст.

ОСОБЕННОСТИ РЕЛИЗА v1.0.9
-------------------------
- Железобетонная обработка путей вывода (-o как файл или директория).
- Автоматическая защита от перезаписи оригиналов (добавляет _lifted).
- Раздельная генерация калибровочных мишеней.
- Автоматический датчик и компенсатор Красной зоны смерти TGC (--tgc-lift auto).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

VERSION = "1.0.9"
MIN_BLACK_DEFAULT = 14 
MAX_WHITE_DEFAULT = 245
MAPS_DIR_NAME = "_prepress_maps"


# ============================================================
# КОНТРАКТ ДАННЫХ
# ============================================================

@dataclass
class ImageAssets:
    image: np.ndarray                                 
    lum: np.ndarray                                   
    depth: Optional[np.ndarray] = None                
    shadow: Optional[np.ndarray] = None               
    meta: dict = field(default_factory=dict)
    debug_data: dict = field(default_factory=dict)    


def validate_assets(a: ImageAssets) -> None:
    H, W = a.image.shape[:2]
    assert a.image.ndim == 3 and a.image.shape[2] == 3
    assert a.lum.shape == (H, W)
    for name in ("depth", "shadow"):
        arr = getattr(a, name)
        if arr is None: continue
        assert arr.shape == (H, W), f"{name} shape mismatch"


# ============================================================
# УТИЛИТЫ И ПУТИ
# ============================================================

def luminance_rgb(arr: np.ndarray) -> np.ndarray:
    return (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)

def stable_seed(input_path: str, base_seed: int = 42) -> int:
    digest = hashlib.sha256(os.path.abspath(input_path).encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "little") + int(base_seed)) % (2**32)

def get_maps_dir(source_path: str) -> str:
    if os.path.isdir(source_path):
        base = os.path.abspath(source_path)
    else:
        base = os.path.dirname(os.path.abspath(source_path))
    drive, tail = os.path.splitdrive(base)
    if not tail or tail == os.sep:
        base = os.path.expanduser("~")
    return os.path.join(base, MAPS_DIR_NAME)

def map_cache_path(source_path: str, stem: str, w: int, h: int, kind: str) -> str:
    return os.path.join(get_maps_dir(source_path), f"{stem}_{w}x{h}_{kind}.png")

def parse_size(size_str: Optional[str]) -> Optional[Tuple[int, int]]:
    if not size_str: return None
    try:
        w, h = size_str.lower().split("x") if "x" in size_str.lower() else size_str.split(",")
        return int(w.strip()), int(h.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(f"Bad size: {size_str}.")


# ============================================================
# МАТЕМАТИКА: sRGB <-> OKLAB
# ============================================================

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c_safe = np.maximum(c, 0.0)
    return np.where(c_safe <= 0.0031308, c_safe * 12.92, 1.055 * (c_safe ** (1/2.4)) - 0.055)

def rgb_to_oklab(rgb_arr: np.ndarray) -> np.ndarray:
    rgb = rgb_arr / 255.0
    rgb_lin = _srgb_to_linear(rgb)
    lms = np.dot(rgb_lin, np.array([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005]
    ]).T)
    lms_ = np.sign(lms) * (np.abs(lms) ** (1/3))
    oklab = np.dot(lms_, np.array([
        [0.2104542553,  0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050,  0.4505937099],
        [0.0259040371,  0.7827717662, -0.8086757660]
    ]).T)
    return oklab.astype(np.float32)

def oklab_to_rgb(oklab: np.ndarray) -> np.ndarray:
    lms_ = np.dot(oklab, np.array([
        [1.0,  0.3963377774,  0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480]
    ]).T)
    lms = lms_ ** 3
    rgb_lin = np.dot(lms, np.array([
        [ 4.0767416621, -3.3077115913,  0.2309699292],
        [-1.2684380046,  2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147,  1.7076147010]
    ]).T)
    rgb = _linear_to_srgb(rgb_lin)
    return np.clip(rgb * 255.0, 0, 255).astype(np.float32)

def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ============================================================
# ГЕНЕРАТОРЫ КАЛИБРОВОЧНЫХ ПАТТЕРНОВ TGC
# ============================================================

def generate_tgc_probes(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Черная мишень
    print("  [PROBE] Generating tgc_blackbox_probe.png...", file=sys.stderr)
    W, H = 900, 1500
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(0, 350):
        val = int((y / 350.0) * 60)
        canvas[y, :] = [val, val, val]

    steps = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50]
    cols = 4
    rows = len(steps) // cols
    cell_w, cell_h = W // cols, 400 // rows
    for i, val in enumerate(steps):
        r, c = i // cols, i % cols
        x0, y0 = c * cell_w, 350 + r * cell_h
        canvas[y0:y0+cell_h, x0:x0+cell_w] = [val, val, val]

    colors = [
        (14, 14, 14), (20, 10, 10), (10, 20, 10), (10, 10, 20),
        (14, 20, 25), (25, 14, 10), (20, 14, 25), (25, 25, 10),
        (8, 8, 8),    (16, 8, 8),   (8, 16, 8),   (8, 8, 16),
        (30, 15, 10), (10, 30, 20), (15, 10, 30), (30, 30, 30)
    ]
    cell_h_c = 350 // 4
    for i, col in enumerate(colors):
        r, c = i // 4, i % 4
        x0, y0 = c * cell_w, 750 + r * cell_h_c
        canvas[y0:y0+cell_h_c, x0:x0+cell_w] = col

    canvas[1100:1500, :450] = 10
    canvas[1100:1500, 450:] = 14
    img_b = Image.fromarray(canvas)
    draw_b = ImageDraw.Draw(img_b)
    for idx, delta in enumerate([2, 5, 10, 15]):
        y_pos = [1150, 1250, 1350, 1430][idx]
        for w_line in range(1, 5):
            draw_b.line([(20+w_line*30, y_pos), (20+w_line*30, y_pos+60)], fill=10+delta, width=w_line)
            draw_b.line([(470+w_line*30, y_pos), (470+w_line*30, y_pos+60)], fill=14+delta, width=w_line)
    
    path_b = os.path.join(out_dir, "tgc_blackbox_probe.png")
    img_b.save(path_b)

    # 2. Белая мишень
    print("  [PROBE] Generating tgc_highlight_probe.png...", file=sys.stderr)
    canvas_w = np.full((H, W, 3), 255, dtype=np.uint8)
    for y in range(0, 350):
        val = int(180 + (y / 350.0) * 75)
        canvas_w[y, :] = [val, val, val]

    h_steps = [180, 190, 200, 210, 220, 230, 240, 243, 245, 248, 250, 252, 254, 255]
    h_rows = (len(h_steps) + cols - 1) // cols
    h_cell_h = 400 // h_rows
    for i, val in enumerate(h_steps):
        r, c = i // cols, i % cols
        x0, y0 = c * cell_w, 350 + r * h_cell_h
        canvas_w[y0:y0+h_cell_h, x0:x0+cell_w] = [val, val, val]

    pastel_colors = [
        (255, 240, 240), (240, 255, 240), (240, 240, 255), (255, 255, 240),
        (250, 230, 220), (220, 250, 230), (230, 220, 250), (250, 250, 220),
        (245, 220, 200), (200, 245, 220), (220, 200, 245), (245, 245, 200),
        (240, 210, 190), (190, 240, 210), (210, 190, 240), (240, 240, 190)
    ]
    for i, col in enumerate(pastel_colors):
        r, c = i // 4, i % 4
        x0, y0 = c * cell_w, 750 + r * (350 // 4)
        canvas_w[y0:y0+(350 // 4), x0:x0+cell_w] = col

    canvas_w[1100:1500, :] = 245
    img_w = Image.fromarray(canvas_w)
    draw_w = ImageDraw.Draw(img_w)
    for idx, val in enumerate([220, 200, 150, 100]):
        y_pos = [1150, 1250, 1350, 1430][idx]
        for w_line in range(1, 5):
            draw_w.line([(50+w_line*60, y_pos), (50+w_line*60, y_pos+60)], fill=val, width=w_line)
            draw_w.line([(500+w_line*60, y_pos), (500+w_line*60, y_pos+60)], fill=(val, val//2, 50), width=w_line)

    path_w = os.path.join(out_dir, "tgc_highlight_probe.png")
    img_w.save(path_w)
    print(f"  [PROBE] Successfully generated both probes in: {out_dir}", file=sys.stderr)


# ============================================================
# ТОЧНЫЙ СИМУЛЯТОР ФИЛЬТРА TGC
# ============================================================

def simulate_tcg_color_filter(arr: np.ndarray) -> np.ndarray:
    norm = arr / 255.0
    black_clip = 0.09
    norm = np.maximum(norm - black_clip, 0.0) / (1.0 - black_clip)
    gamma = 1.25
    norm = np.power(norm, gamma)

    lum = (0.299 * norm[..., 0] + 0.587 * norm[..., 1] + 0.114 * norm[..., 2])[..., None]
    max_val = np.max(norm, axis=-1, keepdims=True)
    desat_mask = 1.0 - np.clip(max_val / 0.25, 0.0, 1.0)
    norm = norm * (1.0 - desat_mask * 0.8) + lum * (desat_mask * 0.8)

    white_point = 0.95
    norm = norm * white_point
    return np.clip(norm * 255.0, 0, 255).astype(np.float32)


# ============================================================
# ШАГ 1: ЗАГРУЗКА И АНАЛИЗ
# ============================================================

class DepthEstimator:
    _instance: Optional["DepthEstimator"] = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ready = False
        return cls._instance
    def _ensure_loaded(self) -> None:
        if self._ready: return
        import torch
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  [DEPTH] Loading MiDaS_small on {self.device}", file=sys.stderr)
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = transforms.small_transform
        self._ready = True
    def infer(self, rgb_float: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        torch = self.torch
        H, W = rgb_float.shape[:2]
        img_u8 = np.clip(rgb_float, 0, 255).astype(np.uint8)
        batch = self.transform(img_u8).to(self.device)
        with torch.no_grad():
            pred = self.midas(batch)
            pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False).squeeze()
        d = pred.cpu().numpy().astype(np.float32)
        dmin, dmax = float(d.min()), float(d.max())
        return (d - dmin) / (dmax - dmin) if dmax - dmin > 1e-6 else np.zeros_like(d)

class ClassicalShadowDetector:
    name = "classical"
    def detect(self, image_rgb: np.ndarray) -> np.ndarray:
        L = luminance_rgb(image_rgb)
        sigma = max(image_rgb.shape[:2]) * 0.05
        L_local = gaussian_filter(L, sigma=sigma)
        ratio = L / np.maximum(L_local, 1.0)
        darkness = np.clip(1.0 - ratio, 0.0, 1.0).astype(np.float32)
        prob = gaussian_filter(darkness, sigma=2.0)
        return np.clip(prob, 0.0, 1.0).astype(np.float32)

def _save_map_png(arr: np.ndarray, path: str, label: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8)).save(path, format="PNG")
    print(f"  [{label}] saved: {path}", file=sys.stderr)

def extract_assets(input_path: str, target_size: Optional[Tuple[int, int]] = None, want_depth: bool = True, want_shadow: bool = False) -> ImageAssets:
    img = Image.open(input_path).convert("RGB")
    if target_size: img = img.resize(target_size, Image.Resampling.LANCZOS)
    image = np.asarray(img, dtype=np.float32)
    H, W = image.shape[:2]
    
    assets = ImageAssets(image=image, lum=luminance_rgb(image), meta={
        "input_path": os.path.abspath(input_path),
        "size_wh": (W, H),
        "seed": stable_seed(input_path),
        "maps_dir": get_maps_dir(input_path)
    })

    if want_depth:
        stem = os.path.splitext(os.path.basename(input_path))[0]
        cache_path = map_cache_path(input_path, stem, W, H, "depth")
        if os.path.isfile(cache_path):
            d_img = Image.open(cache_path).convert("L")
            if d_img.size == (W, H):
                assets.depth = np.asarray(d_img, dtype=np.float32) / 255.0
        if assets.depth is None:
            assets.depth = 1.0 - DepthEstimator().infer(image) 
            _save_map_png(assets.depth, cache_path, "DEPTH")

    if want_shadow:
        assets.shadow = ClassicalShadowDetector().detect(image)
        stem = os.path.splitext(os.path.basename(input_path))[0]
        _save_map_png(assets.shadow, map_cache_path(input_path, stem, W, H, "shadow"), "SHADOW")

    validate_assets(assets)
    return assets


# ============================================================
# ШАГ 2: ДВИЖОК ЛИФТИНГА И PREPRESS HDR
# ============================================================

def apply_lifting_engine(
    assets: ImageAssets, 
    min_black: int = MIN_BLACK_DEFAULT,
    max_white: int = MAX_WHITE_DEFAULT,
    lift_strength: str = "auto",
    depth_protection: float = 0.5,
    detail_boost: float = 1.5, 
    prepress_hdr: float = 1.0,     
    cmyk_compression: float = 0.0, 
    tgc_lift: Union[str, float] = "0.0",
    debug: bool = False
) -> ImageAssets:
    
    arr = assets.image.copy()
    H, W = arr.shape[:2]
    rng = np.random.default_rng(assets.meta["seed"])

    oklab = rgb_to_oklab(arr)
    L, a, b = oklab[..., 0], oklab[..., 1], oklab[..., 2]
    p15, p50 = np.percentile(L, [15, 50])
    
    if lift_strength == "auto":
        strength_val = 1.0 if p15 < 0.1 else (0.7 if p15 < 0.2 else 0.4)
    else:
        strength_val = float(lift_strength)

    if isinstance(tgc_lift, str) and tgc_lift.lower() == "auto":
        temp_sim = simulate_tcg_color_filter(arr)
        temp_sim_lum = luminance_rgb(temp_sim)
        crushed_ratio = np.sum(temp_sim_lum <= 5) / (H * W)
        
        if crushed_ratio > 0.20:
            resolved_tgc_lift = 1.5
        elif crushed_ratio > 0.08:
            resolved_tgc_lift = 1.0
        elif crushed_ratio > 0.02:
            resolved_tgc_lift = 0.5
        else:
            resolved_tgc_lift = 0.0
            
        print(f"  [TGC-LIFT AUTO] Crushed pixels ratio: {crushed_ratio*100:.1f}% -> Resolved tgc_lift = {resolved_tgc_lift}", file=sys.stderr)
    else:
        try:
            resolved_tgc_lift = float(tgc_lift)
        except ValueError:
            resolved_tgc_lift = 0.0

    effective_min_black = max(min_black, 30 if resolved_tgc_lift > 0 else min_black)
    min_black_L = rgb_to_oklab(np.array([[[effective_min_black]*3]], dtype=np.float32))[0,0,0]
    max_white_L = rgb_to_oklab(np.array([[[max_white]*3]], dtype=np.float32))[0,0,0]

    if p50 < 0.15: 
        target_front, target_back = 0.45, 0.18
    else: 
        target_front, target_back = 0.30, 0.15
        
    target_front = max(min_black_L, target_front * strength_val)
    target_back = max(min_black_L, target_back * strength_val)

    if assets.depth is not None:
        target_map = target_front * (1.0 - assets.depth) + target_back * assets.depth
        local_depth_blur = gaussian_filter(assets.depth, sigma=5.0)
        cavity_pocket = np.clip((local_depth_blur - assets.depth) * 3.0, 0.0, 1.0)
        darkness_pocket = 1.0 - smoothstep(0.0, 0.2, L)
        target_map = np.clip(target_map + (cavity_pocket * darkness_pocket * 0.15), 0.0, 0.8)
        focus_mask = 1.0 - assets.depth
    else:
        target_map = np.full((H, W), target_front, dtype=np.float32)
        focus_mask = np.ones((H, W), dtype=np.float32)

    shadow_mask = 1.0 - smoothstep(0.02, 0.55, L)
    L_float = np.ascontiguousarray(L, dtype=np.float32)
    
    try:
        L_base = cv2.ximgproc.guidedFilter(L_float, L_float, 16, 0.01)
    except (AttributeError, ImportError):
        L_base = gaussian_filter(L_float, sigma=8.0)
        
    L_detail = L - L_base 

    deficit = np.maximum(target_map - L_base, 0.0)
    L_base_new = L_base + (deficit * shadow_mask)

    Ix = cv2.Sobel(L_float, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(L_float, cv2.CV_32F, 0, 1, ksize=3)
    Jxx = gaussian_filter(Ix * Ix, sigma=2.0)
    Jyy = gaussian_filter(Iy * Iy, sigma=2.0)
    Jxy = gaussian_filter(Ix * Iy, sigma=2.0)
    
    tmp_inner = np.maximum((Jxx - Jyy)**2 + 4.0 * (Jxy**2), 0.0)
    tmp = np.sqrt(tmp_inner)
    lambda1 = 0.5 * (Jxx + Jyy + tmp)
    lambda2 = 0.5 * (Jxx + Jyy - tmp)
    structure = np.maximum(lambda1 - lambda2, 0.0)
    texture_mask = smoothstep(0.0001, 0.0020, structure)

    detail_multiplier = 1.0 + (detail_boost * shadow_mask * texture_mask * focus_mask)
    L_detail_new = L_detail * detail_multiplier

    L_new = L_base_new + L_detail_new

    if resolved_tgc_lift > 0.0:
        risk_zone = 1.0 - smoothstep(0.0, 0.25, L_new)
        L_new = L_new + (0.12 * resolved_tgc_lift * risk_zone * (1.0 - (texture_mask * 0.5)))
        L_new = np.clip(L_new, 0.0, 1.0)

    if prepress_hdr > 0.0:
        knee = 0.75 
        mask_high = L_new > knee
        if np.any(mask_high):
            L_high = L_new[mask_high]
            L_compressed = knee + (max_white_L - knee) * (1.0 - np.exp(-(L_high - knee) / (max_white_L - knee)))
            L_new[mask_high] = L_high * (1.0 - prepress_hdr) + L_compressed * prepress_hdr

    L_new = np.clip(L_new, min_black_L, max_white_L)

    chroma_mask = 1.0 - smoothstep(0.05, 0.25, L)
    a_smooth = gaussian_filter(a, sigma=3.0) 
    b_smooth = gaussian_filter(b, sigma=3.0)
    a_clean = a * (1.0 - chroma_mask) + a_smooth * chroma_mask
    b_clean = b * (1.0 - chroma_mask) + b_smooth * chroma_mask

    lift_ratio = (L_new + 0.05) / (L + 0.05)
    lift_ratio = np.clip(lift_ratio, 1.0, 1.3)
    a_new = a_clean * lift_ratio
    b_new = b_clean * lift_ratio

    local_a = gaussian_filter(a_new, sigma=5.0)
    local_b = gaussian_filter(b_new, sigma=5.0)
    deep_trench = 1.0 - smoothstep(0.0, 0.10, L_new)
    a_new = a_new * (1.0 - deep_trench) + local_a * deep_trench
    b_new = b_new * (1.0 - deep_trench) + local_b * deep_trench

    if cmyk_compression > 0.0:
        chroma = np.sqrt(a_new**2 + b_new**2)
        shadow_desat_mask = smoothstep(0.0, 0.25, L_new)
        reduction_factor = 1.0 - (cmyk_compression * (1.0 - shadow_desat_mask))
        a_new = a_new * reduction_factor
        b_new = b_new * reduction_factor

    oklab_new = np.stack([L_new, a_new, b_new], axis=-1)
    arr_lifted = oklab_to_rgb(oklab_new)

    dither_mask = shadow_mask * (1.0 - texture_mask) * (1.0 - smoothstep(0.20, 0.35, L_new))
    noise = (rng.random((H, W, 1)) - rng.random((H, W, 1))) * 1.5
    dithered_arr = arr_lifted + (noise * dither_mask[..., None])
    
    min_channel = np.min(dithered_arr, axis=-1, keepdims=True)
    needed_offset = np.maximum(float(effective_min_black) - min_channel, 0.0)
    dithered_arr = dithered_arr + needed_offset

    arr_final = np.clip(dithered_arr, 0.0, float(max_white))
    assets.image = arr_final

    if debug:
        assets.debug_data["oklab_L_original"] = L
        assets.debug_data["oklab_L_lifted"] = L_new
        assets.debug_data["texture_mask_tensor"] = texture_mask
        assets.debug_data["lift_shadow_mask"] = shadow_mask

    return assets


# ============================================================
# СОХРАНЕНИЕ И ОТЛАДКА
# ============================================================

def save_outputs(assets: ImageAssets, out_path: str, debug: bool) -> None:
    """Сохраняет результат. out_path - это уже готовый путь к финальному файлу."""
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    Image.fromarray(assets.image.astype(np.uint8)).save(out_path, format="PNG")
    print(f"  [SAVE] {out_path}", file=sys.stderr)
    
    if debug:
        # Берем имя файла БЕЗ расширения для дебаг-файлов
        final_stem = os.path.splitext(os.path.basename(out_path))[0]
        
        for k in ["oklab_L_original", "oklab_L_lifted", "lift_shadow_mask"]:
            if k in assets.debug_data:
                _save_map_png(assets.debug_data[k], os.path.join(out_dir, f"{final_stem}_dbg_{k}.png"), "DEBUG")
        
        # Симуляция просмотра в TGC
        fake_tcg_img = simulate_tcg_color_filter(assets.image)
        fake_tcg_path = os.path.join(out_dir, f"{final_stem}_fake_TCG_Color_Filter.png")
        Image.fromarray(fake_tcg_img.astype(np.uint8)).save(fake_tcg_path, format="PNG")
        print(f"  [DEBUG-TCG] Fake TCG filter preview saved: {fake_tcg_path}", file=sys.stderr)

        # Маска смерти (Red Zone of Death)
        tcg_lum = luminance_rgb(fake_tcg_img)
        crush_mask = (tcg_lum <= 5).astype(np.float32) 
        
        crushed_vis = assets.image.copy()
        crushed_vis[crush_mask > 0] = [255, 0, 0] 
        
        crush_path = os.path.join(out_dir, f"{final_stem}_dbg_TCG_CrushMask.png")
        Image.fromarray(crushed_vis.astype(np.uint8)).save(crush_path, format="PNG")
        print(f"  [DEBUG-TCG] 'Mask of Doom' saved: {crush_path}", file=sys.stderr)


# ============================================================
# CLI
# ============================================================

def main() -> None:
    p = argparse.ArgumentParser(description=f"Shadow-Lift Prepress Engine {VERSION}")
    
    p.add_argument("-i", "--input", required=False, default="", dest="input_path", help="File or directory")
    p.add_argument("-o", "--output", required=True, dest="output_dir", help="Output directory or exact file path")
    p.add_argument("-s", "--size", type=parse_size, default=None, help="Target size, e.g. 900x1500")
    
    p.add_argument("--generate-probe", action="store_true", help="Generate TGC calibration probes and exit")
    
    p.add_argument("-b", "--min-black", type=int, default=MIN_BLACK_DEFAULT, 
                   help=f"Min black (0-255). Default {MIN_BLACK_DEFAULT}")
    p.add_argument("-w", "--max-white", type=int, default=MAX_WHITE_DEFAULT, 
                   help=f"Max white (0-255). Default {MAX_WHITE_DEFAULT}")
    p.add_argument("--cmyk-compression", type=float, default=0.0,
                   help="CMYK gamut compression in shadows (0.0 = off, 1.0 = max safe)")
    
    p.add_argument("--lift-strength", type=str, default="auto", 
                   help="Global lifting strength: 'auto' or 0.0 - 1.0")
    p.add_argument("--depth-protection", type=float, default=0.5, 
                   help="How much depth protects the background (0.0 = off, 1.0 = max)")
    p.add_argument("--detail-boost", type=float, default=1.5, 
                   help="HDR geometry boost in shadows (1.0 = off, 2.0 = strong)")
    p.add_argument("--prepress-hdr", type=float, default=1.0,
                   help="Prepress HDR strength. 0.0 = off, 1.0 = max compression")
    p.add_argument("--tgc-lift", type=str, default="0.0",
                   help="TGC compensation strength: 'auto', 0.0, 1.0, 1.5, etc. (default: 0.0)")
    
    p.add_argument("--debug", action="store_true", help="Save debug maps, Fake TCG preview, and Crush Mask")
    p.add_argument("--no-shadow", action="store_true", help="Skip shadow map calculation")
    
    args = p.parse_args()

    if args.generate_probe:
        generate_tgc_probes(args.output_dir)
        sys.exit(0)

    if not args.input_path:
        sys.exit("Error: --input is required for image processing")

    if not 0 <= args.min_black <= 255:
        sys.exit("Error: --min-black must be between 0 and 255")

    is_single_file = os.path.isfile(args.input_path)
    
    if is_single_file:
        files = [args.input_path]
    else:
        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        files = sorted([
            os.path.join(args.input_path, f) for f in os.listdir(args.input_path)
            if f.lower().endswith(exts) 
            and "_dbg_" not in f.lower()
            and "_fake_TCG" not in f.lower()
            and "tgc_blackbox_probe" not in f.lower()
            and "tgc_highlight_probe" not in f.lower()
        ])

    # --- ЛОГИКА ОПРЕДЕЛЕНИЯ ПУТИ (Path Resolution) ---
    # Заранее вычисляем, хочет пользователь сохранить в файл или в папку.
    is_out_file = False
    if is_single_file:
        ext = os.path.splitext(args.output_dir)[1].lower()
        # Если расширение картинки или путь явно похож на файл
        if ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
            is_out_file = True
        elif not os.path.isdir(args.output_dir) and not args.output_dir.endswith(('/', '\\')):
            is_out_file = True

    print(f"Processing {len(files)} file(s) with Prepress Engine v{VERSION}...", file=sys.stderr)
    
    ok = 0
    for fpath in files:
        stem = os.path.splitext(os.path.basename(fpath))[0]
        
        # Формируем финальный путь
        if is_out_file:
            out_path = args.output_dir
            # Защита от перезаписи файла
            if os.path.abspath(out_path) == os.path.abspath(fpath):
                base, ext = os.path.splitext(out_path)
                out_path = f"{base}_lifted{ext}"
                print(f"  [SAFE] Output file matches input. Saving to: {out_path}", file=sys.stderr)
        else:
            out_dir = args.output_dir
            out_stem = stem
            # Защита от перезаписи в папке
            if os.path.abspath(out_dir) == os.path.abspath(os.path.dirname(fpath)):
                out_stem = f"{stem}_lifted"
                print(f"  [SAFE] Output dir matches input dir. Saving as: {out_stem}.png", file=sys.stderr)
            out_path = os.path.join(out_dir, f"{out_stem}.png")

        print(f"\n[{stem}]", file=sys.stderr)
        
        try:
            assets = extract_assets(
                input_path=fpath,
                target_size=args.size,
                want_depth=True,
                want_shadow=not args.no_shadow
            )
            
            assets = apply_lifting_engine(
                assets=assets,
                min_black=args.min_black,
                max_white=args.max_white,
                lift_strength=args.lift_strength,
                depth_protection=args.depth_protection,
                detail_boost=args.detail_boost, 
                prepress_hdr=args.prepress_hdr,
                cmyk_compression=args.cmyk_compression,
                tgc_lift=args.tgc_lift,
                debug=args.debug
            )
            
            # Передаем уже готовый out_path
            save_outputs(assets, out_path, args.debug)
            ok += 1
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {e}", file=sys.stderr)

    print(f"\nDone: {ok}/{len(files)}", file=sys.stderr)

if __name__ == "__main__":
    main()