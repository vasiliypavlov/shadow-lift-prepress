#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shadow-Lift Depth/Shadow Prepress Engine
========================================
VERSION 1.7.4 — Intelligent Arcana & Smart-Preserve Edition

ОСОБЕННОСТИ РЕЛИЗА v1.7.4:
-------------------------
- Возврат к чистой математической базе v1.7.0 (удалены жесткие эвристики SGPM v1.7.3).
- Интеграция Dark Scene Analyzer v2.1 (--smart-preserve):
  1. Patch 1 (Ground Rescue): Отвязка target_map от depth_protection в зонах RECOVER.
  2. Patch 2 (Smoke Killer): Жесткий пол TGC и RGB-смещения нейтрализуются в зонах PRESERVE,
     сохраняя безупречные градиенты ночного неба без полошения и "серого дыма".
- Динамический размер окна детектора плато: win_size = max(5, int(W / 250) * 2 + 1).
- Безопасная обработка абсолютного черного (защита от деления на 0 при расчете lift_ratio).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

import argparse
import hashlib
import json
import os
import sys
import warnings

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

# ============================================================
# ИНТЕГРАЦИЯ DARK SCENE ANALYZER
# ============================================================
try:
    from dark_scene.models import DarkSceneMasks
    from dark_scene.spatial_analyzer import SpatialAnalyzer
    from dark_scene.regional_analyzer import RegionalAnalyzer
    from dark_scene.darkness_intent import DarknessIntentAnalyzer
    from dark_scene.mask_generator import MaskGenerator
    HAS_DARK_SCENE = True
except ImportError:
    HAS_DARK_SCENE = False

# ============================================================
# ВЕРСИЯ И КОНСТАНТЫ
# ============================================================
VERSION = "1.7.4"
MIN_BLACK_DEFAULT = 14
MAX_WHITE_DEFAULT = 245
MAPS_DIR_NAME = "_prepress_maps"
DEPTH_MODEL_TAG = "midas_small_70d6b9c8"
SHADOW_MODEL_TAG = "classical_v1"
MIDAS_WEIGHTS_FILENAME = "midas_v21_small_256.pt"
EXPECTED_MIDAS_WEIGHTS_SHA256 = (
    "70d6b9c891758c67f974a6097fb0c608c7ee67fb81ac3e5588847d5596d56fca"
)
SAFE_STRESS_OFFSET = 16

# ============================================================
# TGC SIMULATOR CURVE
# ============================================================
DEFAULT_CURVE_VALUES = {
    "black_clip":   0.09,
    "gamma":        1.25,
    "desat_thresh": 0.25,
    "desat_amount": 0.80,
    "scale_factor": 0.95,
}
DEFAULT_CURVE_BOUNDS = {
    "black_clip":   (0.00, 0.30),
    "gamma":        (0.80, 2.00),
    "desat_thresh": (0.05, 0.50),
    "desat_amount": (0.00, 1.00),
    "scale_factor": (0.70, 1.00),
}
DEFAULT_CURVE_FILENAME = "default_curve.json"

_ACTIVE_CURVE = dict(DEFAULT_CURVE_VALUES)
_ACTIVE_CURVE_SOURCE = "hardcoded defaults"

def _load_curve_json(path: str) -> None:
    global _ACTIVE_CURVE, _ACTIVE_CURVE_SOURCE
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"Error: cannot read curve file '{path}': {e}")

    curve = data.get("curve")
    if not isinstance(curve, dict):
        sys.exit(f"Error: curve file '{path}' has no 'curve' object.")

    new_curve = dict(DEFAULT_CURVE_VALUES)
    for k in DEFAULT_CURVE_VALUES:
        if k not in curve:
            print(f"  [CURVE] WARN: '{k}' missing in {path}, using default {DEFAULT_CURVE_VALUES[k]}", file=sys.stderr)
            continue
        try:
            v = float(curve[k])
        except (TypeError, ValueError):
            sys.exit(f"Error: curve['{k}'] must be a number, got {curve[k]!r}")
        lo, hi = DEFAULT_CURVE_BOUNDS[k]
        if not (lo <= v <= hi):
            sys.exit(f"Error: curve['{k}'] = {v} outside bounds [{lo}, {hi}]. Edit the file or remove it.")
        new_curve[k] = v

    _ACTIVE_CURVE = new_curve
    meta = data.get("curve_meta", {})
    calibrated = [k for k, m in meta.items() if isinstance(m, dict) and m.get("calibrated")]
    profile = data.get("profile", "?")
    _ACTIVE_CURVE_SOURCE = f"{os.path.basename(path)} (profile={profile})"

    print(f"  [CURVE] Loaded: {path}", file=sys.stderr)
    for k in DEFAULT_CURVE_VALUES:
        status = "calibrated" if k in calibrated else "from file"
        print(f"  [CURVE]   {k:14s} = {_ACTIVE_CURVE[k]:.4f}  ({status})", file=sys.stderr)


# ============================================================
# КОНТРАКТ ДАННЫХ
# ============================================================
@dataclass
class ImageAssets:
    image: np.ndarray
    lum: np.ndarray
    depth: Optional[np.ndarray] = None
    shadow: Optional[np.ndarray] = None
    dark_masks: Optional[Any] = None  # DarkSceneMasks
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
# УТИЛИТЫ
# ============================================================
def luminance_rgb(arr: np.ndarray) -> np.ndarray:
    return (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)

def stable_seed(input_path: str, base_seed: int = 42) -> int:
    digest = hashlib.sha256(os.path.abspath(input_path).encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "little") + int(base_seed)) % (2 ** 32)

def file_content_hash(path: str, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()[:16]

def get_maps_dir(source_path: str) -> str:
    if os.path.isdir(source_path):
        base = os.path.abspath(source_path)
    else:
        base = os.path.dirname(os.path.abspath(source_path))
    drive, tail = os.path.splitdrive(base)
    if not tail or tail == os.sep:
        base = os.path.expanduser("~")
    return os.path.join(base, MAPS_DIR_NAME)

def map_cache_path(source_path: str, stem: str, w: int, h: int, kind: str, content_hash: Optional[str] = None) -> str:
    model_tag = DEPTH_MODEL_TAG if kind == "depth" else SHADOW_MODEL_TAG
    if content_hash:
        name = f"{stem}_{model_tag}_{content_hash}_{w}x{h}_{kind}.png"
    else:
        name = f"{stem}_{model_tag}_{w}x{h}_{kind}.png"
    return os.path.join(get_maps_dir(source_path), name)

def parse_size(size_str: Optional[str]) -> Optional[Tuple[int, int]]:
    if not size_str: return None
    try:
        parts = size_str.lower().split("x") if "x" in size_str.lower() else size_str.split(",")
        w, h = parts
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
    return np.where(c_safe <= 0.0031308, c_safe * 12.92, 1.055 * (c_safe ** (1 / 2.4)) - 0.055)

def rgb_to_oklab(rgb_arr: np.ndarray) -> np.ndarray:
    rgb = rgb_arr / 255.0
    rgb_lin = _srgb_to_linear(rgb)
    lms = np.dot(rgb_lin, np.array([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]).T)
    lms_ = np.sign(lms) * (np.abs(lms) ** (1 / 3))
    oklab = np.dot(lms_, np.array([
        [0.2104542553,  0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050,  0.4505937099],
        [0.0259040371,  0.7827717662, -0.8086757660],
    ]).T)
    return oklab.astype(np.float32)

def oklab_to_rgb(oklab: np.ndarray) -> np.ndarray:
    lms_ = np.dot(oklab, np.array([
        [1.0,  0.3963377774,  0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480],
    ]).T)
    lms = lms_ ** 3
    rgb_lin = np.dot(lms, np.array([
        [ 4.0767416621, -3.3077115913,  0.2309699292],
        [-1.2684380046,  2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147,  1.7076147010],
    ]).T)
    rgb = _linear_to_srgb(rgb_lin)
    return np.clip(rgb * 255.0, 0, 255).astype(np.float32)

def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ============================================================
# GUIDED FILTER / DITHERING / ANCHOR
# ============================================================
def fast_guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    guide = np.ascontiguousarray(guide, dtype=np.float32)
    src = np.ascontiguousarray(src, dtype=np.float32)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        return cv2.ximgproc.guidedFilter(guide=guide, src=src, radius=r, eps=eps)
    
    mean_I = cv2.blur(guide, (r, r))
    mean_p = cv2.blur(src, (r, r))
    mean_Ip = cv2.blur(guide * src, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.blur(guide * guide, (r, r))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.blur(a, (r, r))
    mean_b = cv2.blur(b, (r, r))
    return mean_a * guide + mean_b

def cross_bilateral_align_depth(depth: np.ndarray, guide_gray: np.ndarray, radius: int = 12, eps: float = 0.01) -> np.ndarray:
    refined = fast_guided_filter(guide_gray, depth, r=radius, eps=eps)
    return np.clip(refined, 0.0, 1.0).astype(np.float32)

def _generate_blue_noise_field(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, 1.0, (h, w)).astype(np.float32)
    noise = (cv2.GaussianBlur(noise, (3, 3), 0.5) - cv2.GaussianBlur(noise, (7, 7), 1.2))
    nstd = float(np.std(noise))
    if nstd > 1e-6:
        noise = (noise / nstd) * 0.45
    else:
        noise = np.zeros_like(noise)
    return noise.astype(np.float32)

def apply_black_anchor(target_map: np.ndarray, L: np.ndarray, black_anchor: float) -> np.ndarray:
    if black_anchor <= 0.0: return target_map
    toe_mask = smoothstep(0.0, black_anchor, L)
    return target_map * toe_mask


# ============================================================
# AUTO-DEPTH-PROTECTION
# ============================================================
def auto_depth_protection(assets: ImageAssets) -> Tuple[float, str]:
    if assets.depth is None: return 0.0, "no depth map"
    far_mask = assets.depth > 0.7
    far_fraction = float(far_mask.mean())
    if far_fraction < 0.05: return 0.0, f"far background <5% ({far_fraction * 100:.1f}%)"
    far_lum = assets.lum[far_mask] / 255.0
    far_mean = float(far_lum.mean())
    far_std = float(far_lum.std())
    far_shadow = float(assets.shadow[far_mask].mean()) if assets.shadow is not None else 0.0

    if far_mean < 0.15 and far_std < 0.10 and far_shadow < 0.15:
        return 1.0, f"night sky (lum={far_mean:.2f}±{far_std:.2f})"
    if far_mean < 0.20: return 0.5, f"dark distant objects (lum={far_mean:.2f})"
    if far_mean < 0.50: return 0.6, f"haze/twilight (lum={far_mean:.2f})"
    return 0.2, f"daylight sky (lum={far_mean:.2f})"


# ============================================================
# СИМУЛЯТОР TGC
# ============================================================
def simulate_tcg_color_filter(arr: np.ndarray, mode: str = "safe") -> np.ndarray:
    norm = arr.astype(np.float32) / 255.0
    h, w, _ = norm.shape

    if mode == "safe":
        original_lum = (0.299 * norm[..., 0] + 0.587 * norm[..., 1] + 0.114 * norm[..., 2])
        tgc_color_matrix = np.array([[1.03, -0.02,  0.00], [-0.01, 1.02, -0.01], [ 0.00, -0.03, 1.03]])
        flat = norm.reshape(-1, 3)
        profiled = np.dot(flat, tgc_color_matrix.T).reshape(h, w, 3)
        profiled_lum = (0.299 * profiled[..., 0] + 0.587 * profiled[..., 1] + 0.114 * profiled[..., 2])
        with np.errstate(divide="ignore", invalid="ignore"):
            lum_ratio = np.where(profiled_lum > 1e-5, original_lum / profiled_lum, 1.0)
        profiled = np.clip(profiled * lum_ratio[..., None], 0.0, 1.0)
        return (profiled * 255.0).astype(np.float32)

    curve = _ACTIVE_CURVE
    norm_str = np.maximum(norm - float(curve["black_clip"]), 0.0) / (1.0 - float(curve["black_clip"]))
    norm_str = np.power(norm_str, float(curve["gamma"]))
    lum = (0.299 * norm_str[..., 0] + 0.587 * norm_str[..., 1] + 0.114 * norm_str[..., 2])[..., None]
    max_val = np.max(norm_str, axis=-1, keepdims=True)
    desat_mask = 1.0 - np.clip(max_val / float(curve["desat_thresh"]), 0.0, 1.0)
    norm_str = (norm_str * (1.0 - desat_mask * float(curve["desat_amount"])) + lum * (desat_mask * float(curve["desat_amount"])))
    norm_str = norm_str * float(curve["scale_factor"])
    return np.clip(norm_str * 255.0, 0, 255).astype(np.float32)


# ============================================================
# MODEL RESOLUTION / DEPTH ESTIMATOR
# ============================================================
def _resolve_models_dir() -> Optional[str]:
    local = Path(__file__).resolve().parent / "models"
    return str(local) if local.is_dir() else None

def _print_mismatch_banner(expected: str, actual: str, ckpt: str) -> None:
    print(f"\n  {'!'*62}\n  [WARN] MiDaS weights differ from pinned reference.\n         Expected: {expected}\n         Actual:   {actual}\n         File:     {ckpt}\n  {'!'*62}\n", file=sys.stderr)

def _verify_midas_weights(accept_mismatch: bool = False) -> Optional[str]:
    import torch as _torch
    ckpt = os.path.join(_torch.hub.get_dir(), "checkpoints", MIDAS_WEIGHTS_FILENAME)
    if not os.path.isfile(ckpt): return None
    h = hashlib.sha256()
    with open(ckpt, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    actual = h.hexdigest()
    actual8 = actual[:8]
    if actual == EXPECTED_MIDAS_WEIGHTS_SHA256: return actual8

    _print_mismatch_banner(EXPECTED_MIDAS_WEIGHTS_SHA256, actual, ckpt)
    if accept_mismatch: return actual8
    if not sys.stdin.isatty(): sys.exit(2)
    try:
        answer = input("  Continue with these weights? [y/N]: ").strip().lower()
    except EOFError: sys.exit(2)
    if answer in ("y", "yes"): return actual8
    sys.exit(2)

class DepthEstimator:
    _instance: Optional["DepthEstimator"] = None
    accept_models: bool = False

    def __new__(cls):
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._ready = False
            inst.torch = None
            inst.midas = None
            inst.transform = None
            inst.device = None
            cls._instance = inst
        return cls._instance

    @staticmethod
    def _torch_available() -> bool:
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def _detect_device(self):
        import torch
        if torch.cuda.is_available(): return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")

    def _ensure_loaded(self) -> None:
        if self._ready: return
        if not self._torch_available():
            self._ready = True
            return
        try:
            import torch
            self.torch = torch
            self.device = self._detect_device()
            print(f"  [DEPTH] Loading MiDaS_small on {self.device}", file=sys.stderr)
            local_dir = _resolve_models_dir()
            if local_dir:
                torch.hub.set_dir(local_dir)
                print(f"  [DEPTH] Using LOCAL models: {local_dir}", file=sys.stderr)
            
            pre_sha8 = _verify_midas_weights(accept_mismatch=DepthEstimator.accept_models)
            self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(self.device).eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = transforms.small_transform
        except Exception as e:
            print(f"  [DEPTH] Neural model failed ({e}); fallback to algorithmic", file=sys.stderr)
            self.torch = None
            self.midas = None
        self._ready = True

    def _algorithmic_depth(self, rgb_float: np.ndarray) -> np.ndarray:
        H, W = rgb_float.shape[:2]
        gray = (0.299 * rgb_float[..., 0] + 0.587 * rgb_float[..., 1] + 0.114 * rgb_float[..., 2]) / 255.0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark = cv2.erode(np.min(rgb_float / 255.0, axis=2), kernel)
        y_grad = np.linspace(0.2, 1.0, H, dtype=np.float32)[:, None]
        y_grad = np.repeat(y_grad, W, axis=1)
        pseudo = (1.0 - dark * 0.7) * 0.4 + y_grad * 0.4 + (1.0 - gray) * 0.2
        pmin, pmax = float(pseudo.min()), float(pseudo.max())
        pseudo = (pseudo - pmin) / (pmax - pmin + 1e-6)
        return np.clip(1.0 - pseudo, 0.0, 1.0).astype(np.float32)

    def infer(self, rgb_float: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        if self.torch is None or self.midas is None:
            return self._algorithmic_depth(rgb_float)
        H, W = rgb_float.shape[:2]
        img_u8 = np.clip(rgb_float, 0, 255).astype(np.uint8)
        batch = self.transform(img_u8).to(self.device)
        with self.torch.no_grad():
            pred = self.midas(batch)
            pred = self.torch.nn.functional.interpolate(pred.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False).squeeze()
        d = pred.cpu().numpy().astype(np.float32)
        dmin, dmax = float(d.min()), float(d.max())
        return ((d - dmin) / (dmax - dmin) if dmax - dmin > 1e-6 else np.zeros_like(d))

class ClassicalShadowDetector:
    def detect(self, image_rgb: np.ndarray) -> np.ndarray:
        L = luminance_rgb(image_rgb)
        L_local = gaussian_filter(L, sigma=max(image_rgb.shape[:2]) * 0.05)
        ratio = L / np.maximum(L_local, 1.0)
        darkness = np.clip(1.0 - ratio, 0.0, 1.0).astype(np.float32)
        return np.clip(gaussian_filter(darkness, sigma=2.0), 0.0, 1.0).astype(np.float32)

def _save_map_png(arr: np.ndarray, path: str, label: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8)).save(path, format="PNG")
    print(f"  [{label}] saved: {path}", file=sys.stderr)


# ============================================================
# ИЗВЛЕЧЕНИЕ АССЕТОВ
# ============================================================
def extract_assets(
    input_path: str,
    target_size: Optional[Tuple[int, int]] = None,
    want_depth: bool = True,
    want_shadow: bool = False,
    force_recompute_depth: bool = False,
    want_dark_analysis: bool = False,
) -> ImageAssets:
    img = Image.open(input_path).convert("RGB")
    if target_size: img = img.resize(target_size, Image.Resampling.LANCZOS)
    image = np.asarray(img, dtype=np.float32)
    H, W = image.shape[:2]
    content_hash = file_content_hash(input_path)

    assets = ImageAssets(image=image, lum=luminance_rgb(image), meta={
        "input_path": os.path.abspath(input_path), "content_hash": content_hash,
        "size_wh": (W, H), "seed": stable_seed(input_path), "maps_dir": get_maps_dir(input_path),
        "depth_cache": "n/a", "shadow_cache": "n/a",
    })

    stem = os.path.splitext(os.path.basename(input_path))[0]

    if want_depth:
        if not force_recompute_depth:
            cache_path = map_cache_path(input_path, stem, W, H, "depth", content_hash=content_hash)
            if os.path.isfile(cache_path):
                d_img = Image.open(cache_path).convert("L")
                if d_img.size == (W, H):
                    assets.depth = np.asarray(d_img, dtype=np.float32) / 255.0
                    assets.meta["depth_cache"] = f"hit ({content_hash[:8]})"
        
        if assets.depth is None:
            assets.depth = 1.0 - DepthEstimator().infer(image)
            _save_map_png(assets.depth, map_cache_path(input_path, stem, W, H, "depth", content_hash=content_hash), "DEPTH")
            assets.meta["depth_cache"] = f"miss ({content_hash[:8]})"
            
        assets.depth = cross_bilateral_align_depth(assets.depth, assets.lum / 255.0)

    if want_shadow:
        assets.shadow = ClassicalShadowDetector().detect(image)
        _save_map_png(assets.shadow, map_cache_path(input_path, stem, W, H, "shadow", content_hash=content_hash), "SHADOW")
        assets.meta["shadow_cache"] = f"computed ({content_hash[:8]})"

    if want_dark_analysis and HAS_DARK_SCENE:
        print("  [DARK-SCENE] Analyzing darkness intent...", file=sys.stderr)
        spatial_analyzer = SpatialAnalyzer()
        regional_analyzer = RegionalAnalyzer()
        intent_analyzer = DarknessIntentAnalyzer()
        mask_generator = MaskGenerator()

        spatial_res = spatial_analyzer.analyze(image)
        regional_res = regional_analyzer.analyze(spatial_res)
        regional_res.regions = intent_analyzer.analyze_all(regional_res.regions, image_height=H)
        
        assets.dark_masks = mask_generator.generate(spatial_res, regional_res)
        print(f"  [DARK-SCENE] Protected void: {np.mean(assets.dark_masks.preservation_mask)*100:.1f}%", file=sys.stderr)

    validate_assets(assets)
    return assets


# ============================================================
# ADVANCED CONTINUOUS AUTO-TUNER (v1.7.4 TAROT & LOW-KEY)
# ============================================================
@dataclass
class ContinuousSceneFeatures:
    q005: float
    q05: float
    q15: float
    q50: float
    crush_risk_pct: float
    deep_shadow_pct: float
    shadow_smoothness: float
    avg_shadow_chroma: float
    highlight_energy: float
    preserve_ratio: float
    recover_ratio: float
    radial_symmetry: float

@dataclass
class AutoPrepressProfile:
    archetype: str
    archetype_name: str
    tgc_mode: str
    tgc_strength: float
    soft_floor: float
    detail_boost: float
    depth_protection: float
    lift_strength: float
    black_anchor: float
    cmyk_compression: float
    explanation: str

def extract_scene_features(
    assets: ImageAssets, 
    oklab_L: np.ndarray, 
    oklab_a: np.ndarray, 
    oklab_b: np.ndarray
) -> ContinuousSceneFeatures:
    H, W = oklab_L.shape[:2]
    q005, q05, q15, q50 = np.percentile(oklab_L, [0.5, 5.0, 15.0, 50.0])
    
    crush_mask = oklab_L < 0.09
    shadow_mask = oklab_L < 0.18
    crush_risk_pct = float(np.mean(crush_mask) * 100.0)
    deep_shadow_pct = float(np.mean(shadow_mask) * 100.0)
    
    chroma = np.sqrt(oklab_a ** 2 + oklab_b ** 2)
    avg_shadow_chroma = float(np.mean(chroma[shadow_mask])) if np.any(shadow_mask) else 0.02

    gy, gx = np.gradient(oklab_L)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    shadow_grads = grad_mag[shadow_mask]
    if shadow_grads.size > 0:
        mean_shadow_grad = float(np.mean(shadow_grads))
        shadow_smoothness = float(np.clip(1.0 - (mean_shadow_grad / 0.025), 0.0, 1.0))
    else:
        shadow_smoothness = 0.5

    high_lum = oklab_L > 0.82
    highlight_energy = float(np.mean(high_lum) * 100.0)

    if assets.dark_masks is not None:
        preserve_ratio = float(np.mean(assets.dark_masks.preservation_mask > 0.4))
        recover_ratio = float(np.mean(assets.dark_masks.recovery_mask > 0.4))
    else:
        preserve_ratio = 0.0
        recover_ratio = 0.0

    y, x = np.ogrid[:H, :W]
    center_y, center_x = H / 2.0, W / 2.0
    r_dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    r_dist /= np.max(r_dist)
    core_mask = r_dist < 0.35
    periph_mask = r_dist > 0.70
    center_lum = float(np.mean(oklab_L[core_mask]))
    border_lum = float(np.mean(oklab_L[periph_mask]))
    radial_symmetry = float(np.clip(center_lum - border_lum, 0.0, 1.0))

    return ContinuousSceneFeatures(
        q005=float(q005),
        q05=float(q05),
        q15=float(q15),
        q50=float(q50),
        crush_risk_pct=crush_risk_pct,
        deep_shadow_pct=deep_shadow_pct,
        shadow_smoothness=shadow_smoothness,
        avg_shadow_chroma=avg_shadow_chroma,
        highlight_energy=highlight_energy,
        preserve_ratio=preserve_ratio,
        recover_ratio=recover_ratio,
        radial_symmetry=radial_symmetry
    )

def auto_tune_prepress_scene(
    assets: ImageAssets, 
    oklab_L: np.ndarray, 
    oklab_a: np.ndarray, 
    oklab_b: np.ndarray
) -> AutoPrepressProfile:
    f = extract_scene_features(assets, oklab_L, oklab_a, oklab_b)

    tgc_mode = "stress"
    risk_factor = np.clip(f.crush_risk_pct / 25.0, 0.0, 1.0)
    recover_factor = np.clip(f.recover_ratio * 2.0, 0.0, 1.0)
    preserve_damping = np.clip(f.preserve_ratio * 0.5, 0.0, 0.4)
    calculated_strength = 0.80 + (0.45 * risk_factor) + (0.15 * recover_factor) - preserve_damping
    tgc_strength = float(np.clip(calculated_strength, 0.70, 1.35))

    soft_floor = float(np.clip(1.2 + (2.3 * f.shadow_smoothness), 1.2, 3.5))

    base_anchor = max(0.012, f.q005 * 0.90)
    if f.preserve_ratio > 0.10:
        base_anchor = max(base_anchor, 0.025 + (f.preserve_ratio * 0.025))
    black_anchor = float(np.clip(base_anchor, 0.015, 0.055))

    detail_base = 1.35
    if f.avg_shadow_chroma > 0.05:
        detail_base += 0.25
    if f.shadow_smoothness > 0.70:
        detail_base -= 0.30
    detail_boost = float(np.clip(detail_base, 0.95, 1.75))

    dp_calc = 0.40 + (f.preserve_ratio * 0.40) + (0.20 if f.radial_symmetry < 0.1 else 0.0)
    depth_protection = float(np.clip(dp_calc, 0.20, 0.85))

    lift_strength = float(np.clip(0.60 + (0.40 * risk_factor), 0.50, 1.0))

    if f.avg_shadow_chroma > 0.065 and f.q15 < 0.22:
        arch_code = "CHROMA_METAL_LOWKEY"
        arch_name = "Оккультные металлы / Насыщенные тени"
    elif f.highlight_energy > 0.1 and f.crush_risk_pct > 15.0:
        arch_code = "FIRE_HOTSPOT_CONTRAST"
        arch_name = "Огонь/факелы в глубокой тьме"
    elif f.preserve_ratio > 0.25 or (f.q05 < 0.04 and f.shadow_smoothness > 0.6):
        arch_code = "COSMIC_INTENTIONAL_VOID"
        arch_name = "Глубокая пустота / Сохранный вакуум"
    elif f.radial_symmetry > 0.25:
        arch_code = "VIGNETTE_CENTRAL_SUBJECT"
        arch_name = "Центральная фигура / Виньетирование"
    else:
        arch_code = "CONTINUOUS_BALANCED_TAROT"
        arch_name = "Сбалансированная колода (Tarot Core)"

    explanation = (
        f"L-Quantiles=[q005:{f.q005:.3f}, q15:{f.q15:.3f}, q50:{f.q50:.3f}], "
        f"CrushRisk={f.crush_risk_pct:.1f}%, Smoothness={f.shadow_smoothness:.2f}, "
        f"PreserveRatio={f.preserve_ratio:.2f}, Anchor={black_anchor:.3f}"
    )

    return AutoPrepressProfile(
        archetype=arch_code,
        archetype_name=arch_name,
        tgc_mode=tgc_mode,
        tgc_strength=tgc_strength,
        soft_floor=soft_floor,
        detail_boost=detail_boost,
        depth_protection=depth_protection,
        lift_strength=lift_strength,
        black_anchor=black_anchor,
        cmyk_compression=0.0,
        explanation=explanation
    )


# ============================================================
# ДВИЖОК ЛИФТИНГА (v1.7.4 — Fixed & Cleaned)
# ============================================================
def apply_lifting_engine(
    assets: ImageAssets,
    min_black: int = MIN_BLACK_DEFAULT,
    max_white: int = MAX_WHITE_DEFAULT,
    lift_strength: Union[str, float] = "auto",
    depth_protection: Union[str, float] = "auto",
    detail_boost: float = 1.5,
    prepress_hdr: float = 1.0,
    cmyk_compression: float = 0.0,
    tgc_lift_mode: str = "off",
    tgc_lift_strength: float = 1.0,
    soft_floor: float = 0.0,
    black_anchor: float = 0.0,
    debug: bool = False,
    floor_diag: bool = False,
    auto_tune: bool = False,
    deband_strength: float = 0.0,
) -> ImageAssets:
    arr = assets.image.copy()
    H, W = arr.shape[:2]
    rng = np.random.default_rng(assets.meta["seed"])

    oklab = rgb_to_oklab(arr)
    L = oklab[..., 0]; a = oklab[..., 1]; b = oklab[..., 2]
    p15, p50 = np.percentile(L, [15, 50])

    if auto_tune:
        if soft_floor > 0.0:
            print(f"  [WARN] --auto переопределяет заданный вручную --soft-floor ({soft_floor}).", file=sys.stderr)
        profile = auto_tune_prepress_scene(assets, L, a, b)
        tgc_mode = profile.tgc_mode
        tgc_strength = profile.tgc_strength
        soft_floor = profile.soft_floor
        detail_boost = profile.detail_boost
        resolved_dp = profile.depth_protection
        strength_val = profile.lift_strength
        black_anchor = profile.black_anchor
        cmyk_compression = profile.cmyk_compression
        depth_scene_label = profile.archetype_name
        
        print(f"  [AUTO-TUNE] Архитип: {profile.archetype_name}", file=sys.stderr)
        print(f"  [AUTO-TUNE] Метрики: {profile.explanation}", file=sys.stderr)
        print(f"  [AUTO-TUNE] Синтез: TGC={tgc_mode} x{tgc_strength:.2f} | Floor={soft_floor:.1f} | Anchor={black_anchor:.3f} | Detail={detail_boost:.2f} | Deband={deband_strength:.1f}", file=sys.stderr)
    else:
        if isinstance(depth_protection, str) and depth_protection.lower() == "auto":
            resolved_dp, depth_scene_label = auto_depth_protection(assets)
        else:
            try: resolved_dp = float(depth_protection)
            except (ValueError, TypeError): resolved_dp = 0.5; depth_scene_label = "invalid value"
        resolved_dp = float(np.clip(resolved_dp, 0.0, 1.0))

        if str(lift_strength).lower() == "auto":
            strength_val = 1.0 if p15 < 0.1 else (0.7 if p15 < 0.2 else 0.4)
        else:
            try: strength_val = float(lift_strength)
            except (ValueError, TypeError): strength_val = 1.0
        strength_val = float(np.clip(strength_val, 0.0, 1.0))

        tgc_mode = str(tgc_lift_mode).strip().lower()
        if tgc_mode not in ("off", "safe", "stress"): tgc_mode = "off"
        tgc_strength = float(np.clip(tgc_lift_strength, 0.0, 2.0))
        if tgc_mode == "off": tgc_strength = 0.0

    effective_min_black = (min_black + SAFE_STRESS_OFFSET * tgc_strength if tgc_mode in ("safe", "stress") else float(min_black))

    print(f"\n{'='*66}\n Shadow-Lift Prepress Engine v{VERSION}\n{'='*66}", file=sys.stderr)
    print(f" [INPUT]  {assets.meta['input_path']}\n [SIZE]   {W}x{H} px", file=sys.stderr)
    print(f" [DARK-SCENE] {'ENABLED' if assets.dark_masks is not None else 'DISABLED'}", file=sys.stderr)
    print(f"   tgc-lift: {tgc_mode} {tgc_strength:.2f} | soft-floor: {soft_floor:.1f} | black-anchor: {black_anchor:.3f}", file=sys.stderr)
    print(f"   min-black: {min_black} (eff: {effective_min_black:.0f}) | detail-boost: {detail_boost:.2f} | deband: {deband_strength:.1f}", file=sys.stderr)
    print(f"{'='*66}\n", file=sys.stderr)

    L_float = L.astype(np.float32)
    fire_mask = (L > 0.85) & (b > 0.05)
    
    if np.any(fire_mask):
        fire_area = np.mean(fire_mask)
        adaptive_fire_thresh = 0.05 if p50 < 0.10 else 0.08
        if fire_area < adaptive_fire_thresh:
            L_for_base = L_float.copy()
            L_for_base[fire_mask] = float(np.percentile(L_float[~fire_mask], 90))
            L_base = fast_guided_filter(L_for_base, L_for_base, r=16, eps=0.01)
        else:
            L_base = fast_guided_filter(L_float, L_float, r=16, eps=0.01)
    else:
        L_base = fast_guided_filter(L_float, L_float, r=16, eps=0.01)

    min_black_L = rgb_to_oklab(np.array([[[float(min_black)] * 3]], dtype=np.float32))[0, 0, 0]
    effective_min_black_L = rgb_to_oklab(np.array([[[float(effective_min_black)] * 3]], dtype=np.float32))[0, 0, 0]
    max_white_L = rgb_to_oklab(np.array([[[max_white] * 3]], dtype=np.float32))[0, 0, 0]

    if soft_floor > 0.0 and tgc_mode in ("safe", "stress"):
        floor_plus_L = rgb_to_oklab(np.array([[[float(effective_min_black) + float(soft_floor)] * 3]], dtype=np.float32))[0, 0, 0]
        soft_floor_L = float(floor_plus_L - effective_min_black_L)
    else:
        soft_floor_L = 0.0

    if p50 < 0.15: target_front, target_back = 0.45, 0.18
    else: target_front, target_back = 0.30, 0.15
    target_front = max(min_black_L, target_front * strength_val)
    target_back = max(min_black_L, target_back * strength_val)

    if assets.depth is not None:
        far_bg_map = np.clip((assets.depth - 0.7) / 0.3, 0.0, 1.0)
        foreground_gate = 1.0 - far_bg_map
    else:
        far_bg_map = np.zeros((H, W), dtype=np.float32)
        foreground_gate = np.ones((H, W), dtype=np.float32)

    if assets.depth is not None:
        target_map = (target_front * (1.0 - assets.depth) + target_back * assets.depth)
        if assets.shadow is not None:
            shadow_weight = np.clip(assets.shadow * 1.5, 0.0, 1.0)
            target_map = (target_map * (1.0 - shadow_weight) + target_front * shadow_weight)
            is_sky = 1.0 - np.clip(assets.shadow * 2.5, 0.0, 1.0)
            protection = far_bg_map * resolved_dp * is_sky
        else:
            protection = far_bg_map * resolved_dp

        # --- Patch 1: Ground Rescue ---
        if assets.dark_masks is not None:
            protection = protection * (1.0 - assets.dark_masks.recovery_mask)

        target_map = (target_map * (1.0 - protection) + 0.05 * protection)
        cavity_pocket = np.clip((gaussian_filter(assets.depth, sigma=5.0) - assets.depth) * 3.0, 0.0, 1.0)
        target_map = np.clip(target_map + (cavity_pocket * (1.0 - smoothstep(0.0, 0.2, L)) * 0.15), 0.0, 0.8)
        focus_mask = 1.0 - assets.depth
    else:
        target_map = np.full((H, W), target_front, dtype=np.float32)
        focus_mask = np.ones((H, W), dtype=np.float32)

    if black_anchor > 0.0:
        target_map = apply_black_anchor(target_map, L, black_anchor)

    shadow_mask = 1.0 - smoothstep(0.02, 0.55, L)
    L_detail = L - L_base

    chroma = np.sqrt(a ** 2 + b ** 2)
    color_protection = smoothstep(0.02, 0.12, chroma)
    gradient_mag = np.sqrt(cv2.Sobel(L_float, cv2.CV_32F, 1, 0, ksize=3)**2 + cv2.Sobel(L_float, cv2.CV_32F, 0, 1, ksize=3)**2)
    flat_area_mask = 1.0 - smoothstep(0.001, 0.02, gradient_mag)
    deep_shadow_factor = 1.0 - smoothstep(0.04, 0.18, L)
    local_preservation = np.maximum(color_protection * (1.0 - deep_shadow_factor * 0.75), flat_area_mask * 0.7)

    # =========================================================================
    # ИДЕЯ Б: Непрерывный Oklab-шлюз яркости (Плавное гашение лифтинга)
    # =========================================================================
    L_blur_macro = gaussian_filter(L_float, sigma=15.0)
    sky_gate = (1.0 - smoothstep(0.05, 0.25, L_blur_macro)) * (1.0 - local_preservation)

    if assets.dark_masks is not None:
        preserve_weight = np.clip(assets.dark_masks.preservation_mask * sky_gate, 0.0, 1.0)
        recovery_weight = np.clip(assets.dark_masks.recovery_mask, 0.0, 1.0)
    else:
        preserve_weight = sky_gate * 0.5
        recovery_weight = np.ones((H, W), dtype=np.float32)

    deficit = np.maximum(target_map - L_base, 0.0) * (1.0 - local_preservation)
    deficit = deficit * (1.0 - preserve_weight * 0.85) * recovery_weight

    smooth_mask = 1.0 - smoothstep(0.005, 0.03, gradient_mag)
    deficit_smoothed = gaussian_filter(deficit, sigma=5.0)
    deficit_final = (deficit * (1.0 - smooth_mask) + deficit_smoothed * smooth_mask)

    L_base_new = L_base + (deficit_final * shadow_mask)

    Ix = cv2.Sobel(L_float, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(L_float, cv2.CV_32F, 0, 1, ksize=3)
    Jxx = gaussian_filter(Ix * Ix, sigma=2.0)
    Jyy = gaussian_filter(Iy * Iy, sigma=2.0)
    Jxy = gaussian_filter(Ix * Iy, sigma=2.0)  # <--- ИСПРАВЛЕНО: теперь Ix * Iy вместо Ix * Ix
    tmp = np.sqrt(np.maximum((Jxx - Jyy)**2 + 4.0 * (Jxy**2), 0.0))
    texture_mask = smoothstep(0.0001, 0.0020, np.maximum(0.5*(Jxx+Jyy+tmp) - 0.5*(Jxx+Jyy-tmp), 0.0))
    
    if assets.dark_masks is not None:
        texture_mask = texture_mask * (1.0 - np.clip(assets.dark_masks.preservation_mask, 0.0, 1.0))

    detail_multiplier = 1.0 + (detail_boost * shadow_mask * texture_mask * focus_mask)
    pos_detail = np.maximum(L_detail, 0.0)
    neg_detail = np.minimum(L_detail, 0.0)
    L_detail_new = (pos_detail * detail_multiplier) + (neg_detail * 1.0)
    
    L_new = L_base_new + L_detail_new

    # =========================================================================
    # TGC FLOORS (Smoke Killer Integration)
    # =========================================================================
    if tgc_mode == "safe":
        pixel_floor_L = effective_min_black_L
        if assets.dark_masks is not None:
            pixel_floor_L = min_black_L + (pixel_floor_L - min_black_L) * (1.0 - np.clip(assets.dark_masks.preservation_mask, 0.0, 1.0))

        if soft_floor_L > 0.0:
            _delta = np.maximum(pixel_floor_L - L_new, 0.0)
            L_new = L_new + (_delta * _delta) / (_delta + soft_floor_L)
        else:
            L_new = np.maximum(L_new, pixel_floor_L)
            
    elif tgc_mode == "stress":
        lum_blur_for_sky = gaussian_filter(L, sigma=15.0)
        sky_darkness_gate = 1.0 - smoothstep(0.08, 0.15, lum_blur_for_sky)
        is_sky = far_bg_map * sky_darkness_gate * (1.0 - texture_mask)
        pixel_floor_L = (min_black_L + (effective_min_black_L - min_black_L) * (foreground_gate * (1.0 - is_sky)))
        
        if assets.dark_masks is not None:
            pixel_floor_L = min_black_L + (pixel_floor_L - min_black_L) * (1.0 - np.clip(assets.dark_masks.preservation_mask, 0.0, 1.0))

        if soft_floor_L > 0.0:
            _delta = np.maximum(pixel_floor_L - L_new, 0.0)
            L_new = L_new + (_delta * _delta) / (_delta + soft_floor_L)
        else:
            L_new = np.maximum(L_new, pixel_floor_L)

    if prepress_hdr > 0.0:
        mask_high = L_new > 0.75
        if np.any(mask_high):
            L_high = L_new[mask_high]
            L_new[mask_high] = L_high * (1.0 - prepress_hdr) + (0.75 + (max_white_L - 0.75) * (1.0 - np.exp(-(L_high - 0.75) / (max_white_L - 0.75)))) * prepress_hdr

    L_new = np.clip(L_new, 0.0, 1.0)

    # Хроматическая коррекция (Oklab)
    chroma_mask = 1.0 - smoothstep(0.02, 0.35, L)
    a_new = a * (1.0 - chroma_mask) + gaussian_filter(a, sigma=7.0) * (chroma_mask * 0.8)
    b_new = b * (1.0 - chroma_mask) + gaussian_filter(b, sigma=7.0) * (chroma_mask * 0.8)
    
    safe_L_denom = np.maximum(L + 0.05, 1e-4)
    lift_ratio = np.clip((L_new + 0.05) / safe_L_denom, 1.0, 1.3)
    a_new = a_new * lift_ratio
    b_new = b_new * lift_ratio

    deep_trench = 1.0 - smoothstep(0.0, 0.10, L_new)
    a_new = a_new * (1.0 - deep_trench) + gaussian_filter(a_new, sigma=5.0) * deep_trench
    b_new = b_new * (1.0 - deep_trench) + gaussian_filter(b_new, sigma=5.0) * deep_trench

    if cmyk_compression > 0.0:
        reduction_factor = 1.0 - (cmyk_compression * (1.0 - smoothstep(0.0, 0.25, L_new)))
        a_new = a_new * reduction_factor
        b_new = b_new * reduction_factor

    if assets.dark_masks is not None:
        preserve_2d = np.clip(assets.dark_masks.preservation_mask, 0.0, 1.0)
        a_new = a_new * (1.0 - preserve_2d) + a * preserve_2d
        b_new = b_new * (1.0 - preserve_2d) + b * preserve_2d

    arr_lifted = oklab_to_rgb(np.stack([L_new, a_new, b_new], axis=-1))

    # =========================================================================
    # ДИЗЕРИНГ И ИТОГОВЫЙ ПОЛ В RGB
    # =========================================================================
    gradient_smoothness = 1.0 - smoothstep(0.0001, 0.0020, gradient_mag)
    deband_mask = gradient_smoothness * (1.0 - smoothstep(0.10, 0.25, L_float))
    
    win_size = max(5, int(W / 250) * 2 + 1)
    L_8bit = np.clip(L_float * 255.0, 0, 255).astype(np.uint8).astype(np.float32)
    mean_L = cv2.blur(L_8bit, (win_size, win_size))
    mean_L2 = cv2.blur(L_8bit**2, (win_size, win_size))
    variance = np.maximum(0, mean_L2 - mean_L**2)
    plateau_mask = (variance < 0.15) & (L_float < 0.18)
    deband_mask = np.maximum(deband_mask, plateau_mask.astype(np.float32))

    dither_mask = (shadow_mask * (1.0 - texture_mask) * (1.0 - smoothstep(0.20, 0.40, L_new)))
    if assets.dark_masks is not None:
        dither_mask = dither_mask * (1.0 - np.clip(assets.dark_masks.preservation_mask, 0.0, 1.0))
        
    noise_field = _generate_blue_noise_field(H, W, rng)
    base_dither_amp = 0.45
    sky_shaping = smoothstep(0.005, 0.12, L_float) * (1.0 - smoothstep(0.12, 0.30, L_float))
    deband_boost = np.maximum(deband_mask, sky_shaping)
    local_dither_amp = base_dither_amp * (1.0 + deband_boost * deband_strength * 0.95)
    
    dithered_arr = arr_lifted + (noise_field[..., None] * local_dither_amp[..., None] * dither_mask[..., None])

    min_channel = np.min(dithered_arr, axis=-1, keepdims=True)
    if tgc_mode == "safe":
        pixel_floor_rgb = float(effective_min_black)
    elif tgc_mode == "stress":
        lum_blur_for_sky_clamp = gaussian_filter(L, sigma=15.0)
        sky_darkness_clamp = 1.0 - smoothstep(0.08, 0.15, lum_blur_for_sky_clamp)
        is_sky_clamp = far_bg_map * sky_darkness_clamp * (1.0 - texture_mask)
        pixel_floor_rgb = (float(min_black) + (effective_min_black - float(min_black)) * (foreground_gate * (1.0 - is_sky_clamp)))[..., np.newaxis]
    else:
        pixel_floor_rgb = float(min_black)

    if soft_floor > 0.0 and tgc_mode in ("safe", "stress"):
        _d = np.maximum(pixel_floor_rgb - min_channel, 0.0)
        needed_offset = (_d * _d) / (_d + float(soft_floor))
    else:
        needed_offset = np.maximum(pixel_floor_rgb - min_channel, 0.0)

    if assets.dark_masks is not None:
        preserve_smooth = gaussian_filter(assets.dark_masks.preservation_mask, sigma=8.0)
        # ИСПРАВЛЕНО: добавлено [..., np.newaxis] для корректного весового бродкастинга
        needed_offset = needed_offset * (1.0 - np.clip(preserve_smooth * 0.75, 0.0, 0.9))[..., np.newaxis]

    dithered_arr = dithered_arr + needed_offset
    assets.image = np.clip(dithered_arr, 0.0, float(max_white))

    if debug:
        assets.debug_data["oklab_L_original"] = L
        assets.debug_data["oklab_L_lifted"] = L_new
        assets.debug_data["texture_mask"] = texture_mask
        assets.debug_data["target_map"] = target_map
        if assets.dark_masks is not None:
            assets.debug_data["dark_preserve_mask"] = assets.dark_masks.preservation_mask
            assets.debug_data["dark_recovery_mask"] = assets.dark_masks.recovery_mask

    assets.meta["tgc_mode"] = tgc_mode
    assets.meta["tgc_strength"] = tgc_strength
    return assets


# ============================================================
# СОХРАНЕНИЕ И ДИАГНОСТИКА
# ============================================================
def _save_prepress_filter(assets: ImageAssets, out_dir: str, final_stem: str) -> None:
    tgc_mode = assets.meta.get("tgc_mode", "off")
    prepress_sim = simulate_tcg_color_filter(assets.image, mode="stress")
    Image.fromarray(prepress_sim.astype(np.uint8)).save(os.path.join(out_dir, f"{final_stem}_prepress.png"), format="PNG")
    
    filter_crush_mask = (luminance_rgb(assets.image) > 10.0) & (luminance_rgb(prepress_sim) < 8.0)
    crush_count = int(np.sum(filter_crush_mask))
    crush_pct = 100.0 * crush_count / max(filter_crush_mask.size, 1)
    
    risk_vis = assets.image.astype(np.float32).copy()
    risk_vis[filter_crush_mask] = risk_vis[filter_crush_mask] * 0.5 + np.array([255.0, 0.0, 0.0]) * 0.5
    Image.fromarray(np.clip(risk_vis, 0, 255).astype(np.uint8)).save(os.path.join(out_dir, f"{final_stem}_prepress_risk.png"), format="PNG")

    severity = "OK" if crush_pct < 1.0 else ("MEDIUM" if crush_pct < 5.0 else "HIGH")
    print(f"  [PREPRESS] Filter crush: {crush_count:,} px ({crush_pct:.2f}%) — {severity}", file=sys.stderr)

def create_qa_sheet(original: np.ndarray, depth: Optional[np.ndarray], shadow: Optional[np.ndarray], output: np.ndarray, qa_path: str, target_h: int = 512) -> None:
    H, W = original.shape[:2]
    target_w = max(64, int(round(W * target_h / max(H, 1))))
    panels = [np.asarray(Image.fromarray(np.clip(original, 0, 255).astype(np.uint8)).resize((target_w, target_h), Image.Resampling.LANCZOS))]
    if depth is not None: panels.append(np.asarray(Image.fromarray(np.clip(depth * 255.0, 0, 255).astype(np.uint8)).resize((target_w, target_h), Image.Resampling.LANCZOS).convert("RGB")))
    if shadow is not None: panels.append(np.asarray(Image.fromarray(np.clip(shadow * 255.0, 0, 255).astype(np.uint8)).resize((target_w, target_h), Image.Resampling.LANCZOS).convert("RGB")))
    panels.append(np.asarray(Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)).resize((target_w, target_h), Image.Resampling.LANCZOS)))
    os.makedirs(os.path.dirname(qa_path), exist_ok=True)
    Image.fromarray(np.hstack(panels)).save(qa_path, format="JPEG", quality=88)

def save_outputs(assets: ImageAssets, out_path: str, debug: bool, prepress_filter: bool = False, qa_dir: Optional[str] = None, original_for_qa: Optional[np.ndarray] = None) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(assets.image.astype(np.uint8)).save(out_path, format="PNG")
    final_stem = os.path.splitext(os.path.basename(out_path))[0]

    if prepress_filter: _save_prepress_filter(assets, os.path.dirname(out_path), final_stem)
    if qa_dir: create_qa_sheet(original_for_qa if original_for_qa is not None else assets.image, assets.depth, assets.shadow, assets.image, os.path.join(qa_dir, f"{final_stem}_qa.jpg"))

    if debug:
        for k in ["oklab_L_original", "oklab_L_lifted", "texture_mask", "target_map", 
                  "dark_preserve_mask", "dark_recovery_mask"]:
            if k in assets.debug_data: _save_map_png(assets.debug_data[k], os.path.join(os.path.dirname(out_path), f"{final_stem}_dbg_{k}.png"), "DEBUG")
        print(f"  [DEBUG] Maps saved.", file=sys.stderr)


# ============================================================
# LOGGING / WORKER / CLI
# ============================================================
class _TeeStderr:
    def __init__(self, stream, log_path: str):
        self.stream = stream
        try: self.log_file = open(log_path, "a", encoding="utf-8", buffering=1)
        except OSError: self.log_file = None; return
        self.stream.write(f"\n{'='*72}\n Session: {datetime.now().isoformat(timespec='seconds')}\n{'='*72}\n")
    def write(self, data):
        self.stream.write(data)
        if self.log_file:
            try: self.log_file.write(data)
            except Exception: self.log_file = None
    def flush(self):
        self.stream.flush()
        if self.log_file:
            try: self.log_file.flush()
            except Exception: self.log_file = None
    def isatty(self): return self.stream.isatty()

def _worker_init(curve_dict: dict, curve_source: str, accept_models: bool) -> None:
    global _ACTIVE_CURVE, _ACTIVE_CURVE_SOURCE
    _ACTIVE_CURVE = dict(curve_dict); _ACTIVE_CURVE_SOURCE = curve_source
    DepthEstimator.accept_models = accept_models

def _worker_process(item: dict) -> Tuple[str, bool, str]:
    try:
        want_dark = item["smart_preserve"] or item["auto"]
        assets = extract_assets(
            input_path=item["input_path"], target_size=item["size"], want_depth=True,
            want_shadow=not item["no_shadow"], force_recompute_depth=item["recompute_depth"],
            want_dark_analysis=want_dark,
        )
        assets = apply_lifting_engine(
            assets=assets, min_black=item["min_black"], max_white=item["max_white"],
            lift_strength=item["lift_strength"], depth_protection=item["depth_protection"],
            detail_boost=item["detail_boost"], prepress_hdr=item["prepress_hdr"],
            cmyk_compression=item["cmyk_compression"], tgc_lift_mode=item["tgc_mode"],
            tgc_lift_strength=item["tgc_strength"], soft_floor=item["soft_floor"],
            black_anchor=item["black_anchor"], debug=item["debug"],
            floor_diag=False, auto_tune=item["auto"],
            deband_strength=item["deband_strength"],
        )
        original_for_qa = None
        if item["qa_dir"]:
            img = Image.open(item["input_path"]).convert("RGB")
            if item["size"]: img = img.resize(item["size"], Image.Resampling.LANCZOS)
            original_for_qa = np.asarray(img, dtype=np.float32)
        save_outputs(assets, item["out_path"], item["debug"], item["prepress_filter"], item["qa_dir"], original_for_qa)
        return item["input_path"], True, "OK"
    except Exception as e:
        import traceback; traceback.print_exc()
        return item["input_path"], False, str(e)

def main() -> None:
    p = argparse.ArgumentParser(
        description=f"Shadow-Lift Prepress Engine {VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        ⚠️ WARNING: FOR PREVIEW PURPOSES ONLY! Heuristic desaturation of shadows 
        (--cmyk-compression). DO NOT use for final files if the print house will be applying 
        its own ICC profile (e.g., GRACoL). Doing so will result in DOUBLE correction.

        ⚠️ ВНИМАНИЕ: ТОЛЬКО ДЛЯ ПРЕДПРОСМОТРА! Эвристическая десатурация теней 
        (--cmyk-compression). КАТЕГОРИЧЕСКИ НЕ ИСПОЛЬЗУЙТЕ для финальных файлов, 
        если типография будет применять свой ICC-профиль.
        """
    )
    p.add_argument("-i", "--input", required=False, default="", dest="input_path", help="File or directory")
    p.add_argument("-o", "--output", required=True, dest="output_dir", help="Output directory or exact file path")
    p.add_argument("-s", "--size", type=parse_size, default=None, help="Target size, e.g. 900x1500")
    p.add_argument("-b", "--min-black", type=int, default=MIN_BLACK_DEFAULT, help=f"Min black (0-255). Default {MIN_BLACK_DEFAULT}")
    p.add_argument("-w", "--max-white", type=int, default=MAX_WHITE_DEFAULT, help=f"Max white (0-255). Default {MAX_WHITE_DEFAULT}")
    p.add_argument("--cmyk-compression", type=float, default=0.0, help="⚠️ PREVIEW ONLY! Heuristic shadow desaturation (0.0 = off).")
    p.add_argument("--lift-strength", type=str, default="auto", help="Global lifting strength: 'auto' or 0.0-1.0")
    p.add_argument("--depth-protection", type=str, default="auto", help="Depth protection: 'auto' or 0.0-1.0")
    p.add_argument("--detail-boost", type=float, default=1.5, help="HDR geometry boost in shadows")
    p.add_argument("--prepress-hdr", type=float, default=1.0, help="Prepress HDR strength")
    p.add_argument("--black-anchor", type=float, default=0.0, help="OKLAB L threshold below which blacks are pinned (0 = off, typical 0.02-0.05)")
    p.add_argument("--auto", action="store_true", default=False, help="Autonomous Continuous Scene Classifier & Auto-Tuner (v1.7.4)")
    p.add_argument("--smart-preserve", action="store_true", default=False, help="Intelligent Dark Scene Analyzer. Protects space/voids and gradients.")
    p.add_argument("--tgc-lift", nargs="+", default=["off"], metavar="ARG", help="TGC printer compensation: MODE [STRENGTH]. MODE: off | safe | stress.")
    p.add_argument("--soft-floor", type=float, default=0.0, metavar="RGB", help="Soften the hard floor by up to N RGB units (0 = hard).")
    p.add_argument("--custom-curve", metavar="PATH", default=None, help="Load calibration JSON (from calibrate.py)")
    p.add_argument("--prepress-filter", action="store_true", help="Typographer-facing output (*_prepress.png + *_prepress_risk.png)")
    p.add_argument("--qa-dir", default=None, metavar="PATH", help="Side-by-side QA sheets (input | depth | shadow | output).")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers. Default 1 (serial). WARNING: each worker loads its own copy of MiDaS (~2GB RAM).")
    p.add_argument("--debug", action="store_true", help="Save technical debug maps (for developers).")
    p.add_argument("--no-shadow", action="store_true", help="Skip shadow map calculation")
    p.add_argument("--recompute-depth", action="store_true", help="Force recompute depth maps (ignore cache)")
    p.add_argument("--accept-models", action="store_true", help="If MiDaS weights SHA256 differs, proceed without prompting.")
    p.add_argument("--log", action="store_true", default=False, help="Write full run log to shadow_lift_prepress.log next to the script.")
    p.add_argument("--deband-strength", type=float, default=None, metavar="VAL",
                   help="Gradient debanding strength (0.0-2.0). Default 1.0 with --auto, 0.0 otherwise.")

    args = p.parse_args()
    DepthEstimator.accept_models = args.accept_models

    if args.log:
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shadow_lift_prepress.log")
        sys.stderr = _TeeStderr(sys.stderr, log_path)

    if args.deband_strength is None:
        args.deband_strength = 1.0 if args.auto else 0.0

    if args.custom_curve:
        if not Path(args.custom_curve).is_file(): sys.exit(f"Error: file not found: {args.custom_curve}")
        _load_curve_json(args.custom_curve)
    else:
        _default_json = Path(__file__).resolve().parent / DEFAULT_CURVE_FILENAME
        if _default_json.is_file(): _load_curve_json(str(_default_json))

    if not args.input_path: sys.exit("Error: --input is required")
    if not 0 <= args.min_black <= 255: sys.exit("Error: --min-black must be between 0 and 255")
    if args.soft_floor < 0.0 or args.soft_floor > 20.0: sys.exit("Error: --soft-floor must be between 0 and 20")

    dp_val = args.depth_protection.strip().lower()
    if dp_val != "auto":
        try:
            if not 0.0 <= float(dp_val) <= 1.0: sys.exit("Error: --depth-protection must be 'auto' or between 0.0 and 1.0")
        except ValueError: sys.exit("Error: --depth-protection must be 'auto' or a number")

    tgc_args = args.tgc_lift if isinstance(args.tgc_lift, list) else [args.tgc_lift]
    tgc_mode_raw = str(tgc_args[0]).strip().lower()
    if tgc_mode_raw not in ("off", "safe", "stress"): sys.exit("Error: --tgc-lift mode must be off/safe/stress")

    tgc_strength = 1.0
    if len(tgc_args) > 1:
        try: tgc_strength = float(tgc_args[1])
        except ValueError: sys.exit("Error: --tgc-lift STRENGTH must be a number")
        if not 0.0 <= tgc_strength <= 2.0: sys.exit("Error: --tgc-lift STRENGTH must be 0.0-2.0")

    if args.soft_floor > 0.0 and tgc_mode_raw == "off":
        print("  [WARN] --soft-floor has no effect in --tgc-lift off mode. Ignoring.", file=sys.stderr)
        args.soft_floor = 0.0

    is_single_file = os.path.isfile(args.input_path)
    if is_single_file:
        files = [args.input_path]
    else:
        files = sorted([os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")) and "_dbg_" not in f.lower() and "_prepress" not in f.lower()])

    if not files: sys.exit("[WARN] No matching images found to process.")

    is_out_file = False
    if is_single_file:
        ext = os.path.splitext(args.output_dir)[1].lower()
        if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff") or (not os.path.isdir(args.output_dir) and not args.output_dir.endswith(("/", "\\"))):
            is_out_file = True

    qa_dir = args.qa_dir
    if qa_dir: os.makedirs(qa_dir, exist_ok=True)

    print(f"Processing {len(files)} file(s) with Prepress Engine v{VERSION}...", file=sys.stderr)

    work_items = []
    for fpath in files:
        stem = os.path.splitext(os.path.basename(fpath))[0]
        if is_out_file:
            out_path = args.output_dir
            if os.path.abspath(out_path) == os.path.abspath(fpath):
                out_path = f"{os.path.splitext(out_path)[0]}_lifted{os.path.splitext(out_path)[1]}"
        else:
            out_stem = f"{stem}_lifted" if os.path.abspath(args.output_dir) == os.path.abspath(os.path.dirname(fpath)) else stem
            if args.cmyk_compression > 0.0:
                out_stem = f"{out_stem}_CMYK_PREVIEW"
                print(f"  [WARN] CMYK PREVIEW SAVE: {out_stem}.png DO NOT print as master!", file=sys.stderr)
            out_path = os.path.join(args.output_dir, f"{out_stem}.png")

        work_items.append({
            "input_path": fpath, "out_path": out_path, "size": args.size, "qa_dir": qa_dir,
            "min_black": args.min_black, "max_white": args.max_white, "lift_strength": args.lift_strength,
            "depth_protection": args.depth_protection, "detail_boost": args.detail_boost, "prepress_hdr": args.prepress_hdr,
            "cmyk_compression": args.cmyk_compression, "tgc_mode": tgc_mode_raw, "tgc_strength": tgc_strength,
            "soft_floor": args.soft_floor, "black_anchor": args.black_anchor, "debug": args.debug,
            "prepress_filter": args.prepress_filter, "no_shadow": args.no_shadow, "recompute_depth": args.recompute_depth,
            "auto": args.auto, "smart_preserve": args.smart_preserve,
            "deband_strength": args.deband_strength,
        })

    ok = 0
    if args.workers <= 1:
        for item in work_items:
            print(f"\n[{os.path.splitext(os.path.basename(item['input_path']))[0]}]", file=sys.stderr)
            _, success, msg = _worker_process(item)
            if success: ok += 1
            else: print(f"  FAILED: {msg}", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init, initargs=(_ACTIVE_CURVE, _ACTIVE_CURVE_SOURCE, args.accept_models)) as executor:
            for fut in as_completed({executor.submit(_worker_process, item): item["input_path"] for item in work_items}):
                path, success, msg = fut.result()
                if success: ok += 1; print(f"[{os.path.splitext(os.path.basename(path))[0]}] OK", file=sys.stderr)
                else: print(f"[{os.path.splitext(os.path.basename(path))[0]}] FAILED: {msg}", file=sys.stderr)

    print(f"\nDone: {ok}/{len(files)}", file=sys.stderr)

if __name__ == "__main__":
    main()