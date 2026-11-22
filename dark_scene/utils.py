# dark_scene/utils.py
import numpy as np
import cv2

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def rgb_to_oklab(rgb_arr: np.ndarray) -> np.ndarray:
    """sRGB [0,255] → OKLAB (L,a,b)."""
    rgb = rgb_arr / 255.0
    rgb_lin = _srgb_to_linear(rgb)
    lms = np.dot(rgb_lin, np.array([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005]
    ]).T)
    lms_ = np.sign(lms) * (np.abs(lms) ** (1 / 3))
    oklab = np.dot(lms_, np.array([
        [0.2104542553,  0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050,  0.4505937099],
        [0.0259040371,  0.7827717662, -0.8086757660]
    ]).T)
    return oklab.astype(np.float32)
    
def fast_guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Edge-preserving Guided Filter с fallback."""
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        return cv2.ximgproc.guidedFilter(
            guide=guide.astype(np.float32), 
            src=src.astype(np.float32), 
            radius=r, eps=eps
        )
    
    # NumPy fallback
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