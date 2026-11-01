"""
Shadow-Lift — Standalone Calibration Toolkit
=============================================

Runs WITHOUT torch, cv2, scipy. Only numpy + Pillow.

Usage:
    # 1. Generate calibration target (900x1500, 41 grayscale steps + ref patches)
    python calibrate.py --generate-probe target.png

    # 2. Print on target paper, scan, crop to exact 900x1500, save as PNG

    # 3. Fit curve against your scan:
    python calibrate.py target.png my_scan.png
    # → writes my_scan.json next to scan

    # 4. Use in main script:
    python shadow_lift_depth_shadow_prepress.py -i cards/ -o out/ \
        --tgc-lift stress --prepress-filter --custom-curve my_scan.json

Other modes:
    --generate-probe-legacy DIR   → old tgc_blackbox_probe.png / tgc_highlight_probe.png
                                    (visual inspection, NOT used by calibration)
    --dump-default OUT.json       → default_curve.json (for main script auto-load)

Calibration method:
    For each 0..40 step, compares scan vs black reference patch (normalized).
    Finds first input RGB where scan rises above noise floor.
    Only `black_clip` is calibrated; gamma/desat/scale remain from reverse
    engineering (scanner cannot separate them from printer response).
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw


# --- Must match main script defaults ---
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

# --- Calibration probe geometry ---
CALIB_W, CALIB_H = 900, 1500
CALIB_STEPS = list(range(0, 41))        # 41 steps: 0..40
CALIB_REF_HEIGHT = 150                  # y=0..149: white (left) + black (right)
CALIB_GRID_TOP = 180                    # grid starts here
CALIB_COLS = 4
CALIB_ROW_H = 120                       # (1500-180)/11 = 120 exactly
CALIB_ROWS = 11

# Detection thresholds
DETECT_MULTIPLIER = 3.0                 # first step where norm > 3*noise_norm
DETECT_MIN = 0.005                      # floor for noise_norm (avoid tiny denominators)


def _load_default_curve() -> dict:
    p = Path(__file__).resolve().parent / DEFAULT_CURVE_FILENAME
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            curve = data.get("curve", {})
            merged = dict(DEFAULT_CURVE_VALUES)
            merged.update({k: v for k, v in curve.items() if k in DEFAULT_CURVE_VALUES})
            return merged
        except Exception as e:
            print(f"[WARN] Cannot parse {p}: {e}. Using hardcoded defaults.",
                  file=sys.stderr)
    return dict(DEFAULT_CURVE_VALUES)


def _curve_meta_default() -> dict:
    meta = {}
    for k in DEFAULT_CURVE_VALUES:
        lo, hi = DEFAULT_CURVE_BOUNDS[k]
        meta[k] = {
            "calibrated": False,
            "locked": True,
            "bounds": [lo, hi],
            "source": "reverse_engineered",
        }
    return meta


def _write_json(path: str, data: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[WRITE] {p}", file=sys.stderr)


# ============================================================
# Calibration target (new)
# ============================================================

def generate_probe_calibration(out_path: str) -> None:
    """
    Generate a 900x1500 calibration target:
      y=0..149:    white patch (x=0..449)  |  black patch (x=450..899)
      y=180..1499: 41 grayscale steps (0..40) in 4 cols x 11 rows
      corners:     30x30 white alignment marks

    Step patches are sampled from their interior (margin on all sides),
    so the corner markers do not affect measurement.
    """
    W, H = CALIB_W, CALIB_H
    img = np.full((H, W, 3), 128, dtype=np.uint8)

    # Reference zone: left white (245), right black (0)
    img[0:CALIB_REF_HEIGHT, 0:450] = 245
    img[0:CALIB_REF_HEIGHT, 450:W] = 0

    # Grayscale steps
    col_w = W // CALIB_COLS
    for i, val in enumerate(CALIB_STEPS):
        r, c = i // CALIB_COLS, i % CALIB_COLS
        x0 = c * col_w
        y0 = CALIB_GRID_TOP + r * CALIB_ROW_H
        x1 = min(x0 + col_w, W)
        y1 = min(y0 + CALIB_ROW_H, H)
        img[y0:y1, x0:x1] = val

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)

    # Corner alignment markers
    m = 30
    draw.rectangle([0, 0, m - 1, m - 1], fill=255)
    draw.rectangle([W - m, 0, W - 1, m - 1], fill=255)
    draw.rectangle([0, H - m, m - 1, H - 1], fill=255)
    draw.rectangle([W - m, H - m, W - 1, H - 1], fill=255)

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pil.save(p)
    print(f"[PROBE] Calibration target: {p}", file=sys.stderr)
    print(
        f"[PROBE]   {W}x{H}px, {len(CALIB_STEPS)} steps (0..{CALIB_STEPS[-1]}), "
        f"ref patches top",
        file=sys.stderr,
    )
    print(
        f"[PROBE]   Print → scan (>=300 DPI, NO auto-correction) → "
        f"crop to {W}x{H} → save as PNG",
        file=sys.stderr,
    )


# ============================================================
# Legacy probes (moved from main script)
# ============================================================

def generate_probe_legacy(out_dir: str) -> None:
    """
    Port of the original generate_tgc_probes() from v1.4.3.
    Produces tgc_blackbox_probe_legacy.png and tgc_highlight_probe_legacy.png.
    NOT used by the calibration fitting — for visual inspection only.
    """
    os.makedirs(out_dir, exist_ok=True)
    W, H = 900, 1500

    print("  [PROBE-LEGACY] Generating tgc_blackbox_probe_legacy.png...", file=sys.stderr)
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
        canvas[y0:y0 + cell_h, x0:x0 + cell_w] = [val, val, val]

    colors = [
        (14, 14, 14), (20, 10, 10), (10, 20, 10), (10, 10, 20),
        (14, 20, 25), (25, 14, 10), (20, 14, 25), (25, 25, 10),
        (8, 8, 8),    (16, 8, 8),   (8, 16, 8),   (8, 8, 16),
        (30, 15, 10), (10, 30, 20), (15, 10, 30), (30, 30, 30),
    ]
    cell_h_c = 350 // 4
    for i, col in enumerate(colors):
        r, c = i // 4, i % 4
        x0, y0 = c * cell_w, 750 + r * cell_h_c
        canvas[y0:y0 + cell_h_c, x0:x0 + cell_w] = col

    canvas[1100:1500, :450] = 10
    canvas[1100:1500, 450:] = 14
    img_b = Image.fromarray(canvas)
    draw_b = ImageDraw.Draw(img_b)
    for idx, delta in enumerate([2, 5, 10, 15]):
        y_pos = [1150, 1250, 1350, 1430][idx]
        for w_line in range(1, 5):
            draw_b.line([(20 + w_line * 30, y_pos), (20 + w_line * 30, y_pos + 60)],
                        fill=10 + delta, width=w_line)
            draw_b.line([(470 + w_line * 30, y_pos), (470 + w_line * 30, y_pos + 60)],
                        fill=14 + delta, width=w_line)
    img_b.save(os.path.join(out_dir, "tgc_blackbox_probe_legacy.png"))

    print("  [PROBE-LEGACY] Generating tgc_highlight_probe_legacy.png...", file=sys.stderr)
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
        canvas_w[y0:y0 + h_cell_h, x0:x0 + cell_w] = [val, val, val]

    pastel_colors = [
        (255, 240, 240), (240, 255, 240), (240, 240, 255), (255, 255, 240),
        (250, 230, 220), (220, 250, 230), (230, 220, 250), (250, 250, 220),
        (245, 220, 200), (200, 245, 220), (220, 200, 245), (245, 245, 200),
        (240, 210, 190), (190, 240, 210), (210, 190, 240), (240, 240, 190),
    ]
    for i, col in enumerate(pastel_colors):
        r, c = i // 4, i % 4
        x0, y0 = c * cell_w, 750 + r * (350 // 4)
        canvas_w[y0:y0 + (350 // 4), x0:x0 + cell_w] = col

    canvas_w[1100:1500, :] = 245
    img_w = Image.fromarray(canvas_w)
    draw_w = ImageDraw.Draw(img_w)
    for idx, val in enumerate([220, 200, 150, 100]):
        y_pos = [1150, 1250, 1350, 1430][idx]
        for w_line in range(1, 5):
            draw_w.line([(50 + w_line * 60, y_pos), (50 + w_line * 60, y_pos + 60)],
                        fill=val, width=w_line)
            draw_w.line([(500 + w_line * 60, y_pos), (500 + w_line * 60, y_pos + 60)],
                        fill=(val, val // 2, 50), width=w_line)
    img_w.save(os.path.join(out_dir, "tgc_highlight_probe_legacy.png"))
    print(f"  [PROBE-LEGACY] Done: {out_dir}", file=sys.stderr)


# ============================================================
# Dump default curve
# ============================================================

def dump_default_curve(out_path: str) -> None:
    curve = _load_default_curve()
    data = {
        "version": "1.0",
        "profile": "default",
        "calibrated_at": datetime.now().isoformat(timespec="seconds"),
        "curve": curve,
        "curve_meta": _curve_meta_default(),
        "calibration": {
            "method": "none",
            "note": "Defaults from reverse-engineering of TGC web preview. "
                    "Not calibrated against physical print.",
        },
        "warning": (
            "Only black_clip is calibratable via calibrate.py. "
            "gamma/desat_thresh/desat_amount/scale_factor remain from "
            "reverse engineering and are NOT validated for your printer."
        ),
    }
    _write_json(out_path, data)
    print(
        "[DUMP] All 5 params are `locked: true` by default. To experiment, "
        "set `locked: false` and edit `curve`.",
        file=sys.stderr,
    )


# ============================================================
# Calibration (fit black_clip against physical scan)
# ============================================================

def _sample_patch(arr: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Mean-luminance of interior region, avoiding borders."""
    h = y1 - y0
    w = x1 - x0
    my = max(1, h // 3)
    mx = max(1, w // 4)
    region = arr[y0 + my:y1 - my, x0 + mx:x1 - mx]
    lum = 0.299 * region[..., 0] + 0.587 * region[..., 1] + 0.114 * region[..., 2]
    return lum.astype(np.float32)


def calibrate(target_path: str, scan_path: str) -> int:
    print(f"[CALIBRATE] Target: {target_path}", file=sys.stderr)
    print(f"[CALIBRATE] Scan:   {scan_path}", file=sys.stderr)

    try:
        tgt = Image.open(target_path).convert("RGB")
    except Exception as e:
        print(f"Error: cannot open target '{target_path}': {e}", file=sys.stderr)
        return 1
    try:
        scan = Image.open(scan_path).convert("RGB")
    except Exception as e:
        print(f"Error: cannot open scan '{scan_path}': {e}", file=sys.stderr)
        return 1

    if tgt.size != (CALIB_W, CALIB_H):
        print(
            f"Error: target must be {CALIB_W}x{CALIB_H}, got {tgt.size[0]}x{tgt.size[1]}.\n"
            f"       Regenerate with: python calibrate.py --generate-probe target.png",
            file=sys.stderr,
        )
        return 1
    if scan.size != (CALIB_W, CALIB_H):
        print(
            f"Error: scan must be {CALIB_W}x{CALIB_H}, got {scan.size[0]}x{scan.size[1]}.\n"
            f"       Crop the scan to exact target bounds before running calibration.",
            file=sys.stderr,
        )
        return 1

    tgt_arr = np.asarray(tgt, dtype=np.uint8)
    scan_arr = np.asarray(scan, dtype=np.uint8)

    # Sanity: verify target layout matches expectations
    white_target = float(_sample_patch(tgt_arr, 0, 0, 450, CALIB_REF_HEIGHT).mean())
    black_target = float(_sample_patch(tgt_arr, 450, 0, CALIB_W, CALIB_REF_HEIGHT).mean())
    if white_target < 200 or black_target > 55:
        print(
            f"Error: target does not look like the calibration probe "
            f"(white={white_target:.0f}, black={black_target:.0f}).\n"
            f"       Regenerate with --generate-probe.",
            file=sys.stderr,
        )
        return 1

    # Reference patches from the SCAN
    white_scan = _sample_patch(scan_arr, 0, 0, 450, CALIB_REF_HEIGHT)
    black_scan = _sample_patch(scan_arr, 450, 0, CALIB_W, CALIB_REF_HEIGHT)

    white_mean = float(white_scan.mean())
    black_mean = float(black_scan.mean())
    black_std = float(black_scan.std())

    print(
        f"[CALIBRATE] Reference patches in scan: "
        f"white={white_mean:.1f}  black={black_mean:.1f}±{black_std:.1f}",
        file=sys.stderr,
    )

    if white_mean - black_mean < 60:
        print(
            f"Error: insufficient contrast in scan "
            f"(white-black = {white_mean - black_mean:.1f}).\n"
            f"       Check scanner settings: disable auto-levels, use color mode, "
            f"do not crop into patches.",
            file=sys.stderr,
        )
        return 1

    if black_std > 5.0:
        print(
            f"[CALIBRATE] WARN: black patch noise is high (std={black_std:.1f}). "
            f"Result may be imprecise.",
            file=sys.stderr,
        )

    # Sample each step from the scan
    col_w = CALIB_W // CALIB_COLS
    measurements = []  # list of (target_value, scan_mean_lum, normalized)
    span = max(white_mean - black_mean, 1.0)
    for i, val in enumerate(CALIB_STEPS):
        r, c = i // CALIB_COLS, i % CALIB_COLS
        x0 = c * col_w
        y0 = CALIB_GRID_TOP + r * CALIB_ROW_H
        x1 = min(x0 + col_w, CALIB_W)
        y1 = min(y0 + CALIB_ROW_H, CALIB_H)
        patch = _sample_patch(scan_arr, x0, y0, x1, y1)
        mean = float(patch.mean())
        normalized = (mean - black_mean) / span
        measurements.append((val, mean, normalized))

    # Noise floor in normalized units
    noise_norm = max(black_std / span, DETECT_MIN)
    threshold = max(DETECT_MULTIPLIER * noise_norm, DETECT_MIN)
    print(
        f"[CALIBRATE] Noise floor: {black_std:.2f} raw = {noise_norm:.4f} normalized "
        f"(threshold = {threshold:.4f})",
        file=sys.stderr,
    )

    # Print a compact ladder so the user can eyeball the fit
    print("[CALIBRATE] Step ladder (target → scan mean → normalized):", file=sys.stderr)
    for val, mean, norm in measurements:
        marker = "  <-- first above threshold" if norm > threshold and val > 0 and \
                 (val == 0 or measurements[val - 1][2] <= threshold) else ""
        print(
            f"[CALIBRATE]   {val:3d} → {mean:6.1f} → {norm:+.4f}{marker}",
            file=sys.stderr,
        )

    first_above = None
    for val, _, norm in measurements:
        if norm > threshold:
            first_above = val
            break

    if first_above is None:
        print(
            f"[CALIBRATE] WARN: no step rose above threshold in 0..{CALIB_STEPS[-1]}.\n"
            f"            Either the printer crushes everything, or the scan is too noisy.\n"
            f"            Falling back to default black_clip={DEFAULT_CURVE_VALUES['black_clip']:.4f}.",
            file=sys.stderr,
        )
        first_above = int(round(DEFAULT_CURVE_VALUES["black_clip"] * 255))

    black_clip = first_above / 255.0
    lo, hi = DEFAULT_CURVE_BOUNDS["black_clip"]
    if not (lo <= black_clip <= hi):
        print(
            f"[CALIBRATE] WARN: fitted black_clip={black_clip:.4f} is outside "
            f"bounds [{lo}, {hi}]. Clamping.",
            file=sys.stderr,
        )
        black_clip = float(np.clip(black_clip, lo, hi))

    # Confidence heuristic
    if black_std < 1.5 and first_above <= CALIB_STEPS[-1] - 5:
        confidence = "high"
    elif black_std < 4.0:
        confidence = "medium"
    else:
        confidence = "low"

    default_clip = DEFAULT_CURVE_VALUES["black_clip"]
    default_rgb = int(round(default_clip * 255))
    delta_rgb = int(round(black_clip * 255)) - default_rgb

    print("", file=sys.stderr)
    print("[CALIBRATE] RESULT:", file=sys.stderr)
    print(
        f"[CALIBRATE]   black_clip ≈ {first_above} / 255 = {black_clip:.4f}  "
        f"(confidence: {confidence})",
        file=sys.stderr,
    )
    print(
        f"[CALIBRATE]   Default was {default_clip:.4f} ({default_rgb} RGB).",
        file=sys.stderr,
    )
    if delta_rgb > 0:
        print(
            f"[CALIBRATE]   Your printer+paper crushes ~{delta_rgb} RGB LATER than model.",
            file=sys.stderr,
        )
    elif delta_rgb < 0:
        print(
            f"[CALIBRATE]   Your printer+paper crushes ~{-delta_rgb} RGB EARLIER than model.",
            file=sys.stderr,
        )
    else:
        print("[CALIBRATE]   Matches default.", file=sys.stderr)

    # Write result JSON next to the scan
    out_path = str(Path(scan_path).with_suffix(".json"))
    curve = dict(DEFAULT_CURVE_VALUES)
    curve["black_clip"] = float(black_clip)

    meta = _curve_meta_default()
    meta["black_clip"]["calibrated"] = True
    meta["black_clip"]["locked"] = False
    meta["black_clip"]["source"] = "physical_scan"

    data = {
        "version": "1.0",
        "profile": Path(scan_path).stem,
        "calibrated_at": datetime.now().isoformat(timespec="seconds"),
        "curve": curve,
        "curve_meta": meta,
        "calibration": {
            "method": "physical_scan",
            "target_path": os.path.abspath(target_path),
            "scan_path": os.path.abspath(scan_path),
            "white_patch": round(white_mean, 2),
            "black_patch": round(black_mean, 2),
            "black_std": round(black_std, 2),
            "noise_norm": round(noise_norm, 5),
            "threshold_norm": round(threshold, 5),
            "first_above_rgb": int(first_above),
            "confidence": confidence,
        },
        "warning": (
            "Only black_clip is calibrated against your physical print. "
            "gamma/desat_thresh/desat_amount/scale_factor remain from reverse "
            "engineering and are NOT validated for your printer. "
            "Scanner adds ±2-3 RGB uncertainty. Use --prepress-filter with this file, "
            "then verify on a physical proof before running the full batch."
        ),
    }
    _write_json(out_path, data)
    print(
        f"[CALIBRATE] Use with: "
        f"python shadow_lift_depth_shadow_prepress.py -i cards/ -o out/ "
        f"--tgc-lift stress --prepress-filter --custom-curve \"{out_path}\"",
        file=sys.stderr,
    )
    return 0


# ============================================================
# CLI
# ============================================================

def main() -> int:
    p = argparse.ArgumentParser(
        description="Shadow-Lift calibration toolkit — standalone (numpy + Pillow only).",
        epilog=(
            "Examples:\n"
            "  python calibrate.py --generate-probe target.png\n"
            "  python calibrate.py target.png my_scan.png\n"
            "  python calibrate.py --dump-default default_curve.json\n"
            "  python calibrate.py --generate-probe-legacy ./probes/\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--generate-probe", metavar="OUT.png", default=None,
                   help="Generate calibration target (900x1500, 41 steps, ref patches).")
    p.add_argument("--generate-probe-legacy", metavar="OUT_DIR", default=None,
                   help="Generate the old tgc_blackbox_probe_legacy.png / "
                        "tgc_highlight_probe_legacy.png (visual inspection only).")
    p.add_argument("--dump-default", metavar="OUT.json", default=None,
                   help="Write default_curve.json (for main script auto-load).")

    p.add_argument("files", nargs="*", metavar="FILE",
                   help="TARGET SCAN — fit black_clip against a scanned probe.")

    args = p.parse_args()

    # Dispatch modes
    if args.generate_probe:
        generate_probe_calibration(args.generate_probe)
        return 0

    if args.generate_probe_legacy:
        generate_probe_legacy(args.generate_probe_legacy)
        return 0

    if args.dump_default:
        dump_default_curve(args.dump_default)
        return 0

    if len(args.files) == 2:
        return calibrate(args.files[0], args.files[1])

    if len(args.files) == 0:
        p.print_help()
        return 0

    print(
        f"Error: expected 2 positional args (TARGET SCAN), got {len(args.files)}.\n"
        f"       Example: python calibrate.py target.png my_scan.png",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())