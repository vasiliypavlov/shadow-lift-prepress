"""Shadow-Lift model fingerprint and downloader toolkit.
Reports: torch/torchvision versions, MiDaS repo snapshot, weights SHA256.
Verifies model presence and integrity against the author-pinned reference.
Prompts to download missing or mismatched models into the ./models folder.

Usage:
    python check_models.py
    python check_models.py --download          # auto-download without prompting
    python check_models.py --force-download    # re-download author version
    python check_models.py --verbose           # list all cached weight files
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, List

# --- Constants aligned with shadow_lift_depth_shadow_prepress.py ---
MIDAS_WEIGHTS_FILENAME = "midas_v21_small_256.pt"
EXPECTED_MIDAS_WEIGHTS_SHA256 = (
    "70d6b9c891758c67f974a6097fb0c608c7ee67fb81ac3e5588847d5596d56fca"
)
EXPECTED_SHA8 = EXPECTED_MIDAS_WEIGHTS_SHA256[:8]
MIDAS_WEIGHTS_URL = (
    "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt"
)
DEPTH_MODEL_TAG = "midas_small_70d6b9c8"
SHADOW_MODEL_TAG = "classical_v1"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """Calculates SHA256 digest of file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def human(n: int | float) -> str:
    """Formats bytes to human-readable string."""
    val = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if val < 1024:
            return f"{val:.1f} {unit}"
        val /= 1024
    return f"{val:.1f} TB"


def _print_mismatch_banner(expected: str, actual: str, ckpt: str) -> None:
    """Prints mismatch alert banner identically to shadow_lift_depth_shadow_prepress.py."""
    bar = "!" * 62
    print("", file=sys.stderr)
    print("  " + bar, file=sys.stderr)
    print("  [WARN] MiDaS weights differ from pinned reference.", file=sys.stderr)
    print(f"         Expected: {expected}", file=sys.stderr)
    print(f"         Actual:   {actual}", file=sys.stderr)
    print(f"         File:     {ckpt}", file=sys.stderr)
    print("  " + bar, file=sys.stderr)
    print("", file=sys.stderr)


def find_hub_dir() -> Path:
    """torch.hub.get_dir() is canonical; fall back to standard defaults."""
    try:
        import torch
        return Path(torch.hub.get_dir())
    except Exception:
        env = os.environ.get("TORCH_HOME")
        if env:
            return Path(env) / "hub"
        return Path.home() / ".cache" / "torch" / "hub"


def prompt_user_yes_no(question: str, default: bool = True, auto_yes: bool = False) -> bool:
    """Asks user a yes/no question. If non-interactive and no auto_yes, returns False."""
    if auto_yes:
        print(f"  {question} [auto-accepted: YES]")
        return True

    suffix = "[Y/n]: " if default else "[y/N]: "
    if not sys.stdin.isatty():
        print(f"  {question} (Non-interactive terminal — skipping. Use --download to auto-accept)", file=sys.stderr)
        return False

    try:
        ans = input(f"  {question} {suffix}").strip().lower()
        if not ans:
            return default
        return ans in ("y", "yes", "д", "да")
    except (EOFError, KeyboardInterrupt):
        print("\n  Aborted by user.", file=sys.stderr)
        return False


def download_file_with_progress(
    url: str,
    target_path: Path,
    expected_sha256: str
) -> bool:
    """
    Downloads file with a clean progress bar, computes SHA256 streaming,
    and atomically moves to target_path on verification success.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".tmp_download")

    print(f"\n  [DOWNLOAD] Connecting to {url}...")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Shadow-Lift-Prepress)"}
        )
        with urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as out_f:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            hasher = hashlib.sha256()
            chunk_size = 1 << 20  # 1MB chunks
            t0 = time.time()

            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_f.write(chunk)
                hasher.update(chunk)
                downloaded += len(chunk)

                # Progress bar
                elapsed = max(0.001, time.time() - t0)
                speed = downloaded / elapsed
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    bar_w = 28
                    filled = int(bar_w * downloaded / total_size)
                    bar = "=" * filled + ">" + " " * max(0, bar_w - filled - 1)
                    print(
                        f"\r  [{bar[:bar_w]}] {pct:5.1f}%  {human(downloaded)}/{human(total_size)}  "
                        f"({human(speed)}/s)  ",
                        end="",
                        flush=True,
                    )
                else:
                    print(f"\r  Downloaded {human(downloaded)} ({human(speed)}/s)...", end="", flush=True)

        print("\n  [DOWNLOAD] Download complete. Verifying SHA256 checksum...")
        actual_sha256 = hasher.hexdigest()

        if actual_sha256.lower() != expected_sha256.lower():
            print(f"  [ERROR] SHA256 mismatch on downloaded file!", file=sys.stderr)
            print(f"          Expected: {expected_sha256}", file=sys.stderr)
            print(f"          Actual:   {actual_sha256}", file=sys.stderr)
            if tmp_path.is_file():
                tmp_path.unlink()
            return False

        # Atomic rename
        if target_path.is_file():
            target_path.unlink()
        tmp_path.replace(target_path)
        print(f"  [OK] Saved to: {target_path}")
        print(f"  [OK] Checksum verified: sha256={actual_sha256[:8]}...")
        return True

    except Exception as exc:
        print(f"\n  [ERROR] Download failed: {exc}", file=sys.stderr)
        if tmp_path.is_file():
            tmp_path.unlink()
        return False


def install_author_model_into_models_dir(models_dir: Path) -> bool:
    """
    Downloads author's pinned model into `./models` folder.
    Creates checkpoints/midas_v21_small_256.pt and root models/midas_v21_small_256.pt.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = models_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_file = ckpt_dir / MIDAS_WEIGHTS_FILENAME
    root_file = models_dir / MIDAS_WEIGHTS_FILENAME

    success = download_file_with_progress(
        MIDAS_WEIGHTS_URL,
        ckpt_file,
        EXPECTED_MIDAS_WEIGHTS_SHA256,
    )
    if not success:
        return False

    # Also place/symlink in models root for convenience
    try:
        if not root_file.is_file():
            try:
                root_file.symlink_to(ckpt_file)
            except Exception:
                shutil.copy2(ckpt_file, root_file)
    except Exception:
        pass

    print(f"\n  [SUCCESS] Model installed in local folder: {models_dir}")
    print(f"            shadow_lift_depth_shadow_prepress.py will automatically pick it up via _resolve_models_dir().")
    return True


def locate_existing_models(models_dir: Path, hub_dir: Path) -> List[Tuple[Path, str, bool]]:
    """
    Finds all potential MiDaS weight files in local models folder and hub cache.
    Returns: list of (Path, sha256, is_exact_match)
    """
    candidates: List[Path] = []

    # 1. Check local models dir
    if models_dir.is_dir():
        for p in models_dir.rglob("*.pt"):
            if "midas" in p.name.lower() or p.name == MIDAS_WEIGHTS_FILENAME:
                candidates.append(p)
        for p in models_dir.rglob("*.pth"):
            if "midas" in p.name.lower():
                candidates.append(p)

    # 2. Check torch hub dir
    if hub_dir.is_dir():
        ckpt_dir = hub_dir / "checkpoints"
        if ckpt_dir.is_dir():
            for p in ckpt_dir.glob("*.pt"):
                if "midas" in p.name.lower() or p.name == MIDAS_WEIGHTS_FILENAME:
                    candidates.append(p)
            for p in ckpt_dir.glob("*.pth"):
                if "midas" in p.name.lower():
                    candidates.append(p)

    # Deduplicate
    seen = set()
    unique_candidates: List[Path] = []
    for c in candidates:
        r = c.resolve()
        if r not in seen and c.is_file():
            seen.add(r)
            unique_candidates.append(c)

    results: List[Tuple[Path, str, bool]] = []
    for c in unique_candidates:
        digest = sha256_of(c)
        is_match = (digest.lower() == EXPECTED_MIDAS_WEIGHTS_SHA256.lower())
        results.append((c, digest, is_match))

    return results


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify MiDaS depth models for Shadow-Lift Prepress and download author-pinned versions."
    )
    ap.add_argument(
        "--verbose", action="store_true", help="list every cached weight file, not just MiDaS"
    )
    ap.add_argument(
        "-d", "--download", "--yes", "-y",
        dest="auto_download",
        action="store_true",
        help="automatically download correct models without asking confirmation",
    )
    ap.add_argument(
        "--force-download",
        action="store_true",
        help="force download/overwrite author's model into ./models even if a file already exists",
    )
    ap.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="path to local models directory (default: ./models next to script)",
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    local_models_dir = args.models_dir or (script_dir / "models")

    print("=" * 66)
    print(" Shadow-Lift — Model Inspector & Downloader")
    print("=" * 66)

    # --- Python / Torch Environment ---
    print(f"\n[PYTHON]   {sys.version.split()[0]}  ({sys.executable})")
    torch = None
    try:
        import torch
        print(f"[TORCH]    {torch.__version__}")
        print(f"[CUDA]     available={torch.cuda.is_available()}", end="")
        if torch.cuda.is_available():
            print(f"  device={torch.cuda.get_device_name(0)}")
        else:
            print()
    except ImportError:
        print("[TORCH]    NOT INSTALLED (standalone download mode active)")

    try:
        import torchvision
        print(f"[VISION]   {torchvision.__version__}")
    except ImportError:
        print("[VISION]   NOT INSTALLED")

    # --- Paths ---
    hub_dir = find_hub_dir()
    print(f"\n[LOCAL MODELS] {local_models_dir} (exists={local_models_dir.is_dir()})")
    print(f"[HUB DIR]      {hub_dir} (exists={hub_dir.is_dir()})")

    # --- Inspect cached MiDaS repo snapshot ---
    if hub_dir.is_dir():
        midas_repos = sorted(hub_dir.glob("intel-isl_MiDaS_*")) + sorted(hub_dir.glob("isl-org_MiDaS_*"))
        if midas_repos:
            for repo in midas_repos:
                print(f"[REPO SNAPSHOT] {repo.name}")

    # --- Force download requested ---
    if args.force_download:
        print("\n[ACTION] --force-download requested.")
        ok = install_author_model_into_models_dir(local_models_dir)
        return 0 if ok else 1

    # --- Find and check models ---
    print(f"\n[CHECKING WEIGHTS]")
    print(f"  Target File:     {MIDAS_WEIGHTS_FILENAME}")
    print(f"  Expected SHA256: {EXPECTED_MIDAS_WEIGHTS_SHA256}")
    print(f"  Tag:             {DEPTH_MODEL_TAG}")

    existing_models = locate_existing_models(local_models_dir, hub_dir)

    exact_matches = [m for m in existing_models if m[2]]
    mismatches = [m for m in existing_models if not m[2]]

    # Case 1: An exact match already exists!
    if exact_matches:
        for path, digest, _ in exact_matches:
            size_str = human(path.stat().st_size)
            print(f"\n  ✅ [OK] Verified author-pinned model found:")
            print(f"     Path:   {path}  [{size_str}]")
            print(f"     SHA256: {digest}")

        # Check if it's already in local models dir
        in_local = any(local_models_dir.resolve() in m[0].resolve().parents for m in exact_matches)
        if in_local:
            print(f"     Status: Ready for offline execution via local ./models directory.")
        else:
            print(f"     Status: Found in system torch cache ({hub_dir}).")

        print("\n" + "=" * 66)
        return 0

    # Case 2: Model files exist, BUT none match the author's expected version
    if mismatches and not exact_matches:
        print("\n  ❌ [MISMATCH] Model files were found, but their versions DO NOT match the project reference:")
        for path, digest, _ in mismatches:
            size_str = human(path.stat().st_size)
            print(f"     - {path} [{size_str}]")
            print(f"       Actual SHA256:   {digest}")
            print(f"       Expected SHA256: {EXPECTED_MIDAS_WEIGHTS_SHA256}")
            _print_mismatch_banner(EXPECTED_MIDAS_WEIGHTS_SHA256, digest, str(path))

        question = (
            f"Model version mismatch! Do you want to download the author's pinned model\n"
            f"  ({MIDAS_WEIGHTS_FILENAME}, sha256: {EXPECTED_SHA8}…) into '{local_models_dir}'?"
        )

        should_download = prompt_user_yes_no(question, default=True, auto_yes=args.auto_download)
        if should_download:
            ok = install_author_model_into_models_dir(local_models_dir)
            print("\n" + "=" * 66)
            return 0 if ok else 1
        else:
            print("\n  [WARN] Proceeding with mismatched models may cause degraded depth maps or script aborts.", file=sys.stderr)
            print("         To bypass in shadow_lift_depth_shadow_prepress.py, use --accept-models.", file=sys.stderr)
            print("\n" + "=" * 66)
            return 2

    # Case 3: No models found at all
    print(f"\n  ⚠️  [MISSING] No MiDaS weights ('{MIDAS_WEIGHTS_FILENAME}') found in:")
    print(f"      - {local_models_dir}")
    print(f"      - {hub_dir / 'checkpoints'}")

    question = (
        f"MiDaS model is missing. Would you like to download it now from GitHub\n"
        f"  into local folder '{local_models_dir}'?"
    )

    should_download = prompt_user_yes_no(question, default=True, auto_yes=args.auto_download)
    if should_download:
        ok = install_author_model_into_models_dir(local_models_dir)
        print("\n" + "=" * 66)
        return 0 if ok else 1
    else:
        print(f"\n  [INFO] You can download it later manually by running:", file=sys.stderr)
        print(f"         python check_models.py --download", file=sys.stderr)
        print("\n" + "=" * 66)
        return 1


if __name__ == "__main__":
    sys.exit(main())
