"""
Shadow-Lift model fingerprint.
Reports: torch/torchvision versions, MiDaS repo snapshot, weights SHA256.
Safe to run repeatedly — no downloads, no model loading, read-only.

Usage:
    python check_models.py
    python check_models.py --verbose    # also list all cached weight files
"""

from __future__ import annotations
import argparse
import hashlib
import os
import sys
from pathlib import Path


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def find_hub_dir() -> Path:
    # torch.hub.get_dir() is the canonical source; fall back to defaults
    try:
        import torch
        return Path(torch.hub.get_dir())
    except Exception:
        env = os.environ.get("TORCH_HOME")
        if env:
            return Path(env) / "hub"
        return Path.home() / ".cache" / "torch" / "hub"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true",
                    help="list every cached weight file, not just MiDaS")
    args = ap.parse_args()

    print("=" * 66)
    print(" Shadow-Lift — model fingerprint")
    print("=" * 66)

    # --- Python / torch ---
    print(f"\n[PYTHON]   {sys.version.split()[0]}  ({sys.executable})")

    try:
        import torch
        print(f"[TORCH]    {torch.__version__}")
        print(f"[CUDA]     available={torch.cuda.is_available()}", end="")
        if torch.cuda.is_available():
            print(f"  device={torch.cuda.get_device_name(0)}")
        else:
            print()
    except ImportError:
        print("[TORCH]    NOT INSTALLED")
        torch = None

    try:
        import torchvision
        print(f"[VISION]   {torchvision.__version__}")
    except ImportError:
        print("[VISION]   NOT INSTALLED")

    # --- Torch hub cache ---
    hub_dir = find_hub_dir()
    print(f"\n[HUB DIR]  {hub_dir}")
    if not hub_dir.is_dir():
        print("           (does not exist — MiDaS has not been loaded yet)")
        print("\n  Run the engine once with --debug to populate the cache,")
        print("  then re-run this script.")
        return 0

    # --- MiDaS repo snapshot ---
    # torch.hub caches repos as <hub_dir>/<owner>_<repo>_<branch>
    midas_repos = sorted(hub_dir.glob("intel-isl_MiDaS_*")) + \
                  sorted(hub_dir.glob("isl-org_MiDaS_*"))
    if not midas_repos:
        print("\n[MIDAS]    no cached repo snapshot found")
    else:
        for repo in midas_repos:
            print(f"\n[MIDAS]    {repo.name}")
            # try to read the git commit if it's a real clone
            head = repo / ".git" / "HEAD"
            if head.is_file():
                ref = head.read_text().strip()
                if ref.startswith("ref:"):
                    ref_path = repo / ".git" / ref.split(":", 1)[1].strip()
                    if ref_path.is_file():
                        ref = ref_path.read_text().strip()
                print(f"           commit: {ref}")
            # enumerate weight files inside the snapshot
            weights = sorted(repo.rglob("*.pt")) + sorted(repo.rglob("*.pth"))
            if weights:
                print(f"           weights inside snapshot:")
                for w in weights:
                    size = human(w.stat().st_size)
                    digest = sha256_of(w)
                    print(f"             {w.relative_to(repo)}  "
                          f"[{size}]  sha256={digest[:16]}…")
            else:
                print(f"           (no .pt/.pth files inside snapshot)")

    # --- Weights in the top-level checkpoints/ folder ---
    # torch.hub.download_url_to_file puts weights in <hub_dir>/checkpoints/
    ckpt_dir = hub_dir / "checkpoints"
    if ckpt_dir.is_dir():
        midas_ckpts = sorted(
            p for p in ckpt_dir.glob("*.pt")
            if "midas" in p.name.lower()
        ) + sorted(
            p for p in ckpt_dir.glob("*.pth")
            if "midas" in p.name.lower()
        )
        if midas_ckpts:
            print(f"\n[WEIGHTS]  {ckpt_dir}")
            for w in midas_ckpts:
                size = human(w.stat().st_size)
                digest = sha256_of(w)
                print(f"           {w.name}  [{size}]")
                print(f"             sha256={digest}")
        elif args.verbose:
            print(f"\n[WEIGHTS]  {ckpt_dir}  (no MiDaS files)")

    if args.verbose:
        print(f"\n[ALL CACHED WEIGHTS]  {ckpt_dir if ckpt_dir.is_dir() else '(none)'}")
        if ckpt_dir.is_dir():
            for w in sorted(ckpt_dir.iterdir()):
                if w.is_file():
                    print(f"           {w.name}  [{human(w.stat().st_size)}]")

    # --- Local project cache tag ---
    print("\n[PROJECT CACHE TAG]")
    print("  In shadow_lift_depth_shadow_prepress.py:")
    print("    DEPTH_MODEL_TAG  = \"midas_small_v1\"   ← manually maintained")
    print("    SHADOW_MODEL_TAG = \"classical_v1\"     ← in-repo deterministic")
    print("  These tags do NOT auto-detect upstream model changes.")
    print("  Compare the weights sha256 above with a previous run to detect drift.")

    print("\n" + "=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())