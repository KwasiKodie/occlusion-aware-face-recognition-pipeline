#!/usr/bin/env python3
import argparse
import json
import re
import shutil
from pathlib import Path

"""
python rename_nomask.py --src "F:/oafr backups/public/occlusion aware face recognition/Data/COMASK20/dataset" --json "E:/publication_test/utilities_2/people.json" --out "E:/publication_test/data/persons/" --max 5
"""

# Regex: match filenames that indicate "no mask" (case-insensitive, common variants)
NOMASK_RE = re.compile(r"(?:\bno\s*[-_ ]*\s*mask\b|nomask|without\s*mask)", re.IGNORECASE)

def is_nomask(name: str) -> bool:
    """Return True if filename (without path) suggests 'no mask'."""
    return bool(NOMASK_RE.search(name))

def sanitize_name(s: str) -> str:
    """Filesystem-friendly: lowercase, underscores, alnum-only (plus underscore)."""
    s = s.strip().lower().replace(" ", "_")
    # keep letters, digits, underscores
    return re.sub(r"[^a-z0-9_]", "", s)

def pick_nomask_images(folder: Path, max_count: int = 5):
    """Pick up to `max_count` images with 'nomask' in name from `folder`."""
    # common image extensions
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    # stable order: sort by name so runs are deterministic
    files.sort(key=lambda p: p.name.lower())
    return [p for p in files if is_nomask(p.name)][:max_count]

def main():
    ap = argparse.ArgumentParser(
        description="Copy & rename 5 'nomask' images per folder using names from a JSON list."
    )
    ap.add_argument("--src", required=True, help="Root folder that contains the 312 subfolders")
    ap.add_argument("--json", required=True, help="Path to JSON file (list of 312 dicts with first_name, last_name, age)")
    ap.add_argument("--out", required=True, help="Output root folder where per-person folders will be created")
    ap.add_argument("--max", type=int, default=5, help="How many images to copy per folder (default: 5)")
    args = ap.parse_args()

    src_root = Path(args.src).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Load JSON (list of dicts)
    with open(args.json, "r", encoding="utf-8") as f:
        people = json.load(f)

    # Index subfolders (deterministic: sort by folder name)
    subfolders = [p for p in src_root.iterdir() if p.is_dir()]
    subfolders.sort(key=lambda p: p.name.lower())

    if len(people) != len(subfolders):
        print(f"[WARN] JSON entries ({len(people)}) != subfolders ({len(subfolders)}). Proceeding by index alignment.")

    total_copied = 0
    for idx, folder in enumerate(subfolders):
        if idx >= len(people):
            print(f"[SKIP] No JSON entry for folder index {idx}: {folder.name}")
            continue

        entry = people[idx]
        first = sanitize_name(str(entry.get("first_name", "first")))
        last = sanitize_name(str(entry.get("last_name", "last")))
        person_dir = out_root / f"{first}_{last}"
        person_dir.mkdir(parents=True, exist_ok=True)

        candidates = pick_nomask_images(folder, max_count=args.max)
        if not candidates:
            print(f"[INFO] No 'nomask' files found in: {folder}")
            continue

        for i, src_img in enumerate(candidates, start=1):
            ext = src_img.suffix.lower()
            dst = person_dir / f"{first}_{last}_{i}{ext}"
            shutil.copy2(src_img, dst)
            total_copied += 1

        if len(candidates) < args.max:
            print(f"[INFO] Only found {len(candidates)}/{args.max} 'nomask' images in: {folder}")

    print(f"Done. Copied {total_copied} image(s) into {out_root}")

if __name__ == "__main__":
    main()
