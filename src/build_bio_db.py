#!/usr/bin/env python3
"""
build_bio_db.py

Create SQLite DB 'Eagle_Eye_Detection_Pipeline.db' and populate:
- Persons(person_id PRIMARY KEY, first_name, last_name, age)  <-- person_id unique here
- Bio_Data(person_id FK -> Persons(person_id), first_name, last_name, age, file_path, UNIQUE(person_id, file_path))

Behavior:
- images_root contains subfolders per person, named like 'First_Last' (same person per subfolder).
- bios_json is JSON with dicts: {"first_name": "...", "last_name": "...", "age": <int>}.
- For each folder, find its bio (case-insensitive), upsert into Persons, then insert one Bio_Data row per image.

Usage:
  python build_bio_db.py --images-root ./people_root --bios-json ./people_bios.json
"""

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, Tuple, List

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif"}

DB_NAME = "Eagle_Eye_Detection_Pipeline.db"
TABLE_BIO = "Bio_Data"
TABLE_PERSONS = "Persons"

# ------------------------- Name normalization -------------------------

_SEP_RX = re.compile(r"[ _\-]+")
_NONALPHA_RX = re.compile(r"[^a-z]+")

def _norm_pair_key(first: str, last: str) -> str:
    """Normalize (first, last) to 'firstname lastname' (letters-only, lower)."""
    f = _NONALPHA_RX.sub("", first.lower().strip())
    l = _NONALPHA_RX.sub("", last.lower().strip())
    return f"{f} {l}".strip()

def _parse_folder_person(folder_name: str) -> Tuple[str, str]:
    """
    Parse 'First_Last' / 'First Last' / 'First-Last' into (first, last).
    If >2 tokens, use first and last tokens.
    """
    raw = folder_name.strip()
    tokens = [t for t in _SEP_RX.split(raw) if t]
    if not tokens:
        return raw, ""
    if len(tokens) == 1:
        return tokens[0], ""
    return tokens[0], tokens[-1]

def _folder_key(folder_name: str) -> str:
    first, last = _parse_folder_person(folder_name)
    return _norm_pair_key(first, last)

# ------------------------- JSON loading -------------------------

def load_bios_mapping(json_path: Path) -> Dict[str, Dict]:
    """
    Accepts a list of dicts or dict of dicts with keys: first_name, last_name, age.
    Returns mapping: normalized "firstname lastname" -> {"first_name","last_name","age"}
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    mapping: Dict[str, Dict] = {}

    def add_entry(first: str, last: str, age) -> None:
        key = _norm_pair_key(first, last)
        if not key:
            return
        if key not in mapping:
            mapping[key] = {
                "first_name": str(first).strip(),
                "last_name":  str(last).strip(),
                "age":        int(age),
            }

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and {"first_name", "last_name", "age"} <= set(item.keys()):
                add_entry(item["first_name"], item["last_name"], item["age"])
    elif isinstance(data, dict):
        for v in data.values():
            if isinstance(v, dict) and {"first_name", "last_name", "age"} <= set(v.keys()):
                add_entry(v["first_name"], v["last_name"], v["age"])
    else:
        raise ValueError("Unsupported bios JSON structure (expected list or dict).")

    if not mapping:
        raise ValueError("No valid bio entries found in JSON.")
    return mapping

# ------------------------- DB helpers -------------------------

def ensure_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON;")
    with conn:
        # Canonical people table with UNIQUE person_id
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_PERSONS} (
                person_id  TEXT PRIMARY KEY,
                first_name TEXT NOT NULL,
                last_name  TEXT NOT NULL,
                age        INTEGER NOT NULL
            )
            """
        )
        # Bio_Data references Persons(person_id); allows multiple images per person
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_BIO} (
                person_id  TEXT NOT NULL,
                first_name TEXT NOT NULL,
                last_name  TEXT NOT NULL,
                age        INTEGER NOT NULL,
                file_path  TEXT NOT NULL,
                UNIQUE(person_id, file_path),
                FOREIGN KEY (person_id) REFERENCES {TABLE_PERSONS}(person_id)
            )
            """
        )
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_bio_person ON {TABLE_BIO}(person_id)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_bio_file ON {TABLE_BIO}(file_path)")
    return conn

def upsert_person(conn: sqlite3.Connection, person_id: str, first: str, last: str, age: int) -> None:
    """Ensure a row exists in Persons; update name/age if person_id already present."""
    with conn:
        conn.execute(
            f"""
            INSERT INTO {TABLE_PERSONS} (person_id, first_name, last_name, age)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(person_id) DO UPDATE SET
              first_name=excluded.first_name,
              last_name=excluded.last_name,
              age=excluded.age
            """,
            (person_id, first, last, int(age)),
        )

def insert_bio_row(conn: sqlite3.Connection, person_id: str, first: str, last: str, age: int, file_path: str) -> bool:
    """Insert one image row into Bio_Data (FK requires person to exist in Persons)."""
    try:
        with conn:
            conn.execute(
                f"INSERT OR IGNORE INTO {TABLE_BIO} (person_id, first_name, last_name, age, file_path) VALUES (?, ?, ?, ?, ?)",
                (person_id, first, last, int(age), file_path),
            )
        return True
    except Exception as e:
        print(f"[ERR] Insert failed for {file_path}: {e}")
        return False

# ------------------------- Main import -------------------------

def import_folders(images_root: Path, bios_map: Dict[str, Dict], db_path: Path) -> None:
    conn = ensure_db(db_path)

    total_folders = matched_folders = images_inserted = 0
    unmatched_folders: List[str] = []
    skipped_non_images = empty_folders = 0

    for sub in sorted(images_root.iterdir()):
        if not sub.is_dir():
            continue
        total_folders += 1
        person_id = sub.name

        key = _folder_key(person_id)
        bio = bios_map.get(key)
        if bio is None:
            unmatched_folders.append(person_id)
            continue

        # Ensure person_id exists ONCE in Persons (person_id is UNIQUE here)
        upsert_person(conn, person_id, bio["first_name"], bio["last_name"], bio["age"])

        # Insert each image in this folder
        imgs = [p for p in sorted(sub.iterdir()) if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS]
        if not imgs:
            empty_folders += 1
        for p in sub.iterdir():
            if p.is_file() and p.suffix.lower() not in VALID_IMAGE_EXTS:
                skipped_non_images += 1

        for img in imgs:
            ok = insert_bio_row(
                conn,
                person_id=person_id,
                first=bio["first_name"],
                last=bio["last_name"],
                age=bio["age"],
                file_path=str(img.resolve()),
            )
            if ok:
                images_inserted += 1

        matched_folders += 1

    conn.close()

    # Summary
    print("\n=== Import Summary ===")
    print(f"DB: {db_path}")
    print(f"Tables: {TABLE_PERSONS}, {TABLE_BIO}")
    print(f"Total folders found:         {total_folders}")
    print(f"Matched folders:             {matched_folders}")
    print(f"Unmatched folders (skipped): {len(unmatched_folders)}")
    if unmatched_folders:
        preview = ", ".join(unmatched_folders[:10])
        print(f"  -> {preview}{' ...' if len(unmatched_folders) > 10 else ''}")
    print(f"Empty folders:               {empty_folders}")
    print(f"Images inserted:             {images_inserted}")
    print(f"Non-image files skipped:     {skipped_non_images}")

# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build Eagle_Eye_Detection_Pipeline.db (Persons + Bio_Data) from folder structure + bios JSON (unordered)."
    )
    ap.add_argument("--images-root", required=True, help="Path to root folder containing person subfolders (e.g., First_Last)")
    ap.add_argument("--bios-json", required=True, help="Path to JSON with dictionaries of first_name, last_name, age (unordered)")
    ap.add_argument("--db", default=DB_NAME, help=f"SQLite DB path (default: {DB_NAME})")
    args = ap.parse_args()

    images_root = Path(args.images_root)
    if not images_root.exists() or not images_root.is_dir():
        raise SystemExit(f"[ERR] --images-root must be an existing directory: {images_root}")

    bios_path = Path(args.bios_json)
    if not bios_path.exists() or not bios_path.is_file():
        raise SystemExit(f"[ERR] --bios-json must be an existing file: {bios_path}")

    try:
        bios_map = load_bios_mapping(bios_path)
    except Exception as e:
        raise SystemExit(f"[ERR] Failed to read bios JSON: {e}")

    import_folders(images_root, bios_map, Path(args.db))

if __name__ == "__main__":
    main()
