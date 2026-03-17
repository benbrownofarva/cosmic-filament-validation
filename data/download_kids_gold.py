#!/usr/bin/env python3
"""
Download KiDS-1000 weak-lensing shape catalog for filament validation.

Instead of the full ~12 GB catalog, we use the ESO archive or the
KiDS public data server to get tiles overlapping our filament footprint.

KiDS-1000 data products:
  https://kids.strw.leidenuniv.nl/DR4/KiDS-1000_shearcatalogue.php

The gold WL catalog is a single 16 GB FITS table with 21,262,011 sources.
"""
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)

OUTPUT_DIR = Path(__file__).parent / "kids_lensing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The KiDS-1000 gold WL catalog is a single ~16 GB file (21M sources)
GOLD_CATALOG_URL = (
    "https://kids.strw.leidenuniv.nl/DR4/data_files/"
    "KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits"
)

# For our validation, we need: RA, Dec, e1, e2, weight, z_B (photo-z)
# The gold catalog contains all of this

print("KiDS-1000 Gold WL Catalog Download")
print("=" * 50)
print(f"URL: {GOLD_CATALOG_URL}")
print(f"Expected size: ~16 GB")
print(f"Output: {OUTPUT_DIR}")
print()

output_path = OUTPUT_DIR / "KiDS_DR4.1_WL_gold_cat.fits"
if output_path.exists():
    size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"Already exists: {output_path} ({size_gb:.2f} GB)")
    sys.exit(0)

print("Starting download (this will take a while)...")
try:
    resp = requests.get(GOLD_CATALOG_URL, timeout=30, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024*1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = 100 * downloaded / total
                gb = downloaded / (1024**3)
                print(f"\r  {gb:.2f} GB ({pct:.1f}%)", end="", flush=True)
    print()
    print(f"Done: {os.path.getsize(output_path)/1e9:.2f} GB")
except Exception as e:
    print(f"FAILED: {e}")
    print(f"Manual download: wget -c '{GOLD_CATALOG_URL}' -O '{output_path}'")
    if output_path.exists():
        output_path.unlink()
