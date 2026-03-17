#!/usr/bin/env python3
"""
Download public SDSS cosmic filament catalogs from VizieR/CDS.

Two catalogs:
1. Tempel+ 2014 (Bisous model): J/MNRAS/438/3465
2. Galárraga-Espinosa+ 2022 (DisPerSE-based): J/A+A/661/A115

All data are public. No authentication required.
"""

import os
import sys
import hashlib
import json
from pathlib import Path
from datetime import datetime

# We use astroquery if available, otherwise fall back to direct HTTP
try:
    from astroquery.vizier import Vizier
    HAS_ASTROQUERY = True
except ImportError:
    HAS_ASTROQUERY = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# --- Configuration ---

OUTPUT_DIR = Path(__file__).parent / "sdss_filaments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATALOGS = {
    "tempel2014_bisous": {
        "vizier_id": "J/MNRAS/438/3465",
        "description": "Tempel+ 2014 Bisous-model filament catalog from SDSS",
        "reference": "Tempel, Stoica, Saar 2014, MNRAS 438, 3465",
        "tables": [
            "filaments",   # filament properties
            "galaxies",    # galaxy membership
        ],
        # Direct CDS download URLs (TSV format)
        "direct_urls": {
            "filaments": "https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/tsv?J/MNRAS/438/3465/filaments",
            "galaxies": "https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/tsv?J/MNRAS/438/3465/galaxies",
        },
        # Alternative: full catalog tarball
        "tar_url": "https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/tar.gz?J/MNRAS/438/3465",
    },
    "galarraga2022_disperse": {
        "vizier_id": "J/A+A/661/A115",
        "description": "Galárraga-Espinosa+ 2022 SDSS filament catalog",
        "reference": "Galárraga-Espinosa, Aghanim, Bonnaire, Tanimura 2022, A&A 661, A115",
        "tables": [
            "filaments",
            "nodes",
        ],
        "direct_urls": {
            "filaments": "https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/tsv?J/A+A/661/A115/filaments",
            "nodes": "https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/tsv?J/A+A/661/A115/nodes",
        },
        "tar_url": "https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/tar.gz?J/A+A/661/A115",
    },
}


def sha256_file(filepath):
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_via_requests(url, output_path):
    """Download a file using requests library."""
    if not HAS_REQUESTS:
        print(f"  ERROR: 'requests' not installed. Run: pip install requests")
        return False

    print(f"  Downloading: {url}")
    print(f"  Target: {output_path}")

    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Downloaded: {size_mb:.2f} MB")
        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def download_via_astroquery(vizier_id, table_name, output_path):
    """Download a VizieR table using astroquery."""
    if not HAS_ASTROQUERY:
        print(f"  ERROR: 'astroquery' not installed. Run: pip install astroquery")
        return False

    print(f"  Querying VizieR: {vizier_id}/{table_name}")

    try:
        v = Vizier(catalog=vizier_id, row_limit=-1)
        tables = v.get_catalogs(vizier_id)

        if len(tables) == 0:
            print(f"  WARNING: No tables returned for {vizier_id}")
            return False

        # Find the matching table
        for t in tables:
            if table_name.lower() in t.meta.get("name", "").lower():
                t.write(str(output_path), format="ascii.csv", overwrite=True)
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"  Downloaded: {size_mb:.2f} MB")
                return True

        # If exact match not found, save the first table
        print(f"  WARNING: Exact table '{table_name}' not found. Saving first table.")
        tables[0].write(str(output_path), format="ascii.csv", overwrite=True)
        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def download_catalog(catalog_key, catalog_info):
    """Download all tables for a catalog."""
    catalog_dir = OUTPUT_DIR / catalog_key
    catalog_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Catalog: {catalog_info['description']}")
    print(f"Reference: {catalog_info['reference']}")
    print(f"VizieR ID: {catalog_info['vizier_id']}")
    print(f"{'='*60}")

    provenance = {
        "catalog_key": catalog_key,
        "vizier_id": catalog_info["vizier_id"],
        "reference": catalog_info["reference"],
        "download_date": datetime.utcnow().isoformat() + "Z",
        "files": {},
    }

    success_count = 0

    # Strategy 1: Try direct URL download first (most reliable)
    for table_name, url in catalog_info.get("direct_urls", {}).items():
        output_path = catalog_dir / f"{table_name}.tsv"

        print(f"\n--- Table: {table_name} ---")
        if download_via_requests(url, output_path):
            file_hash = sha256_file(output_path)
            provenance["files"][table_name] = {
                "path": str(output_path),
                "sha256": file_hash,
                "source_url": url,
                "method": "direct_http",
            }
            success_count += 1
        else:
            # Strategy 2: Try astroquery
            output_path_csv = catalog_dir / f"{table_name}.csv"
            if download_via_astroquery(
                catalog_info["vizier_id"], table_name, output_path_csv
            ):
                file_hash = sha256_file(output_path_csv)
                provenance["files"][table_name] = {
                    "path": str(output_path_csv),
                    "sha256": file_hash,
                    "source": f"astroquery VizieR {catalog_info['vizier_id']}",
                    "method": "astroquery",
                }
                success_count += 1

    # Strategy 3: If individual tables failed, try the tar.gz
    if success_count == 0 and "tar_url" in catalog_info:
        tar_path = catalog_dir / "catalog.tar.gz"
        print(f"\n--- Fallback: downloading full catalog archive ---")
        if download_via_requests(catalog_info["tar_url"], tar_path):
            file_hash = sha256_file(tar_path)
            provenance["files"]["archive"] = {
                "path": str(tar_path),
                "sha256": file_hash,
                "source_url": catalog_info["tar_url"],
                "method": "tar_archive",
            }
            success_count += 1

    # Save provenance
    prov_path = catalog_dir / "PROVENANCE.json"
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"\nProvenance saved: {prov_path}")

    return success_count > 0


def download_sdss_galaxy_catalog():
    """
    Download SDSS DR12 spectroscopic galaxy catalog (subset for filament regions).

    Full SDSS DR12 is very large. We download the CasJobs SQL query results
    or use a pre-defined subset covering the filament footprint.
    """
    galaxy_dir = OUTPUT_DIR / "sdss_dr12_galaxies"
    galaxy_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("SDSS DR12 Spectroscopic Galaxies (for baryonic input)")
    print(f"{'='*60}")

    # Write a SQL query template for SDSS CasJobs
    query = """
-- SDSS DR12 spectroscopic galaxy query for filament validation
-- Execute this query at https://skyserver.sdss.org/casjobs/
-- Or via astroquery.sdss

SELECT
    s.specObjID,
    s.ra,
    s.dec,
    s.z AS redshift,
    s.zErr AS redshift_err,
    s.class,
    s.subClass,
    p.petroMag_r,
    p.petroMag_g,
    p.modelMag_r,
    p.extinction_r,
    p.petroR50_r,
    p.petroR90_r
FROM
    SpecObj AS s
    JOIN PhotoObj AS p ON s.bestObjID = p.objID
WHERE
    s.class = 'GALAXY'
    AND s.zWarning = 0
    AND s.z BETWEEN 0.05 AND 0.40
    AND p.petroMag_r BETWEEN 14.5 AND 17.77
    AND p.type = 3
ORDER BY s.ra
"""

    query_path = galaxy_dir / "sdss_dr12_galaxy_query.sql"
    with open(query_path, "w") as f:
        f.write(query)

    print(f"  SQL query template saved: {query_path}")
    print(f"  Execute at: https://skyserver.sdss.org/casjobs/")
    print(f"  Or use: astroquery.sdss.SDSS.query_sql()")

    # Also write a Python script to execute via astroquery
    script = '''#!/usr/bin/env python3
"""Execute SDSS DR12 galaxy query via astroquery."""
from astroquery.sdss import SDSS
from astropy.table import Table
import json, hashlib, os
from datetime import datetime

query = open("sdss_dr12_galaxy_query.sql").read()

print("Querying SDSS DR12 (this may take several minutes)...")
result = SDSS.query_sql(query, data_release=12)

if result is not None:
    output = "sdss_dr12_galaxies.fits"
    result.write(output, format="fits", overwrite=True)
    print(f"Saved {len(result)} galaxies to {output}")

    # Provenance
    h = hashlib.sha256()
    with open(output, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    prov = {
        "source": "SDSS DR12 CasJobs via astroquery",
        "n_galaxies": len(result),
        "date": datetime.utcnow().isoformat() + "Z",
        "sha256": h.hexdigest(),
    }
    with open("PROVENANCE.json", "w") as f:
        json.dump(prov, f, indent=2)
else:
    print("Query returned no results!")
'''

    script_path = galaxy_dir / "execute_query.py"
    with open(script_path, "w") as f:
        f.write(script)

    print(f"  Execution script saved: {script_path}")
    return True


def download_redmapper_clusters():
    """Download redMaPPer cluster catalog (SDSS DR8-based, public)."""
    cluster_dir = OUTPUT_DIR / "redmapper_clusters"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("redMaPPer Cluster Catalog (SDSS DR8)")
    print(f"{'='*60}")

    # redMaPPer v6.3 catalog is publicly available
    url = "https://risa.stanford.edu/redmapper/v6.3/redmapper_dr8_public_v6.3_catalog.fits.gz"
    output_path = cluster_dir / "redmapper_dr8_v6.3_catalog.fits.gz"

    if download_via_requests(url, output_path):
        provenance = {
            "source": "redMaPPer v6.3 SDSS DR8",
            "reference": "Rykoff+ 2014, ApJ 785, 104",
            "url": url,
            "date": datetime.utcnow().isoformat() + "Z",
            "sha256": sha256_file(output_path),
        }
        prov_path = cluster_dir / "PROVENANCE.json"
        with open(prov_path, "w") as f:
            json.dump(provenance, f, indent=2)
        return True
    else:
        print("  NOTE: redMaPPer catalog may need manual download from:")
        print(f"  {url}")
        return False


def main():
    print("=" * 60)
    print("SDSS Cosmic Filament Data Acquisition Pipeline")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Date: {datetime.utcnow().isoformat()}Z")
    print("=" * 60)

    results = {}

    # Download filament catalogs
    for key, info in CATALOGS.items():
        results[key] = download_catalog(key, info)

    # Download galaxy catalog (template + script)
    results["sdss_galaxies"] = download_sdss_galaxy_catalog()

    # Download cluster catalog
    results["redmapper"] = download_redmapper_clusters()

    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    for key, success in results.items():
        status = "OK" if success else "NEEDS ATTENTION"
        print(f"  {key}: {status}")

    print(f"\nAll outputs in: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Execute sdss_dr12_galaxies/execute_query.py for galaxy catalog")
    print("  2. Run download_kids_lensing.py for weak-lensing data")
    print("  3. Run build_metrology_roster.py to select train/holdout filaments")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
