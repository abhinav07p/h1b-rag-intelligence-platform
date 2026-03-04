"""
H-1B Multi-Year DOL Data Downloader
=====================================
Downloads LCA Disclosure Excel files from the US DOL website for one or
more fiscal years, cleans each file, and produces:
  - data/cleaned_h1b_FY{YEAR}.csv   — per-year cleaned company data
  - data/h1b_multiyear.csv           — combined file with fiscal_year column

DOL URL pattern (confirmed FY2022–FY2025):
  https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY{YEAR}_Q4.xlsx

Usage (CLI):
    python scripts/downloader.py --years 2022 2023 2024
    python scripts/downloader.py --years 2024 --dry-run

Usage (Python):
    from scripts.downloader import download_years
    combined_df = download_years([2022, 2023, 2024])
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT      = Path(__file__).parent.parent
_DATA_DIR  = _ROOT / "data"
_DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# URL patterns — DOL publishes Q4 as the full-year roll-up
# ---------------------------------------------------------------------------
_URL_TEMPLATE = (
    "https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/"
    "LCA_Disclosure_Data_FY{year}_Q4.xlsx"
)

# Some earlier years use a slightly different filename pattern
_URL_FALLBACKS: dict[int, list[str]] = {
    2022: [
        "https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2022_Q4.xlsx",
    ],
    2023: [
        "https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/LCA_Disclosure_Data_FY2023_Q4.xlsx",
    ],
}

SUPPORTED_YEARS = list(range(2020, 2027))


# ---------------------------------------------------------------------------
# Downloader helpers
# ---------------------------------------------------------------------------

def _build_url(year: int) -> str:
    """Return the primary DOL download URL for a fiscal year."""
    return _URL_TEMPLATE.format(year=year)


def _download_excel(year: int, dest: Path, dry_run: bool = False) -> Optional[Path]:
    """
    Download the LCA Excel file for *year* to *dest*.

    Args:
        year:    Fiscal year (e.g. 2024).
        dest:    Destination file path.
        dry_run: If True, only print the URL without downloading.

    Returns:
        Path to the downloaded file, or None if dry_run.

    Raises:
        RuntimeError: If the download fails for all candidate URLs.
    """
    urls = _URL_FALLBACKS.get(year, []) + [_build_url(year)]

    for url in urls:
        if dry_run:
            print(f"  [DRY RUN] Would download: {url}")
            return None

        print(f"  ⬇️  Downloading FY{year} from:\n     {url}")
        try:
            resp = requests.get(url, stream=True, timeout=120)
            if resp.status_code == 200:
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            print(f"\r     {pct:.1f}% ({downloaded // 1_048_576} MB / {total // 1_048_576} MB)  ", end="")
                print()
                print(f"  ✅ Saved to {dest} ({dest.stat().st_size // 1_048_576} MB)")
                return dest
            else:
                print(f"  ⚠️  HTTP {resp.status_code} for {url}")
        except requests.RequestException as e:
            print(f"  ⚠️  Request failed: {e}")

    raise RuntimeError(
        f"Failed to download FY{year} LCA data from all candidate URLs:\n" +
        "\n".join(f"  - {u}" for u in urls)
    )


# ---------------------------------------------------------------------------
# Per-year processing
# ---------------------------------------------------------------------------

def process_year(year: int, top_n: int = 50, min_filings: int = 100) -> pd.DataFrame:
    """
    Download + clean LCA data for a single fiscal year.

    Args:
        year:        Fiscal year.
        top_n:       Maximum number of companies to keep.
        min_filings: Minimum certified filings threshold.

    Returns:
        Cleaned DataFrame with a ``fiscal_year`` column added.
    """
    # Import here to avoid circular issues when running as __main__
    sys.path.insert(0, str(_ROOT))
    from src.data.pipeline import process_uploaded_file

    raw_path = _DATA_DIR / f"LCA_raw_FY{year}.xlsx"

    if raw_path.exists():
        print(f"  ♻️  Found cached raw file: {raw_path}")
    else:
        _download_excel(year, raw_path)

    print(f"  🧹 Cleaning FY{year} data…")
    cleaned_df, stats = process_uploaded_file(
        raw_path,
        top_n=top_n,
        min_filings=min_filings,
        fiscal_year=year,
    )

    out_path = _DATA_DIR / f"cleaned_h1b_FY{year}.csv"
    cleaned_df.to_csv(out_path, index=False)
    print(f"  💾 Saved → {out_path}  ({len(cleaned_df)} companies)")

    for step in stats.get("steps", []):
        print(f"     • {step}")

    return cleaned_df


# ---------------------------------------------------------------------------
# Multi-year orchestrator
# ---------------------------------------------------------------------------

def download_years(
    years: List[int],
    top_n: int = 50,
    min_filings: int = 100,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Download, clean, and combine LCA data for multiple fiscal years.

    Args:
        years:       List of fiscal years to process.
        top_n:       Max companies per year.
        min_filings: Minimum filing threshold.
        dry_run:     If True, only print URLs without downloading.

    Returns:
        Combined multi-year DataFrame saved to data/h1b_multiyear.csv.
    """
    if dry_run:
        print("\n🔍 DRY RUN — URLs that would be downloaded:")
        for year in sorted(years):
            _download_excel(year, Path("/dev/null"), dry_run=True)
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    for year in sorted(years):
        print(f"\n{'='*60}")
        print(f"📅 Processing FY{year}")
        print(f"{'='*60}")
        try:
            df = process_year(year, top_n=top_n, min_filings=min_filings)
            frames.append(df)
        except Exception as e:
            print(f"  ❌ FY{year} failed: {e}")
            continue

    if not frames:
        print("\n❌ No years processed successfully.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    out_path = _DATA_DIR / "h1b_multiyear.csv"
    combined.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"✅ Multi-year dataset saved → {out_path}")
    print(f"   Years: {sorted(combined['fiscal_year'].unique().tolist())}")
    print(f"   Rows:  {len(combined):,}")
    print(f"{'='*60}")

    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and process H-1B LCA Disclosure data from DOL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Supported fiscal years: {SUPPORTED_YEARS[0]}–{SUPPORTED_YEARS[-1]}",
    )
    parser.add_argument(
        "--years", nargs="+", type=int, required=True,
        metavar="YEAR",
        help="Fiscal year(s) to download (e.g. --years 2022 2023 2024)",
    )
    parser.add_argument(
        "--top-n", type=int, default=50,
        help="Max companies per year (default: 50)",
    )
    parser.add_argument(
        "--min-filings", type=int, default=100,
        help="Minimum certified filings threshold (default: 100)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print DOL URLs without downloading",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    invalid = [y for y in args.years if y not in SUPPORTED_YEARS]
    if invalid:
        print(f"❌ Unsupported year(s): {invalid}. Supported: {SUPPORTED_YEARS}")
        sys.exit(1)

    download_years(
        years=args.years,
        top_n=args.top_n,
        min_filings=args.min_filings,
        dry_run=args.dry_run,
    )
