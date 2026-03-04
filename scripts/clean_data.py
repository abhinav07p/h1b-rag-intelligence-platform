"""
H-1B Data Cleaning Script
=========================
Processes one or more raw USCIS LCA Disclosure Excel files into the
cleaned CSV format expected by the training pipeline and Streamlit app.

Supports:
  - Single-year cleaning   → produces cleaned_h1b_data.csv
  - Multi-year cleaning    → produces one CSV per year + a combined CSV
                             (required for LSTM sequence modelling)

Usage:
    python scripts/clean_data.py                        # auto-find files
    python scripts/clean_data.py --years 2022 2023 2024
    python scripts/clean_data.py --input notebooks/LCA_Disclosure_Data_FY2024_Q4.xlsx
    python scripts/clean_data.py --years 2022 2023 2024 --min-filings 50 --top-n 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to sys.path so we can import shared utilities
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.utils import clean_employer_name, convert_to_annual_salary

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NOTEBOOKS_DIR = _PROJECT_ROOT / "notebooks"
_DATA_DIR      = _PROJECT_ROOT / "data"


# ---------------------------------------------------------------------------
# Column discovery
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, *keywords_groups: tuple[str, ...]) -> str | None:
    """Return the first column whose upper name contains all keywords in any group."""
    for keywords in keywords_groups:
        for col in df.columns:
            upper = col.upper()
            if all(k in upper for k in keywords):
                return col
    return None


# ---------------------------------------------------------------------------
# Core cleaning
# ---------------------------------------------------------------------------

def clean_single_file(
    filepath: Path,
    min_filings: int = 50,
    top_n: int = 200,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Clean a single USCIS LCA Disclosure Excel file.

    Steps:
        1. Load Excel file.
        2. Filter H-1B Certified rows (exclude WITHDRAWN).
        3. Standardise employer names.
        4. Convert wages to annual salary.
        5. Filter salary range $30K – $500K.
        6. Aggregate by employer.
        7. Compute sponsorship_score and size_category.

    Args:
        filepath:     Path to the raw Excel file.
        min_filings:  Minimum certified filings to include a company.
        top_n:        Maximum companies in the output.
        verbose:      Print progress messages.

    Returns:
        Cleaned DataFrame with columns: company, state, total_filings,
        avg_salary, median_salary, sponsorship_score, size_category.
    """

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    _log(f"\n📂 Loading {filepath.name}  ({filepath.stat().st_size / 1e6:.1f} MB)…")
    df = pd.read_excel(filepath)
    _log(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Column detection
    employer_col  = _find_col(df, ("EMPLOYER", "NAME"), ("EMPLOYER", "BUSINESS"))
    status_col    = _find_col(df, ("STATUS",), ("CASE", "STATUS"))
    wage_col      = _find_col(df, ("WAGE", "FROM"), ("WAGE", "RATE"), ("PREVAILING", "WAGE"))
    wage_unit_col = _find_col(df, ("WAGE", "UNIT"), ("PW", "UNIT"))
    state_col     = _find_col(df, ("EMPLOYER", "STATE"), ("WORKSITE", "STATE"))
    visa_col      = _find_col(df, ("VISA",), ("CLASS",))

    if employer_col is None:
        raise ValueError(f"Cannot find employer name column in {filepath.name}")

    # Filter: H-1B only
    if visa_col:
        df = df[df[visa_col].astype(str).str.contains("H-1B", case=False, na=False)]
        _log(f"   After H-1B filter: {len(df):,} rows")

    # Filter: Certified only (exclude CERTIFIED-WITHDRAWN)
    if status_col:
        certified = df[status_col].astype(str).str.contains("CERTIFIED", case=False, na=False)
        withdrawn = df[status_col].astype(str).str.contains("WITHDRAWN", case=False, na=False)
        df = df[certified & ~withdrawn]
        _log(f"   After Certified filter: {len(df):,} rows")

    df = df.copy()

    # Clean names
    df["EMPLOYER_CLEAN"] = df[employer_col].apply(clean_employer_name)

    # Normalise wages
    if wage_col and wage_unit_col:
        df["ANNUAL_SALARY"] = df.apply(
            lambda row: convert_to_annual_salary(row[wage_col], row[wage_unit_col]),
            axis=1,
        )
    elif wage_col:
        df["ANNUAL_SALARY"] = pd.to_numeric(df[wage_col], errors="coerce")
    else:
        df["ANNUAL_SALARY"] = np.nan

    df = df[(df["ANNUAL_SALARY"] >= 30_000) & (df["ANNUAL_SALARY"] <= 500_000)]
    _log(f"   After salary filter ($30K–$500K): {len(df):,} rows")

    # Get state
    df["STATE"] = df[state_col].fillna("Unknown") if state_col else "Unknown"

    # Aggregate
    company_stats = df.groupby("EMPLOYER_CLEAN").agg(
        total_filings=("ANNUAL_SALARY", "count"),
        avg_salary=("ANNUAL_SALARY", "mean"),
        median_salary=("ANNUAL_SALARY", "median"),
        min_salary=("ANNUAL_SALARY", "min"),
        max_salary=("ANNUAL_SALARY", "max"),
        salary_std=("ANNUAL_SALARY", "std"),
        state=("STATE", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"),
    ).reset_index().rename(columns={"EMPLOYER_CLEAN": "company"})

    # Filter and sort
    company_stats = (
        company_stats[company_stats["total_filings"] >= min_filings]
        .sort_values("total_filings", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    _log(f"   Companies with ≥{min_filings} filings: {len(company_stats):,}")

    # Sponsorship score (0–100)
    max_fil = company_stats["total_filings"].max()
    sal_max = company_stats["avg_salary"].max()
    sal_min = company_stats["avg_salary"].min()
    sal_range = sal_max - sal_min

    company_stats["volume_score"] = (company_stats["total_filings"] / max_fil * 50).clip(0, 50)
    company_stats["salary_score"] = (
        ((company_stats["avg_salary"] - sal_min) / sal_range * 30).clip(0, 30)
        if sal_range > 0 else 15
    )
    max_std = company_stats["salary_std"].max()
    company_stats["consistency_score"] = (
        ((1 - company_stats["salary_std"].fillna(max_std) / max_std) * 20).clip(0, 20)
        if max_std > 0 else 10
    )
    company_stats["sponsorship_score"] = (
        company_stats["volume_score"]
        + company_stats["salary_score"]
        + company_stats["consistency_score"]
    ).round(1)

    company_stats["size_category"] = pd.cut(
        company_stats["total_filings"],
        bins=[0, 10, 50, 200, 1_000, float("inf")],
        labels=["Very Small", "Small", "Medium", "Large", "Enterprise"],
    )

    # Round numeric columns
    for col in ("avg_salary", "median_salary", "min_salary", "max_salary"):
        company_stats[col] = company_stats[col].round(0).astype(int)
    company_stats["salary_std"] = company_stats["salary_std"].fillna(0).round(0).astype(int)

    final_cols = [
        "company", "state", "total_filings",
        "avg_salary", "median_salary", "min_salary", "max_salary", "salary_std",
        "sponsorship_score", "size_category",
    ]
    return company_stats[final_cols]


# ---------------------------------------------------------------------------
# Multi-year cleaning (for LSTM pipeline)
# ---------------------------------------------------------------------------

def clean_multi_year(
    years: list[int],
    in_dir: Path = _NOTEBOOKS_DIR,
    out_dir: Path = _DATA_DIR,
    **kwargs,
) -> dict[int, pd.DataFrame]:
    """
    Clean one file per fiscal year and assemble a combined multi-year CSV.

    The combined CSV adds a ``fiscal_year`` column and is used by the LSTM
    training pipeline in ``notebooks/H1B_Model_Training.ipynb`` to build
    per-company time sequences.

    Args:
        years:   Fiscal years to process (e.g. [2022, 2023, 2024]).
        in_dir:  Directory containing raw Excel files.
        out_dir: Directory to write cleaned CSVs.
        **kwargs: Passed to :func:`clean_single_file`.

    Returns:
        Dict mapping year → cleaned DataFrame.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    year_dfs: dict[int, pd.DataFrame] = {}

    for year in sorted(years):
        pattern  = f"LCA_Disclosure_Data_FY{year}_Q4.xlsx"
        filepath = in_dir / pattern

        if not filepath.exists():
            print(f"⚠️  FY{year} file not found at {filepath}. "
                  f"Run scripts/download_data.py --years {year} first.")
            continue

        t0 = time.time()
        df = clean_single_file(filepath, **kwargs)
        df["fiscal_year"] = year
        year_dfs[year] = df

        out_path = out_dir / f"cleaned_h1b_FY{year}.csv"
        df.to_csv(out_path, index=False)
        print(f"   💾 Saved → {out_path}  ({len(df):,} companies, {time.time()-t0:.1f}s)")

    if len(year_dfs) > 1:
        combined = pd.concat(year_dfs.values(), ignore_index=True)
        combined_path = out_dir / "cleaned_h1b_multiyear.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n💾 Combined ({len(combined):,} rows) → {combined_path}")

    # Also overwrite the default single-year file with the most recent year
    if year_dfs:
        latest = year_dfs[max(year_dfs.keys())]
        default_path = out_dir / "cleaned_h1b_data.csv"
        latest.to_csv(default_path, index=False)
        print(f"💾 Default app file updated → {default_path}")

    return year_dfs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean raw USCIS LCA Disclosure Excel files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/clean_data.py\n"
            "  python scripts/clean_data.py --years 2022 2023 2024\n"
            "  python scripts/clean_data.py --input notebooks/LCA_Disclosure_Data_FY2024_Q4.xlsx\n"
            "  python scripts/clean_data.py --years 2022 2023 2024 --min-filings 50 --top-n 100\n"
        ),
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=None,
        help="Fiscal years to process. Uses auto-detection if not specified.",
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Path to a single input Excel file.",
    )
    parser.add_argument(
        "--out", type=Path, default=_DATA_DIR,
        help=f"Output directory (default: {_DATA_DIR})",
    )
    parser.add_argument(
        "--min-filings", type=int, default=50,
        help="Minimum certified filings to include a company (default: 50)",
    )
    parser.add_argument(
        "--top-n", type=int, default=200,
        help="Maximum companies per year in output (default: 200)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    kwargs = dict(min_filings=args.min_filings, top_n=args.top_n)

    if args.input:
        # Single-file mode
        df = clean_single_file(args.input, **kwargs)
        out = args.out / "cleaned_h1b_data.csv"
        args.out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\n✅ Saved {len(df):,} companies → {out}")

    elif args.years:
        # Multi-year mode from --years flag
        clean_multi_year(args.years, out_dir=args.out, **kwargs)

    else:
        # Auto-detect: find all LCA Excel files in notebooks/
        found = sorted(_NOTEBOOKS_DIR.glob("LCA_Disclosure_Data_FY*_Q4.xlsx"))
        if not found:
            print("❌ No LCA Excel files found in notebooks/.")
            print("   Run: python scripts/download_data.py")
            sys.exit(1)

        years = []
        for f in found:
            try:
                year = int(f.stem.split("FY")[1].split("_")[0])
                years.append(year)
            except (IndexError, ValueError):
                pass

        print(f"📂 Auto-detected FY files: {years}")
        clean_multi_year(years, out_dir=args.out, **kwargs)
