"""
H-1B Data Pipeline
==================
Processes raw USCIS LCA Disclosure Excel files uploaded via the Streamlit UI.

Flow:
    Upload XLSX → filter H-1B Certified → clean names → normalise salaries
    → aggregate by company → compute sponsorship_score → return cleaned DataFrame
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import clean_employer_name, convert_to_annual_salary


# ---------------------------------------------------------------------------
# Column discovery helpers
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Return the first candidate column name that exists in *df*, else None."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _find_col_containing(df: pd.DataFrame, *keywords: str) -> Optional[str]:
    """Return the first column whose upper-cased name contains ALL keywords."""
    for col in df.columns:
        upper = col.upper()
        if all(kw in upper for kw in keywords):
            return col
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_uploaded_file(
    uploaded_file,
    top_n: int = 50,
    min_filings: int = 100,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    fiscal_year: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    End-to-end ETL for a raw USCIS LCA Disclosure Excel file.

    Args:
        uploaded_file:     File-like object (from st.file_uploader or a path).
        top_n:             Maximum number of companies to return.
        min_filings:       Minimum certified H-1B filings threshold.
        progress_callback: Optional callable(fraction, message) for UI progress.

    Returns:
        Tuple of:
            - cleaned DataFrame (company, state, total_filings, avg_salary,
              median_salary, sponsorship_score, size_category)
            - stats dict with processing metadata
    """
    def _progress(pct: float, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)

    stats: dict = {"steps": []}

    # ------------------------------------------------------------------
    # Step 1 — Load
    # ------------------------------------------------------------------
    _progress(0.10, "Loading Excel file…")
    df = pd.read_excel(uploaded_file)
    stats["original_rows"] = len(df)
    stats["steps"].append(f"Loaded {len(df):,} rows")

    # ------------------------------------------------------------------
    # Step 2 — Discover columns
    # ------------------------------------------------------------------
    _progress(0.20, "Identifying columns…")
    employer_col  = _find_col(df, "EMPLOYER_NAME", "EMPLOYER_BUSINESS_NAME", "EMPLOYER")
    status_col    = _find_col(df, "CASE_STATUS", "STATUS")
    wage_col      = _find_col(df, "WAGE_RATE_OF_PAY_FROM", "WAGE_RATE", "PREVAILING_WAGE")
    wage_unit_col = _find_col(df, "WAGE_UNIT_OF_PAY", "WAGE_RATE_UNIT", "PW_UNIT_OF_PAY")
    state_col     = _find_col(df, "EMPLOYER_STATE", "EMPLOYER_PROVINCE", "WORKSITE_STATE")
    visa_col      = _find_col(df, "VISA_CLASS", "PROGRAM")
    job_col       = _find_col(df, "JOB_TITLE", "POSITION_TITLE", "SOC_TITLE")

    if employer_col is None:
        raise ValueError("Could not locate employer name column in the uploaded file.")

    stats["steps"].append(f"Found columns: employer={employer_col}, status={status_col}")

    # ------------------------------------------------------------------
    # Step 3 — Filter: H-1B only, Certified only
    # ------------------------------------------------------------------
    _progress(0.30, "Filtering H-1B Certified cases…")
    if visa_col:
        df = df[df[visa_col].astype(str).str.contains("H-1B", case=False, na=False)]
    if status_col:
        certified = df[status_col].astype(str).str.contains("CERTIFIED", case=False, na=False)
        withdrawn = df[status_col].astype(str).str.contains("WITHDRAWN", case=False, na=False)
        df = df[certified & ~withdrawn]

    stats["filtered_rows"] = len(df)
    stats["steps"].append(f"After filters: {len(df):,} rows")

    # ------------------------------------------------------------------
    # Step 4 — Clean employer names
    # ------------------------------------------------------------------
    _progress(0.50, "Cleaning employer names…")
    df = df.copy()
    df["EMPLOYER_CLEAN"] = df[employer_col].apply(clean_employer_name)

    # ------------------------------------------------------------------
    # Step 5 — Normalise wages to annual salary
    # ------------------------------------------------------------------
    _progress(0.60, "Processing salaries…")
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

    # ------------------------------------------------------------------
    # Step 6 — Aggregate by company
    # ------------------------------------------------------------------
    _progress(0.70, "Aggregating by company…")
    agg_dict: dict = {"ANNUAL_SALARY": ["count", "mean", "median"]}
    if state_col:
        agg_dict[state_col] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"

    company_stats = df.groupby("EMPLOYER_CLEAN").agg(agg_dict).reset_index()

    if state_col:
        company_stats.columns = ["company", "total_filings", "avg_salary", "median_salary", "state"]
    else:
        company_stats.columns = ["company", "total_filings", "avg_salary", "median_salary"]
        company_stats["state"] = "Unknown"

    # ------------------------------------------------------------------
    # Step 7 — Filter, sort, select top N
    # ------------------------------------------------------------------
    _progress(0.80, f"Selecting top {top_n} companies…")
    company_stats = (
        company_stats[company_stats["total_filings"] >= min_filings]
        .sort_values("total_filings", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    stats["final_companies"] = len(company_stats)

    # ------------------------------------------------------------------
    # Step 8 — Derived features: sponsorship_score, size_category
    # ------------------------------------------------------------------
    _progress(0.90, "Calculating sponsorship scores…")
    max_filings = company_stats["total_filings"].max()
    company_stats["volume_score"] = (
        (company_stats["total_filings"] / max_filings * 40).clip(0, 40)
    )

    sal_max = company_stats["avg_salary"].max()
    sal_min = company_stats["avg_salary"].min()
    sal_range = sal_max - sal_min
    company_stats["salary_score"] = (
        ((company_stats["avg_salary"] - sal_min) / sal_range * 30).clip(0, 30)
        if sal_range > 0
        else 15
    )

    company_stats["sponsorship_score"] = (
        company_stats["volume_score"] + company_stats["salary_score"] + 30
    ).round(1)

    company_stats["size_category"] = pd.cut(
        company_stats["total_filings"],
        bins=[0, 500, 2_000, 10_000, float("inf")],
        labels=["Small", "Medium", "Large", "Enterprise"],
    )

    company_stats["avg_salary"]    = company_stats["avg_salary"].round(0).astype(int)
    company_stats["median_salary"] = company_stats["median_salary"].round(0).astype(int)

    final_cols = ["company", "state", "total_filings", "avg_salary",
                  "median_salary", "sponsorship_score", "size_category"]
    company_stats = company_stats[final_cols]

    # Stamp fiscal year if provided (used by multi-year downloader / LSTM)
    if fiscal_year is not None:
        company_stats.insert(0, "fiscal_year", int(fiscal_year))

    _progress(1.00, "Done!")
    stats["total_filings"] = int(company_stats["total_filings"].sum())
    stats["avg_salary"]    = float(company_stats["avg_salary"].mean())

    return company_stats, stats

