"""
Download Data Page
==================
UI wrapper around scripts/downloader.py.
Select fiscal years, click download, get cleaned CSVs + multi-year dataset.
"""
import sys
import threading
from pathlib import Path

import streamlit as st

# Ensure project root is on path so scripts.downloader can import src.*
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from pages._init_session import init_session_state

init_session_state()

st.markdown('<h1 class="main-header">📥 Download H-1B Data</h1>', unsafe_allow_html=True)
st.markdown("Download LCA Disclosure data from DOL for any fiscal year (2020–2025).")

st.info(
    "**What this does:** Downloads the full LCA Excel from DOL (~120–200 MB each), "
    "cleans it, and saves:\n"
    "- `data/cleaned_h1b_FY{YEAR}.csv` — per-year cleaned company data\n"
    "- `data/h1b_multiyear.csv` — combined file for LSTM training"
)

st.markdown("---")
st.markdown("### 📅 Select Fiscal Years")

SUPPORTED_YEARS = list(range(2020, 2026))

selected_years = st.multiselect(
    "Fiscal Year(s) to Download",
    options=SUPPORTED_YEARS,
    default=[2022, 2023, 2024],
    help="FY2022–FY2024 gives 3 years of data for LSTM training.",
)

col1, col2 = st.columns(2)
with col1:
    top_n = st.slider("Max Companies per Year", 10, 100, 50, 10)
with col2:
    min_filings = st.slider("Min Filings Threshold", 10, 500, 100, 10)

st.markdown("---")

# Show existing files
data_dir = _ROOT / "data"
existing = sorted(data_dir.glob("cleaned_h1b_FY*.csv"))
if existing:
    st.markdown("### 💾 Already Downloaded")
    for f in existing:
        size_mb = f.stat().st_size / 1_048_576
        st.markdown(f"- `{f.name}` ({size_mb:.1f} MB)")
    if (data_dir / "h1b_multiyear.csv").exists():
        import pandas as pd
        multi = pd.read_csv(data_dir / "h1b_multiyear.csv")
        st.success(
            f"✅ `h1b_multiyear.csv` exists — "
            f"{len(multi):,} rows, "
            f"years: {sorted(multi['fiscal_year'].unique().tolist())}"
        )
else:
    st.info("No data downloaded yet.")

st.markdown("---")

if not selected_years:
    st.warning("Select at least one fiscal year above.")
else:
    dry_run = st.checkbox("🔍 Dry Run (show URLs without downloading)", value=False)

    btn_label = "🔍 Show DOL URLs" if dry_run else f"⬇️  Download {len(selected_years)} Year(s)"

    if st.button(btn_label, type="primary", use_container_width=True):
        from scripts.downloader import download_years

        log_box    = st.empty()
        log_lines: list[str] = []

        if dry_run:
            log_lines.append("**DRY RUN — URLs that would be downloaded:**\n")
            result = download_years(selected_years, top_n=top_n, min_filings=min_filings, dry_run=True)
            from scripts.downloader import _build_url
            for y in sorted(selected_years):
                log_lines.append(f"• FY{y}: `{_build_url(y)}`")
            log_box.markdown("\n".join(log_lines))
        else:
            st.warning(
                "⏳ Download in progress — this can take **2–5 minutes per year** "
                "depending on your connection. Do not close the browser."
            )
            progress = st.progress(0)

            try:
                for i, year in enumerate(sorted(selected_years)):
                    log_lines.append(f"**Fetching FY{year}…**")
                    log_box.markdown("\n\n".join(log_lines))

                    from scripts.downloader import process_year
                    df_year = process_year(year, top_n=top_n, min_filings=min_filings)
                    log_lines.append(f"✅ FY{year} done — {len(df_year)} companies")
                    progress.progress((i + 1) / len(selected_years))
                    log_box.markdown("\n\n".join(log_lines))

                # Rebuild multi-year CSV
                import pandas as pd
                frames = [
                    pd.read_csv(data_dir / f"cleaned_h1b_FY{y}.csv")
                    for y in sorted(selected_years)
                    if (data_dir / f"cleaned_h1b_FY{y}.csv").exists()
                ]
                if frames:
                    combined = pd.concat(frames, ignore_index=True)
                    combined.to_csv(data_dir / "h1b_multiyear.csv", index=False)
                    log_lines.append(
                        f"\n💾 **`h1b_multiyear.csv` saved** — "
                        f"{len(combined):,} rows across {combined['fiscal_year'].nunique()} years."
                    )
                    log_box.markdown("\n\n".join(log_lines))

                st.success("✅ All years downloaded successfully! Refresh the page to see the files.")

            except Exception as e:
                st.error(f"❌ Download failed: {e}")

st.markdown("---")
st.markdown("### 🧠 Train LSTM After Downloading")
st.markdown(
    "Once you have `data/h1b_multiyear.csv` (≥2 years), run the LSTM trainer from your terminal:\n"
    "```bash\npython -m src.models.lstm_trainer\n```\n"
    "This saves `data/lstm_predictions.csv` which the **Predictor** page uses for trend arrows."
)
