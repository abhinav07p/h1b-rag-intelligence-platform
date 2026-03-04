"""
Interactive Data Pipeline Page
"""
import streamlit as st
from src.data.pipeline import process_uploaded_file
from pages._init_session import init_session_state

init_session_state()

st.markdown('<h1 class="main-header">Interactive Data Pipeline</h1>', unsafe_allow_html=True)
st.markdown("Upload raw USCIS data → Clean → Load into app")

df = st.session_state.df

st.markdown("### Step 1: Upload USCIS Excel File")
st.markdown("Download from: [DOL LCA Disclosure Data](https://www.dol.gov/agencies/eta/foreign-labor/performance)")

uploaded_file = st.file_uploader(
    "Upload LCA Disclosure Excel File",
    type=["xlsx", "xls"],
    help="Upload the USCIS LCA disclosure Excel file (e.g., LCA_Disclosure_Data_FY2024_Q4.xlsx)",
)

if uploaded_file:
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    st.markdown("### Step 2: Configure Cleaning Parameters")

    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider("Number of Top Companies to Keep", 10, 100, 50, 10)
    with col2:
        min_filings = st.slider("Minimum Filings Threshold", 10, 500, 100, 10)

    st.markdown("### Step 3: Process Data")

    if st.button("🚀 Clean & Load Data", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text  = st.empty()

        def update_progress(progress, message):
            progress_bar.progress(progress)
            status_text.text(message)

        try:
            cleaned_df, stats = process_uploaded_file(
                uploaded_file,
                top_n=top_n,
                min_filings=min_filings,
                progress_callback=update_progress,
            )
            st.session_state.df = cleaned_df
            st.success("✅ Data processed successfully!")

            st.markdown("### 📊 Processing Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Original Rows",   f"{stats['original_rows']:,}")
            col2.metric("After Filtering", f"{stats['filtered_rows']:,}")
            col3.metric("Final Companies", stats["final_companies"])

            st.markdown("**Processing Steps:**")
            for step in stats["steps"]:
                st.markdown(f"- {step}")

            st.markdown("### 📋 Data Preview")
            st.dataframe(cleaned_df.head(10), use_container_width=True)

            csv = cleaned_df.to_csv(index=False)
            st.download_button(
                "📥 Download Cleaned CSV",
                csv,
                "cleaned_h1b_data.csv",
                "text/csv",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
else:
    st.info("👆 Upload a file to get started")
    st.markdown("### 📊 Currently Loaded Data")
    st.markdown(f"- **Companies:** {len(df)}")
    st.markdown(f"- **Total Filings:** {df['total_filings'].sum():,}")
    st.markdown(f"- **Avg Salary:** ${df['avg_salary'].mean():,.0f}")
