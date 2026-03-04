"""
Company Comparison Page
"""
import streamlit as st
import plotly.express as px

from pages._init_session import init_session_state

init_session_state()

st.markdown('<h1 class="main-header">Company Comparison</h1>', unsafe_allow_html=True)
st.markdown("Compare up to **3 companies** side-by-side")

df = st.session_state.df
companies = df["company"].tolist()

@st.cache_data
def get_comparison_data(_df, companies_tuple):
    return _df[_df["company"].isin(companies_tuple)].copy()

col1, col2, col3 = st.columns(3)
with col1:
    company1 = st.selectbox("Company 1", companies, index=0)
with col2:
    company2 = st.selectbox("Company 2", companies, index=min(1, len(companies) - 1))
with col3:
    company3 = st.selectbox("Company 3 (Optional)", ["None"] + companies, index=0)

selected = [company1, company2]
if company3 != "None":
    selected.append(company3)

compare_df = get_comparison_data(df, tuple(selected))

if len(compare_df) >= 2:
    st.markdown("---")
    st.markdown("### 📊 Comparison Chart")

    metrics      = ["total_filings", "avg_salary", "median_salary", "sponsorship_score"]
    metric_names = ["Total Filings", "Avg Salary ($)", "Median Salary ($)", "Sponsorship Score"]

    compare_melted = compare_df.melt(
        id_vars=["company"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Value",
    )
    compare_melted["Metric"] = compare_melted["Metric"].map(dict(zip(metrics, metric_names)))

    fig = px.bar(
        compare_melted,
        x="Metric", y="Value", color="company",
        barmode="group",
        color_discrete_sequence=["#14b8a6", "#f59e0b", "#8b5cf6"],
    )
    fig.update_layout(height=450, xaxis_title="", yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Detailed Comparison")
    # Convert all values to strings to avoid Arrow type errors on transposed DataFrame
    comparison_table = compare_df[
        ["company", "state", "total_filings", "avg_salary",
         "median_salary", "sponsorship_score", "size_category"]
    ].set_index("company").T.astype(str)
    comparison_table.index = [
        "State", "Total Filings", "Avg Salary", "Median Salary",
        "Sponsorship Score", "Size Category",
    ]
    st.dataframe(comparison_table, use_container_width=True)

    st.markdown("### 🏆 Comparison Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Most Filings",   compare_df.loc[compare_df["total_filings"].idxmax(), "company"])
    with col2:
        st.metric("Highest Salary", compare_df.loc[compare_df["avg_salary"].idxmax(), "company"])
    with col3:
        st.metric("Best Score",     compare_df.loc[compare_df["sponsorship_score"].idxmax(), "company"])
else:
    st.warning("Please select at least 2 different companies to compare.")
