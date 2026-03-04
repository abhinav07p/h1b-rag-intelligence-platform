"""
Dashboard Page
"""
import streamlit as st
import plotly.express as px

from src.data.loader import get_company_summary
from pages._init_session import init_session_state

init_session_state()

st.markdown('<h1 class="main-header">H-1B Sponsorship Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**Real data from USCIS LCA Disclosure Files (FY2024)**")

df      = st.session_state.df
summary = get_company_summary(df)

# Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Companies",  len(df))
c2.metric("Total Filings",    f"{df['total_filings'].sum():,}")
c3.metric("Avg Salary",       f"${df['avg_salary'].mean():,.0f}")
c4.metric("Top Sponsor",      summary["most_filings_company"])

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Top 10 H-1B Sponsors")
    top10 = df.nlargest(10, "total_filings")
    fig = px.bar(top10, x="total_filings", y="company", orientation="h",
                 color="avg_salary", color_continuous_scale="Teal")
    fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Highest Paying Companies")
    top_salary = df.nlargest(10, "avg_salary")
    fig = px.bar(top_salary, x="avg_salary", y="company", orientation="h",
                 color="total_filings", color_continuous_scale="Oranges")
    fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Filings by State")
    state_filings = df.groupby("state")["total_filings"].sum().nlargest(10)
    fig = px.pie(values=state_filings.values, names=state_filings.index)
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Company Size Distribution")
    size_dist = df["size_category"].value_counts()
    fig = px.pie(values=size_dist.values, names=size_dist.index,
                 color_discrete_sequence=px.colors.sequential.Teal)
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### 📋 All Companies")
st.dataframe(df.style.format({
    "total_filings":    "{:,}",
    "avg_salary":       "${:,.0f}",
    "median_salary":    "${:,.0f}",
    "sponsorship_score":"{:.1f}",
}), use_container_width=True, height=400)
