"""
Sponsorship Predictor Page
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.models.predictor import H1BSponsorshipPredictor
from pages._init_session import init_session_state

init_session_state()

st.markdown('<h1 class="main-header">Sponsorship Predictor</h1>', unsafe_allow_html=True)
st.markdown("Predict H-1B sponsorship likelihood based on your profile")

st.markdown("### 📝 Enter Your Profile")

col1, col2 = st.columns(2)

with col1:
    job_role = st.selectbox(
        "Job Role Category",
        ["Software Engineer", "Data Scientist / Analyst", "Manager / Lead",
         "Consultant", "Research Scientist", "Other"],
    )
    salary = st.slider("Expected Salary ($)", 60000, 250000, 120000, 5000, format="$%d")
    state  = st.selectbox("Target State", ["CA", "WA", "NY", "TX", "NJ", "MA", "IL", "Other"])

with col2:
    company_name = st.text_input(
        "Target Company Name (optional — for LSTM trend)",
        placeholder="e.g. AMAZON",
        help="Enter a company name to show its LSTM-predicted sponsorship trend if available.",
    )
    company_size_raw = st.selectbox(
        "Target Company Size",
        ["Enterprise (2000+ filings)", "Large (500-2000 filings)",
         "Medium (100-500 filings)", "Small (<100 filings)"],
    )
    company_size = company_size_raw.split(" (")[0]
    education = st.selectbox("Education Level", ["Bachelors", "Masters", "PhD"])

st.markdown("---")

if st.button("🎯 Predict Sponsorship Likelihood", type="primary", use_container_width=True):
    predictor = H1BSponsorshipPredictor()
    result    = predictor.predict(job_role, salary, state, company_size, education)

    # LSTM trend annotation
    lstm_trend = None
    if company_name.strip():
        lstm_trend = predictor.get_lstm_trend(company_name.strip())

    st.markdown("### 🎯 Prediction Result")

    col1, col2 = st.columns([1, 2])

    with col1:
        if result.likelihood == "HIGH":
            st.markdown(f'<p class="prediction-high">🟢 {result.likelihood}</p>', unsafe_allow_html=True)
        elif result.likelihood == "MEDIUM":
            st.markdown(f'<p class="prediction-medium">🟡 {result.likelihood}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="prediction-low">🔴 {result.likelihood}</p>', unsafe_allow_html=True)

        st.metric("Confidence Score", f"{result.confidence:.1f}%")
        st.caption(f"Model: `{result.model_used}`")

        if lstm_trend:
            st.markdown("---")
            st.markdown(f"**LSTM Trend for {company_name.upper()}:**")
            st.markdown(f"### {lstm_trend}")
        elif company_name.strip():
            st.caption("⚠️ No LSTM trend data for this company. Run the downloader + LSTM trainer first.")

    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result.confidence,
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#14b8a6"},
                "steps": [
                    {"range": [0, 40],  "color": "#fee2e2"},
                    {"range": [40, 70], "color": "#fef3c7"},
                    {"range": [70, 100],"color": "#d1fae5"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": result.confidence,
                },
            },
        ))
        fig.update_layout(height=250, margin=dict(t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📊 Factor Analysis")
    for factor, explanation in result.factors.items():
        st.markdown(f"""
        <div class="factor-card">
            <strong>{factor.replace('_', ' ').title()}</strong><br>
            {explanation}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 💡 Recommendations")
    for i, rec in enumerate(result.recommendations, 1):
        st.markdown(f"{i}. {rec}")

    st.markdown("### 📈 Feature Importance")
    importance = predictor.get_feature_importance()
    fig = px.bar(
        x=list(importance.values()),
        y=list(importance.keys()),
        orientation="h",
        color=list(importance.values()),
        color_continuous_scale="Teal",
    )
    fig.update_layout(height=300, xaxis_title="Importance (%)", yaxis_title="", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
