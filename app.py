"""
H-1B RAG Application
====================
AI-powered H-1B visa sponsorship intelligence platform.

INFO 7390 Final Project
Author: Abhinav Kumar Piyush

Features:
- Dashboard with real USCIS data
- Company Comparison (up to 3 companies)
- Sponsorship Prediction Model
- Interactive Data Pipeline
- RAG Chatbot with Pinecone + GPT-4o
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
sys.path.append('src')

from data_loader import load_h1b_data, get_company_summary
from data_pipeline import process_uploaded_file
from prediction_model import H1BSponsorshipPredictor
from vector_store import H1BVectorStore, MockVectorStore
from rag_agent import H1BRAGAgent, MockRAGAgent

@st.cache_data
def get_comparison_data(df, companies):
    return df[df['company'].isin(companies)].copy()

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="H-1B Sponsorship Intelligence",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #14b8a6, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.prediction-high { color: #10b981; font-size: 2rem; font-weight: bold; }
.prediction-medium { color: #f59e0b; font-size: 2rem; font-weight: bold; }
.prediction-low { color: #ef4444; font-size: 2rem; font-weight: bold; }
.factor-card {
    background: #1e293b;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 3px solid #14b8a6;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================
if 'df' not in st.session_state:
    st.session_state.df = load_h1b_data()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'indexed' not in st.session_state:
    st.session_state.indexed = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

df = st.session_state.df
summary = get_company_summary(df)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## 🎯 H-1B Intelligence")
    st.markdown("*Real 2024 USCIS Data*")
    st.markdown("---")
    
    # API Keys
    st.markdown("### 🔑 API Keys")
    
    try:
        default_openai = st.secrets.get("OPENAI_API_KEY", "")
        default_pinecone = st.secrets.get("PINECONE_API_KEY", "")
    except:
        default_openai = ""
        default_pinecone = ""
    
    openai_key = st.text_input("OpenAI API Key", value=default_openai, type="password")
    pinecone_key = st.text_input("Pinecone API Key", value=default_pinecone, type="password")
    
    use_demo = st.checkbox("🎮 Demo Mode", value=(not openai_key or not pinecone_key))
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "📍 Navigation",
        ["📊 Dashboard", "📈 Compare Companies", "🤖 Sponsorship Predictor", 
         "⚙️ Data Pipeline", "💬 AI Advisor", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📈 Data Stats")
    st.metric("Companies", len(df))
    st.metric("Total Filings", f"{df['total_filings'].sum():,}")

# =============================================================================
# PAGE: DASHBOARD
# =============================================================================
if page == "📊 Dashboard":
    st.markdown('<h1 class="main-header">H-1B Sponsorship Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Real data from USCIS LCA Disclosure Files (FY2024)**")
    
    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Companies", len(df))
    c2.metric("Total Filings", f"{df['total_filings'].sum():,}")
    c3.metric("Avg Salary", f"${df['avg_salary'].mean():,.0f}")
    c4.metric("Top Sponsor", summary['most_filings_company'])
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top 10 H-1B Sponsors")
        top10 = df.nlargest(10, 'total_filings')
        fig = px.bar(top10, x='total_filings', y='company', orientation='h',
                     color='avg_salary', color_continuous_scale='Teal')
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'}, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Highest Paying Companies")
        top_salary = df.nlargest(10, 'avg_salary')
        fig = px.bar(top_salary, x='avg_salary', y='company', orientation='h',
                     color='total_filings', color_continuous_scale='Oranges')
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'}, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Filings by State")
        state_filings = df.groupby('state')['total_filings'].sum().nlargest(10)
        fig = px.pie(values=state_filings.values, names=state_filings.index)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Company Size Distribution")
        size_dist = df['size_category'].value_counts()
        fig = px.pie(values=size_dist.values, names=size_dist.index,
                     color_discrete_sequence=px.colors.sequential.Teal)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.markdown("### 📋 All Companies")
    st.dataframe(df.style.format({
        'total_filings': '{:,}',
        'avg_salary': '${:,.0f}',
        'median_salary': '${:,.0f}',
        'sponsorship_score': '{:.1f}'
    }), use_container_width=True, height=400)

# =============================================================================
# PAGE: COMPARE COMPANIES
# =============================================================================
elif page == "📈 Compare Companies":
    st.markdown('<h1 class="main-header">Company Comparison</h1>', unsafe_allow_html=True)
    st.markdown("Compare up to **3 companies** side-by-side")
    
    # Company selection
    companies = df['company'].tolist()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        company1 = st.selectbox("Company 1", companies, index=0)
    with col2:
        company2 = st.selectbox("Company 2", companies, index=min(1, len(companies)-1))
    with col3:
        company3 = st.selectbox("Company 3 (Optional)", ["None"] + companies, index=0)
    
    # Get selected companies data
    selected = [company1, company2]
    if company3 != "None":
        selected.append(company3)
    
    # compare_df = df[df['company'].isin(selected)].copy()
    compare_df = get_comparison_data(df, tuple(selected))
    
    if len(compare_df) >= 2:
        st.markdown("---")
        st.markdown("### 📊 Comparison Chart")
        
        # Create comparison bar chart
        metrics = ['total_filings', 'avg_salary', 'median_salary', 'sponsorship_score']
        metric_names = ['Total Filings', 'Avg Salary ($)', 'Median Salary ($)', 'Sponsorship Score']
        
        # Normalize for visualization
        compare_melted = compare_df.melt(
            id_vars=['company'], 
            value_vars=metrics,
            var_name='Metric', 
            value_name='Value'
        )
        
        # Map metric names
        compare_melted['Metric'] = compare_melted['Metric'].map(dict(zip(metrics, metric_names)))
        
        fig = px.bar(
            compare_melted, 
            x='Metric', 
            y='Value', 
            color='company',
            barmode='group',
            color_discrete_sequence=['#14b8a6', '#f59e0b', '#8b5cf6']
        )
        fig.update_layout(height=450, xaxis_title="", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📋 Detailed Comparison")
        
        # Comparison table
        comparison_table = compare_df[['company', 'state', 'total_filings', 'avg_salary', 
                                        'median_salary', 'sponsorship_score', 'size_category']]
        comparison_table = comparison_table.set_index('company').T
        comparison_table.index = ['State', 'Total Filings', 'Avg Salary', 'Median Salary', 
                                   'Sponsorship Score', 'Size Category']
        
        # st.dataframe(comparison_table, use_container_width=True)
        st.dataframe(comparison_table, use_container_width=True, key="comparison_table")
        
        # Winner summary
        st.markdown("### 🏆 Comparison Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            most_filings = compare_df.loc[compare_df['total_filings'].idxmax(), 'company']
            st.metric("Most Filings", most_filings)
        
        with col2:
            highest_salary = compare_df.loc[compare_df['avg_salary'].idxmax(), 'company']
            st.metric("Highest Salary", highest_salary)
        
        with col3:
            best_score = compare_df.loc[compare_df['sponsorship_score'].idxmax(), 'company']
            st.metric("Best Score", best_score)
    else:
        st.warning("Please select at least 2 different companies to compare.")

# =============================================================================
# PAGE: SPONSORSHIP PREDICTOR
# =============================================================================
elif page == "🤖 Sponsorship Predictor":
    st.markdown('<h1 class="main-header">Sponsorship Predictor</h1>', unsafe_allow_html=True)
    st.markdown("Predict H-1B sponsorship likelihood based on your profile")
    
    # Input form
    st.markdown("### 📝 Enter Your Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        job_role = st.selectbox(
            "Job Role Category",
            ["Software Engineer", "Data Scientist / Analyst", "Manager / Lead", 
             "Consultant", "Research Scientist", "Other"]
        )
        
        salary = st.slider(
            "Expected Salary ($)",
            min_value=60000,
            max_value=250000,
            value=120000,
            step=5000,
            format="$%d"
        )
        
        state = st.selectbox(
            "Target State",
            ["CA", "WA", "NY", "TX", "NJ", "MA", "IL", "Other"]
        )
    
    with col2:
        company_size = st.selectbox(
            "Target Company Size",
            ["Enterprise (2000+ filings)", "Large (500-2000 filings)", 
             "Medium (100-500 filings)", "Small (<100 filings)"]
        )
        # Extract just the size category
        company_size = company_size.split(" (")[0]
        
        education = st.selectbox(
            "Education Level",
            ["Bachelors", "Masters", "PhD"]
        )
    
    st.markdown("---")
    
    # Predict button
    if st.button("🎯 Predict Sponsorship Likelihood", type="primary", use_container_width=True):
        
        predictor = H1BSponsorshipPredictor()
        result = predictor.predict(job_role, salary, state, company_size, education)
        
        # Display result
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
        
        with col2:
            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result.confidence,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#14b8a6"},
                    'steps': [
                        {'range': [0, 40], 'color': "#fee2e2"},
                        {'range': [40, 70], 'color': "#fef3c7"},
                        {'range': [70, 100], 'color': "#d1fae5"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': result.confidence
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        
        # Factor explanations
        st.markdown("### 📊 Factor Analysis")
        
        for factor, explanation in result.factors.items():
            st.markdown(f"""
            <div class="factor-card">
                <strong>{factor.replace('_', ' ').title()}</strong><br>
                {explanation}
            </div>
            """, unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        for i, rec in enumerate(result.recommendations, 1):
            st.markdown(f"{i}. {rec}")
        
        # Feature importance
        st.markdown("### 📈 Feature Importance")
        
        importance = predictor.get_feature_importance()
        fig = px.bar(
            x=list(importance.values()),
            y=list(importance.keys()),
            orientation='h',
            color=list(importance.values()),
            color_continuous_scale='Teal'
        )
        fig.update_layout(
            height=300,
            xaxis_title="Importance (%)",
            yaxis_title="",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE: DATA PIPELINE
# =============================================================================
elif page == "⚙️ Data Pipeline":
    st.markdown('<h1 class="main-header">Interactive Data Pipeline</h1>', unsafe_allow_html=True)
    st.markdown("Upload raw USCIS data → Clean → Load into app")
    
    st.markdown("### Step 1: Upload USCIS Excel File")
    st.markdown("""
    Download from: [DOL LCA Disclosure Data](https://www.dol.gov/agencies/eta/foreign-labor/performance)
    """)
    
    uploaded_file = st.file_uploader(
        "Upload LCA Disclosure Excel File",
        type=['xlsx', 'xls'],
        help="Upload the USCIS LCA disclosure Excel file (e.g., LCA_Disclosure_Data_FY2024_Q4.xlsx)"
    )
    
    if uploaded_file:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        st.markdown("### Step 2: Configure Cleaning Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            top_n = st.slider(
                "Number of Top Companies to Keep",
                min_value=10,
                max_value=100,
                value=50,
                step=10
            )
        
        with col2:
            min_filings = st.slider(
                "Minimum Filings Threshold",
                min_value=10,
                max_value=500,
                value=100,
                step=10
            )
        
        st.markdown("### Step 3: Process Data")
        
        if st.button("🚀 Clean & Load Data", type="primary", use_container_width=True):
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(progress, message):
                progress_bar.progress(progress)
                status_text.text(message)
            
            try:
                cleaned_df, stats = process_uploaded_file(
                    uploaded_file,
                    top_n=top_n,
                    min_filings=min_filings,
                    progress_callback=update_progress
                )
                
                # Update session state
                st.session_state.df = cleaned_df
                
                st.success("✅ Data processed successfully!")
                
                # Show stats
                st.markdown("### 📊 Processing Summary")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Original Rows", f"{stats['original_rows']:,}")
                col2.metric("After Filtering", f"{stats['filtered_rows']:,}")
                col3.metric("Final Companies", stats['final_companies'])
                
                st.markdown("**Processing Steps:**")
                for step in stats['steps']:
                    st.markdown(f"- {step}")
                
                # Preview
                st.markdown("### 📋 Data Preview")
                st.dataframe(cleaned_df.head(10), use_container_width=True)
                
                # Download option
                csv = cleaned_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Cleaned CSV",
                    csv,
                    "cleaned_h1b_data.csv",
                    "text/csv",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    else:
        st.info("👆 Upload a file to get started")
        
        # Show current data info
        st.markdown("### 📊 Currently Loaded Data")
        st.markdown(f"- **Companies:** {len(df)}")
        st.markdown(f"- **Total Filings:** {df['total_filings'].sum():,}")
        st.markdown(f"- **Avg Salary:** ${df['avg_salary'].mean():,.0f}")

# =============================================================================
# PAGE: AI ADVISOR
# =============================================================================
elif page == "💬 AI Advisor":
    st.markdown('<h1 class="main-header">AI Advisor (RAG)</h1>', unsafe_allow_html=True)
    st.markdown("Ask questions about H-1B sponsorship using GPT-4o + Pinecone")
    
    # Setup
    if use_demo:
        st.info("🎮 **Demo Mode** - Using mock responses. Add API keys for real GPT-4o.")
        vector_store = MockVectorStore(df)
        agent = MockRAGAgent(vector_store, df)
    else:
        if not st.session_state.indexed:
            st.warning("⚠️ Index companies to Pinecone first for semantic search.")
            if st.button("📥 Index to Pinecone", type="primary"):
                with st.spinner("Indexing..."):
                    try:
                        vs = H1BVectorStore(openai_key, pinecone_key)
                        vs.initialize()
                        vs.index_companies(df)
                        st.session_state.vector_store = vs
                        st.session_state.indexed = True
                        st.success("✅ Indexed!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        if st.session_state.indexed:
            vector_store = st.session_state.vector_store
            agent = H1BRAGAgent(openai_key, vector_store)
        else:
            vector_store = MockVectorStore(df)
            agent = MockRAGAgent(vector_store, df)
    
    # Sample questions
    st.markdown("### 💡 Sample Questions")
    samples = ["Top H-1B sponsors?", "Highest paying companies?", "H-1B lottery odds?", "Job search tips?"]
    cols = st.columns(4)
    for i, q in enumerate(samples):
        if cols[i].button(q, key=f"s{i}"):
            st.session_state.chat_history.append({"role": "user", "content": q})
            response = agent.chat(q)
            st.session_state.chat_history.append({"role": "assistant", "content": response.response, "sources": response.sources})
            st.rerun()
    
    st.markdown("---")
    
    # Chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if prompt := st.chat_input("Ask about H-1B..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agent.chat(prompt)
                st.write(response.response)
        st.session_state.chat_history.append({"role": "assistant", "content": response.response, "sources": response.sources})
    
    if st.button("🗑️ Clear"):
        st.session_state.chat_history = []
        st.rerun()

# =============================================================================
# PAGE: ABOUT
# =============================================================================
elif page == "ℹ️ About":
    st.markdown('<h1 class="main-header">About This Project</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 H-1B Sponsorship Intelligence Platform
    
    **INFO 7390: Advances in Data Science and Architecture**  
    **Author:** Abhinav Kumar Piyush  
    **University:** Northeastern University
    
    ---
    
    ### 📋 Project Overview
    
    A RAG (Retrieval-Augmented Generation) application that helps international students 
    analyze H-1B visa sponsorship patterns using **real USCIS data**.
    
    ---
    
    ### 🏗️ Architecture
    
    ```
    ┌─────────────────────────────────────────────────────────┐
    │  DATA PIPELINE                                          │
    │  Raw USCIS Excel → clean_data.py → Cleaned CSV          │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │  STREAMLIT APPLICATION                                  │
    ├─────────────────────────────────────────────────────────┤
    │  📊 Dashboard      │ Visualize top sponsors, salaries   │
    │  📈 Comparison     │ Compare up to 3 companies          │
    │  🤖 Predictor      │ ML model for sponsorship likelihood│
    │  ⚙️ Pipeline       │ Upload → Clean → Load workflow     │
    │  💬 AI Advisor     │ RAG with Pinecone + GPT-4o         │
    └─────────────────────────────────────────────────────────┘
    ```
    
    ---
    
    ### ✅ Requirements Checklist
    
    | Requirement | Implementation | Status |
    |-------------|----------------|--------|
    | Domain Selection | H-1B Visa Sponsorship | ✅ |
    | Data Collection | Real USCIS LCA Data (FY2024) | ✅ |
    | Data Preprocessing | Interactive Pipeline | ✅ |
    | Vector Database | Pinecone + OpenAI Embeddings | ✅ |
    | LLM Integration | GPT-4o RAG | ✅ |
    | Streamlit UI | Multi-page Application | ✅ |
    | ML Model | Sponsorship Predictor | ✅ |
    
    ---
    
    ### 🔮 Future Work
    
    **1. Time-Series Prediction**
    
    With multi-year data (2020-2024), we could:
    - Predict company's 2025 filing volume
    - Identify growing vs declining sponsors
    - Forecast industry trends
    
    *Requirement: 5 years of USCIS LCA data (~3GB total)*
    
    **2. Resume-Based Prediction**
    
    Upload a resume to predict H-1B approval likelihood:
    - Extract skills, education, experience via NLP
    - Match against successful H-1B profiles
    - Provide personalized recommendations
    
    *Requirement: Labeled dataset of approved/denied H-1B applications with resume data (not publicly available)*
    
    ---
    
    ### 📊 Data Source
    
    - **Source:** USCIS LCA Disclosure Data
    - **URL:** https://www.dol.gov/agencies/eta/foreign-labor/performance
    - **Year:** FY2024
    - **Original Records:** ~600,000+
    - **After Cleaning:** Top 50 companies
    
    ---
    
    ### 🛠️ Technologies
    
    | Component | Technology |
    |-----------|------------|
    | Vector DB | Pinecone |
    | LLM | OpenAI GPT-4o |
    | Embeddings | text-embedding-3-small |
    | ML Model | Scikit-learn |
    | Frontend | Streamlit |
    | Data | Pandas, NumPy |
    | Visualization | Plotly |
    
    ---
    
    ### ⚖️ Ethical Considerations
    
    - ✅ Uses only public government data
    - ✅ Clear disclaimers about lottery system (~25% selection)
    - ✅ Not legal advice - informational only
    - ✅ Past patterns ≠ future guarantees
    - ✅ No personal applicant data used
    
    ---
    
    **| Northeastern University | Fall 2025**
    """)
