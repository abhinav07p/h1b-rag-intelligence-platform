"""
About Page
"""
import streamlit as st

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
│  scripts/downloader.py → Cleaned CSVs per year         │
│  h1b_multiyear.csv → LSTM training input               │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  STREAMLIT MULTI-PAGE APPLICATION                       │
├─────────────────────────────────────────────────────────┤
│  📊 Dashboard      │ Visualize top sponsors, salaries   │
│  📈 Comparison     │ Compare up to 3 companies          │
│  🤖 Predictor      │ ML + LSTM trend prediction         │
│  ⚙️ Pipeline       │ Upload → Clean → Load workflow     │
│  💬 AI Advisor     │ RAG with Pinecone + GPT-4o         │
│  📥 Download Data  │ Multi-year DOL downloader          │
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
| ML Model | Sponsorship Predictor (sklearn) | ✅ |
| LSTM Model | PyTorch time-series trend predictor | ✅ |
| Multi-year Data | DOL downloader (2020–2025) | ✅ |

---

### 🔮 Future Work

**1. Time-Series Prediction (Implemented)**

With multi-year data (2022–2024), the LSTM predicts filing trends per company.

**2. Resume-Based Prediction**

Upload a resume to predict H-1B approval likelihood:
- Extract skills, education, experience via NLP
- Match against successful H-1B profiles

*Requirement: Labeled dataset of approved/denied applications with resume data*

---

### 📊 Data Source

- **Source:** USCIS LCA Disclosure Data
- **URL:** https://www.dol.gov/agencies/eta/foreign-labor/performance
- **Years:** FY2022–FY2024 (multi-year LSTM) / FY2024 (demo)
- **Original Records:** ~600,000+ per year
- **After Cleaning:** Top 50 companies

---

### 🛠️ Technologies

| Component | Technology |
|-----------|------------|
| Vector DB | Pinecone |
| LLM | OpenAI GPT-4o |
| Embeddings | text-embedding-3-small |
| ML Model | Scikit-learn (LR, RF, GBM, MLP) |
| LSTM Model | PyTorch |
| Frontend | Streamlit (multi-page) |
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
