# H-1B Sponsorship Intelligence Platform

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red.svg)](https://streamlit.io)

**INFO 7390: Advances in Data Science and Architecture**  
**Author:** Abhinav Kumar Piyush  
**University:** Northeastern University  

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Solution Architecture](#solution-architecture)
4. [Features](#features)
5. [Technology Stack](#technology-stack)
6. [Project Structure](#project-structure)
7. [Installation & Setup](#installation--setup)
8. [Data Pipeline](#data-pipeline)
9. [RAG Implementation](#rag-implementation)
10. [ML Model](#ml-model)
11. [Evaluation](#evaluation)
12. [Deployment](#deployment)
13. [Demo](#demo)
14. [Limitations & Future Work](#limitations--future-work)
15. [References](#references)

---

## Project Overview

This project implements an AI-powered Retrieval-Augmented Generation (RAG) system using Pinecone and GPT-4o to analyze real USCIS H-1B data, enabling semantic search, predictive insights, and explainable recommendations through an interactive Streamlit application.

The system is designed to help international students make data-driven decisions by understanding H-1B visa sponsorship patterns across companies, roles, salaries, and locations.

---

## Problem Statement

International students face significant challenges when navigating the H-1B visa sponsorship landscape:

| Challenge | Description |
|-----------|-------------|
| **Information Asymmetry** | No centralized platform to understand which companies actively sponsor H-1B visas |
| **Lottery Uncertainty** | H-1B has only ~25-30% selection rate; students need data-driven strategies |
| **No Predictive Insights** | Raw government data exists but lacks analysis on sponsorship likelihood |
| **Time-Consuming Research** | Manual research across multiple sources is inefficient |

---

## Solution Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│   📥 Raw USCIS Excel        🔄 clean_data.py          📊 Cleaned CSV       │
│   (600K+ applications)  ──────────────────────►  (Top 50 companies)         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           APPLICATION LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│   ┌───────────────┐   ┌───────────────┐   ┌───────────────────────────────┐ │
│   │   Dashboard   │   │   Predictor   │   │        RAG Chatbot            │ │
│   │   (Plotly)    │   │  (ML Model)   │   │   (Pinecone + GPT-4o)         │ │
│   └───────────────┘   └───────────────┘   └───────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EXTERNAL SERVICES                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│   🔷 Pinecone                              🤖 OpenAI                       │
│   • Vector Database                        • text-embedding-3-small         │
│   • Semantic Search                        • GPT-4o Chat Completions        │
│   • 1536-dim embeddings                    • Context-aware responses        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Features

### 1. Dashboard (📊)
- Top 10 H-1B sponsors visualization
- Salary distribution charts
- State-wise filing breakdown
- Company size distribution
- Interactive data table with search/filter

### 2. Company Comparison (📈)
- Compare up to 3 companies side-by-side
- Bar charts for filings, salary, score
- Detailed comparison table
- Winner summary cards

### 3. Sponsorship Predictor (🤖)
- Input: Job role, salary, state, company size, education
- Output: HIGH/MEDIUM/LOW likelihood
- Confidence score with gauge visualization
- Factor-by-factor explanation
- Personalized recommendations

### 4. Data Pipeline (⚙️)
- Upload raw USCIS Excel files
- Configure cleaning parameters
- Real-time processing with progress bar
- Download cleaned CSV

### 5. AI Advisor (💬)
- RAG-powered chatbot
- Semantic search via Pinecone
- GPT-4o response generation
- Context-aware answers from real data
- Sample question buttons

### 6. About (ℹ️)
- Project documentation
- Architecture diagram
- Requirements checklist
- Future work roadmap

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit | Interactive web application |
| **Vector Database** | Pinecone | Semantic search & retrieval |
| **LLM** | OpenAI GPT-4o | Response generation |
| **Embeddings** | text-embedding-3-small | Convert text to vectors (1536-dim) |
| **ML Models** | Scikit-learn, XGBoost | Sponsorship prediction |
| **Visualization** | Plotly | Interactive charts |
| **Data Processing** | Pandas, NumPy | Data cleaning & analysis |
| **Language** | Python 3.10+ | Backend development |

---

## Project Structure

```
h1b-rag-project/
│
├── app.py                          # Main Streamlit application (6 pages)
├── clean_data.py                   # Standalone data cleaning script
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
├── CITATIONS.md                    # Data sources & references
├── README.md                       # This file
│
├── src/                            # Source modules
│   ├── __init__.py
│   ├── data_loader.py              # Load and validate CSV data
│   ├── data_pipeline.py            # Interactive data cleaning
│   ├── prediction_model.py         # ML sponsorship predictor
│   ├── vector_store.py             # Pinecone vector database
│   └── rag_agent.py                # GPT-4o RAG pipeline
│
├── notebooks/                      # Jupyter notebooks
│   └── H1B_Model_Training.ipynb    # ML model training (Colab ready)
│
├── docs/                           # Documentation
│   └── index.html                  # Project documentation webpage
│
├── data/                           # Data files
│   └── cleaned_h1b_data.csv        # Cleaned H-1B data (user adds)
│
└── .streamlit/                     # Streamlit configuration
    ├── config.toml                 # Theme & server settings
    └── secrets.toml                # API keys (not in git)
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- OpenAI API key (for AI Advisor)
- Pinecone API key (for AI Advisor)

### Step 1: Clone Repository
```bash
git clone https://github.com/abhinav07p/h1b-rag-intelligence-platform.git
cd h1b-rag-project
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Add Data
Download USCIS data from [DOL Website](https://www.dol.gov/agencies/eta/foreign-labor/performance) and clean it:
```bash
python clean_data.py
cp cleaned_h1b_data.csv data/
```

Or use the interactive Data Pipeline in the app.

### Step 4: Configure API Keys (Optional)
Create `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-..."
PINECONE_API_KEY = "pcsk_..."
```

### Step 5: Run Application
```bash
streamlit run app.py
```

App opens at: `http://localhost:8501`

---

## Data Pipeline

### Data Source
| Attribute | Value |
|-----------|-------|
| **Source** | U.S. Department of Labor |
| **Dataset** | LCA Disclosure Data FY2024 Q4 |
| **URL** | https://www.dol.gov/agencies/eta/foreign-labor/performance |
| **Original Size** | 600,000+ H-1B applications |
| **License** | Public Domain (U.S. Government Work) |

### Cleaning Steps
1. **Filter** - H-1B Certified applications only
2. **Standardize** - Clean company names (e.g., "GOOGLE LLC" → "GOOGLE")
3. **Convert** - All wages to annual salary
4. **Aggregate** - Group by company, calculate statistics
5. **Score** - Compute sponsorship score (0-100)
6. **Categorize** - Add size_category, salary_category

### Output Schema
| Column | Type | Description |
|--------|------|-------------|
| company | string | Cleaned company name |
| state | string | Primary state |
| total_filings | int | Number of H-1B filings |
| avg_salary | float | Average salary |
| median_salary | float | Median salary |
| sponsorship_score | float | 0-100 score |
| size_category | string | Enterprise/Large/Medium/Small |
| salary_category | string | High/Medium/Low |

---

## RAG Implementation

### How It Works

```
User Query → Embedding → Pinecone Search → Top-K Results → GPT-4o → Response
     │            │              │               │              │
"Top sponsors?"  [0.12, -0.45,   Find similar   Amazon, Google  "Based on 2024
                  0.78, ...]     companies      Microsoft...    data, the top
                                                                sponsors are..."
```

### Components

**1. Vector Store (Pinecone)**
- Index: `h1b-companies`
- Dimensions: 1536 (OpenAI embeddings)
- Metric: Cosine similarity
- Cloud: AWS us-east-1

**2. Embeddings (OpenAI)**
- Model: `text-embedding-3-small`
- Each company converted to text → embedding
- Stored in Pinecone with metadata

**3. RAG Agent (GPT-4o)**
- Retrieves relevant companies from Pinecone
- Constructs prompt with context
- Generates grounded response

### Why Pinecone?
Pinecone was selected for its managed scalability, fast cosine similarity search, and seamless integration with OpenAI embeddings, allowing efficient semantic retrieval without infrastructure overhead.

### RAG Evaluation
The RAG system was evaluated qualitatively by testing domain-specific queries and verifying that responses were grounded in retrieved company data rather than hallucinated content. Queries consistently returned context-aware answers based on Pinecone retrieval, demonstrating reliable semantic search and factual grounding.

### RAG Evaluation Results

| Test Query | Expected Company | Retrieved? | Response Grounded? |
|------------|------------------|------------|-------------------|
| "Top H-1B sponsors" | Amazon, Microsoft | ✅ Yes | ✅ Yes |
| "Highest paying companies" | Meta, Google, Apple | ✅ Yes | ✅ Yes |
| "Companies in California" | Google, Meta, Apple | ✅ Yes | ✅ Yes |
| "Consulting firms for H-1B" | Cognizant, TCS, Infosys | ✅ Yes | ✅ Yes |
| "Best company in Washington" | Amazon, Microsoft | ✅ Yes | ✅ Yes |

**Grounding Rate:** 100% (5/5 queries returned factually correct, data-backed responses)

**Avg Response Time:** ~2-3 seconds

**Retrieval Accuracy:** Top-K results matched expected companies for all test queries

---

## ML Model

### Sponsorship Predictor

**Input Features:**
| Feature | Type | Values |
|---------|------|--------|
| job_category | categorical | Software Engineer, Data Scientist, Manager, Consultant, Research, Other |
| salary | numerical | $60K - $250K |
| state | categorical | CA, WA, NY, TX, NJ, MA, IL, Other |
| avg_salary | numerical | From company data |

**Output:**
- Likelihood: HIGH / MEDIUM / LOW
- Confidence: 0-100%
- Factor explanations
- Recommendations

### Training Results

| Model | Accuracy | AUC | CV Score |
|-------|----------|-----|----------|
| Logistic Regression | 99.9% | 0.68 | 99.9% |
| Decision Tree | 99.9% | 0.64 | 99.9% |
| Random Forest | 99.9% | 0.64 | 99.9% |
| XGBoost | 99.9% | 0.56 | 99.9% |

**Best Model:** Logistic Regression

### Feature Importance (Trained Weights)

| Feature | Coefficient | Impact |
|---------|-------------|--------|
| state_WA | +0.49 | Positive (Washington) |
| salary_category_High | +0.37 | Positive |
| state_NJ | +0.22 | Slight positive |
| salary_category_Low | -1.69 | Strong negative |
| state_NY | -1.36 | Negative |
| job_category_Other | -1.19 | Negative |

---

## Evaluation

### RAG System
- **Method:** Qualitative testing with domain-specific queries
- **Metrics:** Response grounding, factual accuracy, source citation
- **Result:** Responses consistently grounded in retrieved data

### ML Model
- **Method:** Train/test split (80/20), 5-fold cross-validation
- **Metrics:** Accuracy, Precision, Recall, F1, AUC-ROC
- **Result:** 99.9% accuracy with Logistic Regression

### User Interface
- **Method:** Manual testing across all pages
- **Result:** All features functional, responsive design

---

## Deployment

### Streamlit Cloud (Recommended)

1. Push to GitHub:
```bash
git add .
git commit -m "Deploy H-1B RAG App"
git push origin main
```

2. Go to [share.streamlit.io](https://share.streamlit.io)

3. Connect repository and deploy

4. Add secrets in Streamlit Cloud dashboard:
```toml
OPENAI_API_KEY = "sk-..."
PINECONE_API_KEY = "pcsk_..."
```

### Live Application
**URL:** https://abhinav07p-h1b-rag-intelligence-platform-app-d4spbw.streamlit.app/

---

## Demo

### Video Demo
**YouTube:** https://youtu.be/VLKeaW5GTiA

**Duration:** 10-15 minutes

**Content:**
1. Project introduction and objectives
2. Data cleaning and preparation steps
3. Generative AI integration explanation
4. Live demo of all features
5. Analysis results and conclusions
6. Reflections and challenges

---

## Limitations & Future Work

### Current Limitations
- Relies on historical data; cannot predict policy changes
- Company-level aggregation loses individual application details
- Prediction based on patterns, not guarantees
- Requires API keys for full RAG functionality

### Future Enhancements

**Phase 2: Time-Series Prediction**
- Use 2020-2024 data to predict future trends
- Identify growing vs declining sponsors
- Forecast industry patterns

**Phase 3: Resume-Based Prediction**
- Upload resume for personalized analysis
- NLP extraction of skills and experience
- Match against successful H-1B profiles

---

### Ethical Considerations

**Bias/Fairness:**  
The data reflects historical filing patterns and may over-represent large employers and major tech hubs (CA, WA, NY). Results are not a measure of merit or company quality—interpret cautiously.

**Privacy:**  
This system uses only public, aggregate company-level data from U.S. Department of Labor disclosures. No personal applicant information is collected, stored, or processed.

**Misuse Prevention:**  
This tool is for informational purposes only and should NOT be used for definitive immigration decisions. Always consult a qualified immigration attorney.

**Content Guardrails:**  
The AI Advisor refuses to provide specific legal instructions and instead offers general guidance with cited data sources.

---

## References

### Data Sources
- U.S. Department of Labor - LCA Disclosure Data FY2024
- https://www.dol.gov/agencies/eta/foreign-labor/performance

### Technologies
- [Pinecone Documentation](https://docs.pinecone.io/)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

### Academic References
- Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- Johnson et al. (2019). "Billion-scale similarity search with GPUs"

---

## License

MIT License - see [LICENSE](LICENSE)

---

## Author

**Abhinav Kumar Piyush**  
Master of Science in Information Systems  
Northeastern University  

---

## Acknowledgments

- INFO 7390 Course Instructor
- Northeastern University
- OpenAI and Pinecone for API access
- U.S. Department of Labor for public data

---

*This project is for educational purposes only. The predictions and recommendations are based on historical data patterns and should NOT be considered legal advice. Always consult with a qualified immigration attorney for legal advice.*
