"""
H-1B Data Loader
================
Loads cleaned H-1B data from CSV file.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def load_h1b_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load cleaned H-1B data from CSV."""
    
    search_paths = [
        filepath,
        'data/cleaned_h1b_data.csv',
        'cleaned_h1b_data.csv',
        '../cleaned_h1b_data.csv',
    ]
    
    for path in search_paths:
        if path and Path(path).exists():
            df = pd.read_csv(path)
            print(f"✅ Loaded {len(df)} companies from {path}")
            return df
    
    print("⚠️ CSV not found, using sample data")
    return _generate_sample_data()


def _generate_sample_data() -> pd.DataFrame:
    """Generate sample data if CSV not found."""
    
    data = [
        {'company': 'AMAZON', 'state': 'WA', 'total_filings': 4459, 'avg_salary': 165977, 'median_salary': 165200, 'sponsorship_score': 86.1, 'size_category': 'Large'},
        {'company': 'MICROSOFT', 'state': 'WA', 'total_filings': 2829, 'avg_salary': 167924, 'median_salary': 165380, 'sponsorship_score': 71.9, 'size_category': 'Large'},
        {'company': 'COGNIZANT', 'state': 'TX', 'total_filings': 2709, 'avg_salary': 111289, 'median_salary': 108618, 'sponsorship_score': 58.6, 'size_category': 'Large'},
        {'company': 'TCS', 'state': 'MD', 'total_filings': 1563, 'avg_salary': 91477, 'median_salary': 87734, 'sponsorship_score': 44.0, 'size_category': 'Medium'},
        {'company': 'META', 'state': 'CA', 'total_filings': 1529, 'avg_salary': 212520, 'median_salary': 214000, 'sponsorship_score': 69.9, 'size_category': 'Medium'},
        {'company': 'GOOGLE', 'state': 'CA', 'total_filings': 1393, 'avg_salary': 190151, 'median_salary': 186000, 'sponsorship_score': 63.9, 'size_category': 'Medium'},
        {'company': 'EY', 'state': 'NJ', 'total_filings': 1350, 'avg_salary': 170309, 'median_salary': 166400, 'sponsorship_score': 59.2, 'size_category': 'Medium'},
        {'company': 'APPLE', 'state': 'CA', 'total_filings': 1272, 'avg_salary': 172128, 'median_salary': 169100, 'sponsorship_score': 58.9, 'size_category': 'Medium'},
        {'company': 'WALMART', 'state': 'AR', 'total_filings': 1128, 'avg_salary': 143879, 'median_salary': 146927, 'sponsorship_score': 51.5, 'size_category': 'Medium'},
        {'company': 'INFOSYS', 'state': 'TX', 'total_filings': 1066, 'avg_salary': 110238, 'median_salary': 107141, 'sponsorship_score': 43.6, 'size_category': 'Medium'},
        {'company': 'NVIDIA', 'state': 'CA', 'total_filings': 950, 'avg_salary': 225000, 'median_salary': 220000, 'sponsorship_score': 72.5, 'size_category': 'Medium'},
        {'company': 'JPMORGAN CHASE', 'state': 'NY', 'total_filings': 890, 'avg_salary': 155000, 'median_salary': 152000, 'sponsorship_score': 55.2, 'size_category': 'Medium'},
        {'company': 'DELOITTE', 'state': 'NY', 'total_filings': 850, 'avg_salary': 145000, 'median_salary': 142000, 'sponsorship_score': 52.8, 'size_category': 'Medium'},
        {'company': 'INTEL', 'state': 'CA', 'total_filings': 780, 'avg_salary': 165000, 'median_salary': 162000, 'sponsorship_score': 56.1, 'size_category': 'Medium'},
        {'company': 'SALESFORCE', 'state': 'CA', 'total_filings': 720, 'avg_salary': 178000, 'median_salary': 175000, 'sponsorship_score': 58.3, 'size_category': 'Medium'},
    ]
    
    return pd.DataFrame(data)


def get_company_summary(df: pd.DataFrame) -> dict:
    """Get summary statistics."""
    
    return {
        'total_companies': len(df),
        'total_filings': df['total_filings'].sum(),
        'avg_salary': df['avg_salary'].mean(),
        'highest_paying': df.loc[df['avg_salary'].idxmax(), 'company'],
        'highest_salary': df['avg_salary'].max(),
        'most_filings_company': df.loc[df['total_filings'].idxmax(), 'company'],
        'most_filings': df['total_filings'].max(),
    }
