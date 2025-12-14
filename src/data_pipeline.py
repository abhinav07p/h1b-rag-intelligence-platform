"""
H-1B Data Pipeline
==================
Interactive data cleaning pipeline for USCIS LCA data.
Used in the Streamlit app for Upload → Clean → Load workflow.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import io


def clean_employer_name(name: str) -> str:
    """Standardize employer names."""
    
    if pd.isna(name):
        return "UNKNOWN"
    
    name = str(name).upper().strip()
    
    # Remove common suffixes
    suffixes = [', INC.', ', INC', ' INC.', ' INC', ', LLC', ' LLC', 
               ', LP', ' LP', ', LLP', ' LLP', ' CORP.', ' CORP',
               ', CORPORATION', ' CORPORATION', ' CO.', ' CO',
               ', L.L.C.', ' L.L.C.', ', INCORPORATED', ' INCORPORATED']
    
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    
    # Standardize known companies
    name_mappings = {
        'GOOGLE': 'GOOGLE', 'ALPHABET': 'GOOGLE',
        'META PLATFORMS': 'META', 'FACEBOOK': 'META',
        'AMAZON.COM': 'AMAZON', 'AMAZON WEB SERVICES': 'AMAZON', 'AMAZON.COM SERVICES': 'AMAZON',
        'MICROSOFT': 'MICROSOFT',
        'APPLE': 'APPLE',
        'INFOSYS': 'INFOSYS', 'INFOSYS LIMITED': 'INFOSYS',
        'TATA CONSULTANCY': 'TCS', 'TATA AMERICA': 'TCS',
        'WIPRO': 'WIPRO',
        'COGNIZANT': 'COGNIZANT', 'COGNIZANT TECHNOLOGY': 'COGNIZANT',
        'DELOITTE': 'DELOITTE', 'DELOITTE CONSULTING': 'DELOITTE',
        'ACCENTURE': 'ACCENTURE',
        'IBM': 'IBM', 'INTERNATIONAL BUSINESS MACHINES': 'IBM',
        'INTEL': 'INTEL',
        'NVIDIA': 'NVIDIA',
        'SALESFORCE': 'SALESFORCE',
        'ORACLE': 'ORACLE',
        'CISCO': 'CISCO', 'CISCO SYSTEMS': 'CISCO',
        'UBER': 'UBER', 'UBER TECHNOLOGIES': 'UBER',
        'JPMORGAN': 'JPMORGAN CHASE', 'JP MORGAN': 'JPMORGAN CHASE',
        'GOLDMAN SACHS': 'GOLDMAN SACHS',
        'MORGAN STANLEY': 'MORGAN STANLEY',
        'ERNST & YOUNG': 'EY', 'ERNST AND YOUNG': 'EY',
    }
    
    for key, value in name_mappings.items():
        if key in name:
            return value
    
    return name.strip()


def convert_to_annual_salary(wage, unit) -> float:
    """Convert wage to annual salary."""
    
    if pd.isna(wage) or pd.isna(unit):
        return np.nan
    
    try:
        wage = float(wage)
    except:
        return np.nan
    
    unit = str(unit).upper()
    
    if 'YEAR' in unit:
        return wage
    elif 'MONTH' in unit:
        return wage * 12
    elif 'BI-WEEK' in unit:
        return wage * 26
    elif 'WEEK' in unit:
        return wage * 52
    elif 'HOUR' in unit:
        return wage * 2080
    else:
        return wage


def process_uploaded_file(uploaded_file, top_n: int = 50, min_filings: int = 100, 
                          progress_callback=None) -> Tuple[pd.DataFrame, dict]:
    """
    Process uploaded USCIS Excel file.
    
    Returns:
        Tuple of (cleaned DataFrame, stats dict)
    """
    
    stats = {'steps': []}
    
    # Step 1: Load file
    if progress_callback:
        progress_callback(0.1, "Loading Excel file...")
    
    df = pd.read_excel(uploaded_file)
    stats['original_rows'] = len(df)
    stats['steps'].append(f"Loaded {len(df):,} rows")
    
    # Step 2: Identify columns
    if progress_callback:
        progress_callback(0.2, "Identifying columns...")
    
    # Find employer name column
    employer_col = None
    for col in ['EMPLOYER_NAME', 'EMPLOYER_BUSINESS_NAME', 'EMPLOYER']:
        if col in df.columns:
            employer_col = col
            break
    
    # Find status column
    status_col = None
    for col in ['CASE_STATUS', 'STATUS']:
        if col in df.columns:
            status_col = col
            break
    
    # Find wage columns
    wage_col = None
    for col in ['WAGE_RATE_OF_PAY_FROM', 'WAGE_RATE', 'PREVAILING_WAGE']:
        if col in df.columns:
            wage_col = col
            break
    
    wage_unit_col = None
    for col in ['WAGE_UNIT_OF_PAY', 'WAGE_RATE_UNIT', 'PW_UNIT_OF_PAY']:
        if col in df.columns:
            wage_unit_col = col
            break
    
    # Find state column
    state_col = None
    for col in ['EMPLOYER_STATE', 'EMPLOYER_PROVINCE', 'WORKSITE_STATE']:
        if col in df.columns:
            state_col = col
            break
    
    # Find visa class column
    visa_col = None
    for col in ['VISA_CLASS', 'PROGRAM']:
        if col in df.columns:
            visa_col = col
            break
    
    # Find job title column
    job_col = None
    for col in ['JOB_TITLE', 'POSITION_TITLE', 'SOC_TITLE']:
        if col in df.columns:
            job_col = col
            break
    
    stats['steps'].append(f"Found columns: employer={employer_col}, status={status_col}")
    
    # Step 3: Filter H-1B and Certified
    if progress_callback:
        progress_callback(0.3, "Filtering H-1B Certified cases...")
    
    if visa_col:
        df = df[df[visa_col].str.contains('H-1B', case=False, na=False)]
        stats['steps'].append(f"After H-1B filter: {len(df):,} rows")
    
    if status_col:
        df = df[df[status_col].str.contains('CERTIFIED', case=False, na=False)]
        stats['steps'].append(f"After Certified filter: {len(df):,} rows")
    
    stats['filtered_rows'] = len(df)
    
    # Step 4: Clean employer names
    if progress_callback:
        progress_callback(0.5, "Cleaning employer names...")
    
    df['EMPLOYER_CLEAN'] = df[employer_col].apply(clean_employer_name)
    
    # Step 5: Convert wages
    if progress_callback:
        progress_callback(0.6, "Processing salaries...")
    
    if wage_col and wage_unit_col:
        df['ANNUAL_SALARY'] = df.apply(
            lambda row: convert_to_annual_salary(row[wage_col], row[wage_unit_col]),
            axis=1
        )
    elif wage_col:
        df['ANNUAL_SALARY'] = pd.to_numeric(df[wage_col], errors='coerce')
    
    # Filter realistic salaries
    df = df[(df['ANNUAL_SALARY'] >= 30000) & (df['ANNUAL_SALARY'] <= 500000)]
    
    # Step 6: Aggregate by company
    if progress_callback:
        progress_callback(0.7, "Aggregating by company...")
    
    company_stats = df.groupby('EMPLOYER_CLEAN').agg({
        'ANNUAL_SALARY': ['count', 'mean', 'median'],
        state_col: lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
    }).reset_index()
    
    company_stats.columns = ['company', 'total_filings', 'avg_salary', 'median_salary', 'state']
    
    # Step 7: Filter and sort
    if progress_callback:
        progress_callback(0.8, f"Selecting top {top_n} companies...")
    
    company_stats = company_stats[company_stats['total_filings'] >= min_filings]
    company_stats = company_stats.sort_values('total_filings', ascending=False)
    company_stats = company_stats.head(top_n).reset_index(drop=True)
    
    stats['final_companies'] = len(company_stats)
    stats['steps'].append(f"Final: {len(company_stats)} companies")
    
    # Step 8: Add derived features
    if progress_callback:
        progress_callback(0.9, "Calculating scores...")
    
    # Sponsorship score
    max_filings = company_stats['total_filings'].max()
    company_stats['volume_score'] = (company_stats['total_filings'] / max_filings * 40).clip(0, 40)
    
    max_salary = company_stats['avg_salary'].max()
    min_salary = company_stats['avg_salary'].min()
    if max_salary > min_salary:
        company_stats['salary_score'] = ((company_stats['avg_salary'] - min_salary) / (max_salary - min_salary) * 30).clip(0, 30)
    else:
        company_stats['salary_score'] = 15
    
    company_stats['sponsorship_score'] = (company_stats['volume_score'] + company_stats['salary_score'] + 30).round(1)
    
    # Size category
    company_stats['size_category'] = pd.cut(
        company_stats['total_filings'],
        bins=[0, 500, 2000, 10000, float('inf')],
        labels=['Small', 'Medium', 'Large', 'Enterprise']
    )
    
    # Round salaries
    company_stats['avg_salary'] = company_stats['avg_salary'].round(0).astype(int)
    company_stats['median_salary'] = company_stats['median_salary'].round(0).astype(int)
    
    # Keep only needed columns
    final_cols = ['company', 'state', 'total_filings', 'avg_salary', 'median_salary', 
                  'sponsorship_score', 'size_category']
    company_stats = company_stats[final_cols]
    
    if progress_callback:
        progress_callback(1.0, "Done!")
    
    stats['total_filings'] = company_stats['total_filings'].sum()
    stats['avg_salary'] = company_stats['avg_salary'].mean()
    
    return company_stats, stats
