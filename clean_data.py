"""
H-1B Data Cleaning Script
=========================
Cleans raw USCIS LCA Disclosure Data for the RAG application.

Author: Abhinav Kumar Piyush
Course: INFO 7390

Usage:
    1. Download FY2024 data from DOL website
    2. Place Excel file in same folder as this script
    3. Run: python clean_data.py
    4. Output: cleaned_h1b_data.csv
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


def find_data_file():
    """Find the LCA disclosure Excel file."""
    current_dir = Path('.')
    excel_files = list(current_dir.glob('*LCA*.xlsx')) + list(current_dir.glob('*lca*.xlsx'))
    
    if excel_files:
        return excel_files[0]
    
    excel_files = list(current_dir.glob('*.xlsx'))
    if excel_files:
        return excel_files[0]
    
    return None


def clean_employer_name(name):
    """Standardize employer names."""
    if pd.isna(name):
        return "UNKNOWN"
    
    name = str(name).upper().strip()
    
    suffixes = [', INC.', ', INC', ' INC.', ' INC', ', LLC', ' LLC', 
               ', LP', ' LP', ', LLP', ' LLP', ' CORP.', ' CORP',
               ', CORPORATION', ' CORPORATION', ' CO.', ' CO']
    
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    
    name_mappings = {
        'GOOGLE': 'GOOGLE', 'ALPHABET': 'GOOGLE',
        'META PLATFORMS': 'META', 'FACEBOOK': 'META',
        'AMAZON.COM': 'AMAZON', 'AMAZON WEB SERVICES': 'AMAZON',
        'MICROSOFT': 'MICROSOFT', 'APPLE': 'APPLE',
        'INFOSYS': 'INFOSYS', 'INFOSYS LIMITED': 'INFOSYS',
        'TATA CONSULTANCY': 'TCS', 'TATA AMERICA': 'TCS',
        'COGNIZANT': 'COGNIZANT', 'COGNIZANT TECHNOLOGY': 'COGNIZANT',
        'DELOITTE': 'DELOITTE', 'DELOITTE CONSULTING': 'DELOITTE',
        'ERNST & YOUNG': 'EY', 'ERNST AND YOUNG': 'EY',
    }
    
    for key, value in name_mappings.items():
        if key in name:
            return value
    
    return name.strip()


def convert_to_annual_salary(wage, unit):
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


def clean_data(df, min_filings=100, top_n=50):
    """Main cleaning function."""
    
    print("\n🧹 Cleaning Data...")
    
    # Find columns
    employer_col = next((c for c in ['EMPLOYER_NAME', 'EMPLOYER_BUSINESS_NAME'] if c in df.columns), None)
    status_col = next((c for c in ['CASE_STATUS', 'STATUS'] if c in df.columns), None)
    wage_col = next((c for c in ['WAGE_RATE_OF_PAY_FROM', 'WAGE_RATE'] if c in df.columns), None)
    wage_unit_col = next((c for c in ['WAGE_UNIT_OF_PAY', 'WAGE_RATE_UNIT'] if c in df.columns), None)
    state_col = next((c for c in ['EMPLOYER_STATE', 'WORKSITE_STATE'] if c in df.columns), None)
    visa_col = next((c for c in ['VISA_CLASS', 'PROGRAM'] if c in df.columns), None)
    
    # Filter H-1B Certified
    if visa_col:
        df = df[df[visa_col].str.contains('H-1B', case=False, na=False)]
    if status_col:
        df = df[df[status_col].str.contains('CERTIFIED', case=False, na=False)]
    
    print(f"   After filtering: {len(df):,} rows")
    
    # Clean names
    df['EMPLOYER_CLEAN'] = df[employer_col].apply(clean_employer_name)
    
    # Convert wages
    if wage_col and wage_unit_col:
        df['ANNUAL_SALARY'] = df.apply(
            lambda row: convert_to_annual_salary(row[wage_col], row[wage_unit_col]), axis=1
        )
    elif wage_col:
        df['ANNUAL_SALARY'] = pd.to_numeric(df[wage_col], errors='coerce')
    
    df = df[(df['ANNUAL_SALARY'] >= 30000) & (df['ANNUAL_SALARY'] <= 500000)]
    
    # Aggregate by company
    company_stats = df.groupby('EMPLOYER_CLEAN').agg({
        'ANNUAL_SALARY': ['count', 'mean', 'median'],
        state_col: lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
    }).reset_index()
    
    company_stats.columns = ['company', 'total_filings', 'avg_salary', 'median_salary', 'state']
    
    # Filter and sort
    company_stats = company_stats[company_stats['total_filings'] >= min_filings]
    company_stats = company_stats.sort_values('total_filings', ascending=False).head(top_n)
    
    # Add features
    max_filings = company_stats['total_filings'].max()
    company_stats['volume_score'] = (company_stats['total_filings'] / max_filings * 40).clip(0, 40)
    
    max_salary = company_stats['avg_salary'].max()
    min_salary = company_stats['avg_salary'].min()
    company_stats['salary_score'] = ((company_stats['avg_salary'] - min_salary) / (max_salary - min_salary) * 30).clip(0, 30)
    
    company_stats['sponsorship_score'] = (company_stats['volume_score'] + company_stats['salary_score'] + 30).round(1)
    
    company_stats['size_category'] = pd.cut(
        company_stats['total_filings'],
        bins=[0, 500, 2000, 10000, float('inf')],
        labels=['Small', 'Medium', 'Large', 'Enterprise']
    )
    
    company_stats['avg_salary'] = company_stats['avg_salary'].round(0).astype(int)
    company_stats['median_salary'] = company_stats['median_salary'].round(0).astype(int)
    
    final_cols = ['company', 'state', 'total_filings', 'avg_salary', 'median_salary', 
                  'sponsorship_score', 'size_category']
    
    return company_stats[final_cols].reset_index(drop=True)


def main():
    print("=" * 60)
    print("🎯 H-1B DATA CLEANING SCRIPT")
    print("=" * 60)
    
    filepath = find_data_file()
    
    if filepath is None:
        print("\n❌ No Excel file found!")
        print("Download from: https://www.dol.gov/agencies/eta/foreign-labor/performance")
        return
    
    print(f"\n📂 Loading {filepath}...")
    df = pd.read_excel(filepath)
    print(f"   ✅ Loaded {len(df):,} rows")
    
    cleaned_df = clean_data(df, min_filings=100, top_n=50)
    
    print(f"\n📊 Top 10 Companies:")
    print(cleaned_df.head(10).to_string(index=False))
    
    output_file = 'cleaned_h1b_data.csv'
    cleaned_df.to_csv(output_file, index=False)
    print(f"\n💾 Saved to: {output_file}")
    
    print("\n" + "=" * 60)
    print("✅ DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
