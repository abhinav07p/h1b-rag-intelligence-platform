"""
H-1B Data Cleaning Script - FULL DATA
=====================================
Cleans raw USCIS LCA Disclosure Data and keeps ALL companies.

Author: Abhinav Kumar Piyush
Course: INFO 7390

Usage:
    1. Download FY2024 data from DOL website
    2. Place Excel file in same folder as this script
    3. Run: python clean_data_full.py
    4. Output: cleaned_h1b_data_full.csv
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import time


def find_data_file():
    """Find the LCA disclosure Excel file."""
    current_dir = Path('.')
    
    # Look for LCA files
    patterns = ['*LCA*.xlsx', '*lca*.xlsx', '*H-1B*.xlsx', '*h1b*.xlsx', '*Disclosure*.xlsx']
    
    for pattern in patterns:
        files = list(current_dir.glob(pattern))
        if files:
            return files[0]
    
    # Try any Excel file
    excel_files = list(current_dir.glob('*.xlsx'))
    if excel_files:
        print(f"Found Excel files: {[f.name for f in excel_files]}")
        return excel_files[0]
    
    return None


def clean_employer_name(name):
    """Standardize employer names."""
    if pd.isna(name):
        return "UNKNOWN"
    
    name = str(name).upper().strip()
    
    # Remove common suffixes
    suffixes = [
        ', INC.', ', INC', ' INC.', ' INC', 
        ', LLC', ' LLC', ', LP', ' LP', ', LLP', ' LLP',
        ' CORP.', ' CORP', ', CORPORATION', ' CORPORATION',
        ' CO.', ' CO', ' LTD.', ' LTD', ' LIMITED',
        ', P.C.', ' P.C.', ', PC', ' PC'
    ]
    
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    
    # Standardize known company names
    name_mappings = {
        'GOOGLE': ['GOOGLE', 'ALPHABET'],
        'META': ['META PLATFORMS', 'META', 'FACEBOOK'],
        'AMAZON': ['AMAZON.COM', 'AMAZON WEB SERVICES', 'AMAZON'],
        'MICROSOFT': ['MICROSOFT'],
        'APPLE': ['APPLE'],
        'INFOSYS': ['INFOSYS', 'INFOSYS LIMITED', 'INFOSYS BPM'],
        'TCS': ['TATA CONSULTANCY', 'TATA AMERICA', 'TCS'],
        'COGNIZANT': ['COGNIZANT', 'COGNIZANT TECHNOLOGY'],
        'WIPRO': ['WIPRO'],
        'ACCENTURE': ['ACCENTURE'],
        'DELOITTE': ['DELOITTE', 'DELOITTE CONSULTING', 'DELOITTE & TOUCHE'],
        'EY': ['ERNST & YOUNG', 'ERNST AND YOUNG', 'E&Y'],
        'PWC': ['PRICEWATERHOUSECOOPERS', 'PWC'],
        'KPMG': ['KPMG'],
        'IBM': ['IBM', 'INTERNATIONAL BUSINESS MACHINES'],
        'CAPGEMINI': ['CAPGEMINI', 'CAP GEMINI'],
        'HCL': ['HCL TECHNOLOGIES', 'HCL AMERICA'],
        'TECH MAHINDRA': ['TECH MAHINDRA', 'MAHINDRA SATYAM'],
    }
    
    for standard_name, variants in name_mappings.items():
        for variant in variants:
            if variant in name:
                return standard_name
    
    return name.strip()


def convert_to_annual_salary(wage, unit):
    """Convert wage to annual salary."""
    if pd.isna(wage) or pd.isna(unit):
        return np.nan
    
    try:
        wage = float(str(wage).replace(',', '').replace('$', ''))
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


def clean_data(df, min_filings=1):
    """
    Main cleaning function.
    
    Args:
        df: Raw DataFrame
        min_filings: Minimum filings to include company (default=1, keeps ALL)
    """
    
    print("\n" + "=" * 60)
    print("🧹 CLEANING DATA")
    print("=" * 60)
    
    print(f"\n📊 Raw data: {len(df):,} rows")
    print(f"   Columns: {len(df.columns)}")
    
    # Find columns (they vary by year)
    employer_col = next((c for c in df.columns if 'EMPLOYER' in c.upper() and 'NAME' in c.upper()), None)
    status_col = next((c for c in df.columns if 'STATUS' in c.upper()), None)
    wage_col = next((c for c in df.columns if 'WAGE' in c.upper() and 'FROM' in c.upper()), None)
    if not wage_col:
        wage_col = next((c for c in df.columns if 'WAGE' in c.upper() and 'RATE' in c.upper()), None)
    wage_unit_col = next((c for c in df.columns if 'WAGE' in c.upper() and 'UNIT' in c.upper()), None)
    state_col = next((c for c in df.columns if 'EMPLOYER' in c.upper() and 'STATE' in c.upper()), None)
    if not state_col:
        state_col = next((c for c in df.columns if 'WORKSITE' in c.upper() and 'STATE' in c.upper()), None)
    visa_col = next((c for c in df.columns if 'VISA' in c.upper() or 'CLASS' in c.upper()), None)
    job_col = next((c for c in df.columns if 'JOB' in c.upper() and 'TITLE' in c.upper()), None)
    if not job_col:
        job_col = next((c for c in df.columns if 'SOC' in c.upper() and 'TITLE' in c.upper()), None)
    
    print(f"\n🔍 Detected columns:")
    print(f"   Employer: {employer_col}")
    print(f"   Status: {status_col}")
    print(f"   Wage: {wage_col}")
    print(f"   Wage Unit: {wage_unit_col}")
    print(f"   State: {state_col}")
    print(f"   Visa Class: {visa_col}")
    print(f"   Job Title: {job_col}")
    
    # Filter H-1B only
    if visa_col:
        h1b_mask = df[visa_col].astype(str).str.contains('H-1B', case=False, na=False)
        df = df[h1b_mask]
        print(f"\n✅ After H-1B filter: {len(df):,} rows")
    
    # Filter Certified only
    if status_col:
        certified_mask = df[status_col].astype(str).str.contains('CERTIFIED', case=False, na=False)
        # Exclude "CERTIFIED-WITHDRAWN"
        withdrawn_mask = df[status_col].astype(str).str.contains('WITHDRAWN', case=False, na=False)
        df = df[certified_mask & ~withdrawn_mask]
        print(f"✅ After Certified filter: {len(df):,} rows")
    
    # Clean employer names
    print("\n🔄 Cleaning employer names...")
    df['EMPLOYER_CLEAN'] = df[employer_col].apply(clean_employer_name)
    
    # Convert wages to annual salary
    print("🔄 Converting wages to annual salary...")
    if wage_col and wage_unit_col:
        df['ANNUAL_SALARY'] = df.apply(
            lambda row: convert_to_annual_salary(row[wage_col], row[wage_unit_col]), 
            axis=1
        )
    elif wage_col:
        df['ANNUAL_SALARY'] = pd.to_numeric(df[wage_col], errors='coerce')
    
    # Filter valid salaries
    df = df[(df['ANNUAL_SALARY'] >= 30000) & (df['ANNUAL_SALARY'] <= 500000)]
    print(f"✅ After salary filter ($30K-$500K): {len(df):,} rows")
    
    # Get state
    if state_col:
        df['STATE'] = df[state_col].fillna('Unknown')
    else:
        df['STATE'] = 'Unknown'
    
    # Get job title
    if job_col:
        df['JOB_TITLE'] = df[job_col].fillna('Unknown')
    else:
        df['JOB_TITLE'] = 'Unknown'
    
    # Aggregate by company
    print("\n📊 Aggregating by company...")
    
    company_stats = df.groupby('EMPLOYER_CLEAN').agg({
        'ANNUAL_SALARY': ['count', 'mean', 'median', 'min', 'max', 'std'],
        'STATE': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
        'JOB_TITLE': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
    }).reset_index()
    
    # Flatten column names
    company_stats.columns = [
        'company', 'total_filings', 'avg_salary', 'median_salary', 
        'min_salary', 'max_salary', 'salary_std', 'primary_state', 'primary_job_title'
    ]
    
    # Filter by minimum filings
    company_stats = company_stats[company_stats['total_filings'] >= min_filings]
    print(f"✅ Companies with >= {min_filings} filings: {len(company_stats):,}")
    
    # Sort by filings
    company_stats = company_stats.sort_values('total_filings', ascending=False)
    
    # Calculate sponsorship score (0-100)
    max_filings = company_stats['total_filings'].max()
    max_salary = company_stats['avg_salary'].max()
    min_salary = company_stats['avg_salary'].min()
    
    # Volume score (0-50 points)
    company_stats['volume_score'] = (company_stats['total_filings'] / max_filings * 50).clip(0, 50)
    
    # Salary score (0-30 points)
    salary_range = max_salary - min_salary
    if salary_range > 0:
        company_stats['salary_score'] = ((company_stats['avg_salary'] - min_salary) / salary_range * 30).clip(0, 30)
    else:
        company_stats['salary_score'] = 15
    
    # Consistency score (0-20 points) - lower std = more consistent = better
    max_std = company_stats['salary_std'].max()
    if max_std > 0:
        company_stats['consistency_score'] = ((1 - company_stats['salary_std'].fillna(max_std) / max_std) * 20).clip(0, 20)
    else:
        company_stats['consistency_score'] = 10
    
    # Total sponsorship score
    company_stats['sponsorship_score'] = (
        company_stats['volume_score'] + 
        company_stats['salary_score'] + 
        company_stats['consistency_score']
    ).round(1)
    
    # Size category
    company_stats['size_category'] = pd.cut(
        company_stats['total_filings'],
        bins=[0, 10, 50, 200, 1000, float('inf')],
        labels=['Very Small', 'Small', 'Medium', 'Large', 'Enterprise']
    )
    
    # Salary category
    company_stats['salary_category'] = pd.cut(
        company_stats['avg_salary'],
        bins=[0, 80000, 120000, 160000, float('inf')],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    # Round numeric columns
    company_stats['avg_salary'] = company_stats['avg_salary'].round(0).astype(int)
    company_stats['median_salary'] = company_stats['median_salary'].round(0).astype(int)
    company_stats['min_salary'] = company_stats['min_salary'].round(0).astype(int)
    company_stats['max_salary'] = company_stats['max_salary'].round(0).astype(int)
    company_stats['salary_std'] = company_stats['salary_std'].round(0).fillna(0).astype(int)
    
    # Select final columns
    final_cols = [
        'company', 'primary_state', 'total_filings', 
        'avg_salary', 'median_salary', 'min_salary', 'max_salary', 'salary_std',
        'sponsorship_score', 'size_category', 'salary_category', 'primary_job_title'
    ]
    
    return company_stats[final_cols].reset_index(drop=True)


def main():
    start_time = time.time()
    
    print("=" * 60)
    print("🎯 H-1B DATA CLEANING SCRIPT - FULL DATA")
    print("=" * 60)
    
    # Find data file
    filepath = find_data_file()
    
    if filepath is None:
        print("\n❌ No Excel file found!")
        print("\n📥 Please download the data:")
        print("   1. Go to: https://www.dol.gov/agencies/eta/foreign-labor/performance")
        print("   2. Download LCA Disclosure Data (FY2024)")
        print("   3. Place the Excel file in this folder")
        print("   4. Run this script again")
        return
    
    print(f"\n📂 Found file: {filepath}")
    print(f"   Size: {filepath.stat().st_size / (1024*1024):.1f} MB")
    
    # Load data
    print(f"\n📖 Loading data (this may take a few minutes)...")
    df = pd.read_excel(filepath)
    print(f"   ✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Clean data - keep ALL companies (min_filings=1)
    cleaned_df = clean_data(df, min_filings=1)
    
    # Summary stats
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"\n   Total companies: {len(cleaned_df):,}")
    print(f"   Total filings: {cleaned_df['total_filings'].sum():,}")
    print(f"   Average salary: ${cleaned_df['avg_salary'].mean():,.0f}")
    print(f"   Median salary: ${cleaned_df['median_salary'].median():,.0f}")
    
    print(f"\n   By Size Category:")
    for cat in cleaned_df['size_category'].value_counts().sort_index().items():
        print(f"      {cat[0]}: {cat[1]:,} companies")
    
    print(f"\n   Top 10 Companies by Filings:")
    for i, row in cleaned_df.head(10).iterrows():
        print(f"      {i+1}. {row['company']}: {row['total_filings']:,} filings (${row['avg_salary']:,})")
    
    # Save to CSV
    output_file = 'cleaned_h1b_data_full.csv'
    cleaned_df.to_csv(output_file, index=False)
    
    elapsed = time.time() - start_time
    
    print(f"\n" + "=" * 60)
    print(f"✅ DONE!")
    print(f"=" * 60)
    print(f"\n   Output: {output_file}")
    print(f"   Rows: {len(cleaned_df):,}")
    print(f"   Time: {elapsed:.1f} seconds")
    print(f"\n   You can now use this file with the training notebook!")


if __name__ == "__main__":
    main()
