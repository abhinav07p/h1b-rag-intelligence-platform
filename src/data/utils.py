"""
H-1B Data Utilities
===================
Shared helper functions for employer name cleaning and salary normalization.
Used by both the data pipeline and offline cleaning scripts.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Company name aliases — maps all known variants to a canonical name
# ---------------------------------------------------------------------------
_NAME_MAPPINGS: dict[str, list[str]] = {
    "GOOGLE":        ["GOOGLE", "ALPHABET"],
    "META":          ["META PLATFORMS", "META", "FACEBOOK"],
    "AMAZON":        ["AMAZON.COM", "AMAZON WEB SERVICES", "AMAZON.COM SERVICES", "AMAZON"],
    "MICROSOFT":     ["MICROSOFT"],
    "APPLE":         ["APPLE"],
    "INFOSYS":       ["INFOSYS", "INFOSYS LIMITED", "INFOSYS BPM"],
    "TCS":           ["TATA CONSULTANCY", "TATA AMERICA", "TCS"],
    "COGNIZANT":     ["COGNIZANT", "COGNIZANT TECHNOLOGY"],
    "WIPRO":         ["WIPRO"],
    "ACCENTURE":     ["ACCENTURE"],
    "DELOITTE":      ["DELOITTE", "DELOITTE CONSULTING", "DELOITTE & TOUCHE"],
    "EY":            ["ERNST & YOUNG", "ERNST AND YOUNG", "E&Y"],
    "PWC":           ["PRICEWATERHOUSECOOPERS", "PWC"],
    "KPMG":          ["KPMG"],
    "IBM":           ["IBM", "INTERNATIONAL BUSINESS MACHINES"],
    "CAPGEMINI":     ["CAPGEMINI", "CAP GEMINI"],
    "HCL":           ["HCL TECHNOLOGIES", "HCL AMERICA"],
    "TECH MAHINDRA": ["TECH MAHINDRA", "MAHINDRA SATYAM"],
    "NVIDIA":        ["NVIDIA"],
    "INTEL":         ["INTEL"],
    "SALESFORCE":    ["SALESFORCE"],
    "ORACLE":        ["ORACLE"],
    "CISCO":         ["CISCO", "CISCO SYSTEMS"],
    "UBER":          ["UBER", "UBER TECHNOLOGIES"],
    "JPMORGAN CHASE":["JPMORGAN", "JP MORGAN", "JPMORGAN CHASE"],
    "GOLDMAN SACHS": ["GOLDMAN SACHS"],
    "MORGAN STANLEY":["MORGAN STANLEY"],
}

_SUFFIXES = [
    ", INC.", ", INC", " INC.", " INC",
    ", LLC", " LLC", ", LP", " LP", ", LLP", " LLP",
    " CORP.", " CORP", ", CORPORATION", " CORPORATION",
    " CO.", " CO", " LTD.", " LTD", " LIMITED",
    ", P.C.", " P.C.", ", PC", " PC",
    ", L.L.C.", " L.L.C.", ", INCORPORATED", " INCORPORATED",
]


def clean_employer_name(name: str) -> str:
    """
    Standardize a raw USCIS employer name to a canonical form.

    Steps:
        1. Upper-case and strip whitespace.
        2. Remove common legal suffixes (Inc, LLC, Corp, etc.).
        3. Map known aliases to a canonical company name.

    Args:
        name: Raw employer name string from the LCA disclosure file.

    Returns:
        Cleaned, canonical employer name (e.g. "META PLATFORMS INC." → "META").
    """
    if pd.isna(name):
        return "UNKNOWN"

    name = str(name).upper().strip()

    # Strip legal suffixes
    for suffix in _SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break  # only strip one suffix

    # Canonical mapping
    for canonical, variants in _NAME_MAPPINGS.items():
        for variant in variants:
            if variant in name:
                return canonical

    return name.strip()


def convert_to_annual_salary(wage, unit) -> float:
    """
    Normalise any wage representation to an annual USD figure.

    Handles: Year, Month, Bi-Weekly, Week, Hour pay rates.

    Args:
        wage: Numeric wage value (may contain "$" or "," in string form).
        unit: Pay-period string from the USCIS file (e.g. "Year", "Hour").

    Returns:
        Annual salary as float, or np.nan if conversion is not possible.
    """
    if pd.isna(wage) or pd.isna(unit):
        return np.nan

    # Clean common formatting artifacts
    try:
        wage = float(str(wage).replace(",", "").replace("$", ""))
    except (ValueError, TypeError):
        return np.nan

    unit = str(unit).upper()

    multipliers = {
        "YEAR":    1,
        "MONTH":   12,
        "BI-WEEK": 26,
        "WEEK":    52,
        "HOUR":    2080,
    }

    for key, multiplier in multipliers.items():
        if key in unit:
            return wage * multiplier

    return wage  # assume annual if unit is unknown
