"""
H-1B Sponsorship Predictor
===========================
Predicts H-1B visa sponsorship likelihood from a candidate's profile.

The predictor loads a pre-trained scikit-learn model (trained by ml_trainer.py)
when available, and falls back to a transparent scoring model for demo mode.

Inputs:  job_role, salary, state, company_size, education_level
Output:  PredictionResult — likelihood (HIGH/MEDIUM/LOW), confidence, factors, tips
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Container for a single sponsorship prediction."""
    likelihood:      str          # "HIGH" | "MEDIUM" | "LOW"
    confidence:      float        # 0–100
    factors:         Dict[str, str]
    recommendations: List[str]
    model_used:      str = "scoring"   # "sklearn" | "scoring"
    lstm_trend:      Optional[str] = None  # e.g. "📈 Rising" | "📉 Falling" | "➡️ Stable"


# ---------------------------------------------------------------------------
# Feature encoding helpers (shared between predictor and trainer)
# ---------------------------------------------------------------------------

JOB_ROLES = [
    "Software Engineer",
    "Data Scientist / Analyst",
    "Manager / Lead",
    "Consultant",
    "Research Scientist",
    "Other",
]

STATES = ["CA", "WA", "NY", "TX", "NJ", "MA", "IL", "Other"]

COMPANY_SIZES = ["Enterprise", "Large", "Medium", "Small"]

EDUCATION_LEVELS = ["PhD", "Masters", "Bachelors"]

SALARY_BINS = [0, 80_000, 120_000, 160_000, float("inf")]   # → 0,1,2,3
SALARY_LABELS = ["Low", "Medium", "High", "Very High"]


def encode_features(
    job_role: str,
    salary: float,
    state: str,
    company_size: str,
    education: str,
) -> np.ndarray:
    """
    Convert human-readable profile inputs into a 1-D numeric feature vector.

    Encoding:
        - Categorical fields → ordinal index (unknown → last category)
        - Salary → bin index (0 Low … 3 Very High)

    Returns:
        np.ndarray shape (5,)
    """
    def _idx(choices: list, value: str) -> int:
        try:
            return choices.index(value)
        except ValueError:
            return len(choices) - 1   # fallback to "Other" / last

    salary_bin = int(np.digitize(salary, SALARY_BINS[1:-1]))  # 0-3

    return np.array([
        _idx(JOB_ROLES,        job_role),
        salary_bin,
        _idx(STATES,           state),
        _idx(COMPANY_SIZES,    company_size),
        _idx(EDUCATION_LEVELS, education),
    ], dtype=float).reshape(1, -1)


# ---------------------------------------------------------------------------
# Main predictor class
# ---------------------------------------------------------------------------

# Where ml_trainer.py saves the best sklearn model
_MODEL_PATH = Path(__file__).parent.parent.parent / "notebooks" / "h1b_sponsorship_model.pkl"

# Where lstm_trainer.py saves per-company trend predictions
_LSTM_PREDS_PATH = Path(__file__).parent.parent.parent / "data" / "lstm_predictions.csv"


class H1BSponsorshipPredictor:
    """
    H-1B sponsorship likelihood predictor.

    Loads a pre-trained scikit-learn model when ``notebooks/h1b_sponsorship_model.pkl``
    exists (produced by :class:`src.models.ml_trainer.H1BModelTrainer`).
    Falls back transparently to a weighted scoring model when no pkl is found.
    """

    # ------------------------------------------------------------------
    # Scoring-model weights (used in fallback / demo mode)
    # ------------------------------------------------------------------
    _WEIGHTS = {
        "job_role":     {"Software Engineer": 25, "Data Scientist / Analyst": 15,
                         "Manager / Lead": 22, "Consultant": 28,
                         "Research Scientist": 35, "Other": 15},
        "salary":       {"Low": 5, "Medium": 15, "High": 25, "Very High": 30},
        "state":        {"CA": 18, "WA": 30, "NY": 15, "TX": 20,
                         "NJ": 25, "MA": 18, "IL": 15, "Other": 12},
        "company_size": {"Enterprise": 30, "Large": 25, "Medium": 18, "Small": 10},
        "education":    {"PhD": 25, "Masters": 22, "Bachelors": 15},
    }
    _MAX_SCORE = 35 + 30 + 25 + 30 + 25   # 145

    def __init__(self) -> None:
        self._sklearn_model   = None
        self._label_order     = ["LOW", "MEDIUM", "HIGH"]
        self._lstm_predictions: Dict[str, str] = {}
        self._try_load_sklearn()
        self._try_load_lstm_predictions()

    def _try_load_sklearn(self) -> None:
        """Attempt to load a pre-trained sklearn model from disk."""
        try:
            import joblib
            if _MODEL_PATH.exists():
                self._sklearn_model = joblib.load(_MODEL_PATH)
                print(f"✅ Loaded trained sklearn model from {_MODEL_PATH}")
        except Exception as e:
            print(f"ℹ️  No pre-trained model found ({e}). Using scoring model.")

    def _try_load_lstm_predictions(self) -> None:
        """Load LSTM-generated trend predictions from CSV if available."""
        try:
            if _LSTM_PREDS_PATH.exists():
                import pandas as _pd
                preds = _pd.read_csv(_LSTM_PREDS_PATH)
                self._lstm_predictions = dict(
                    zip(preds["company"].str.upper(), preds["trend"])
                )
                print(f"✅ Loaded LSTM trends for {len(self._lstm_predictions)} companies")
        except Exception as e:
            print(f"ℹ️  LSTM predictions unavailable ({e})")

    def get_lstm_trend(self, company_name: str) -> Optional[str]:
        """
        Return the LSTM-predicted trend for a company, or None if not available.

        Args:
            company_name: Company name (case-insensitive).

        Returns:
            Trend string (e.g. "📈 Rising") or None.
        """
        return self._lstm_predictions.get(company_name.upper())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        job_role:     str,
        salary:       float,
        state:        str,
        company_size: str,
        education:    str,
    ) -> PredictionResult:
        """
        Predict H-1B sponsorship likelihood for a candidate profile.

        Args:
            job_role:     One of the defined JOB_ROLES categories.
            salary:       Expected annual salary in USD.
            state:        US state abbreviation (e.g. "CA").
            company_size: "Enterprise" | "Large" | "Medium" | "Small"
            education:    "PhD" | "Masters" | "Bachelors"

        Returns:
            :class:`PredictionResult` with likelihood, confidence, and tips.
        """
        if self._sklearn_model is not None:
            return self._predict_sklearn(job_role, salary, state, company_size, education)
        return self._predict_scoring(job_role, salary, state, company_size, education)

    # ------------------------------------------------------------------
    # sklearn path
    # ------------------------------------------------------------------

    def _predict_sklearn(self, job_role, salary, state, company_size, education) -> PredictionResult:
        X = encode_features(job_role, salary, state, company_size, education)
        try:
            proba      = self._sklearn_model.predict_proba(X)[0]   # [P_low, P_med, P_high]
            pred_idx   = int(np.argmax(proba))
            likelihood = self._label_order[pred_idx]
            confidence = float(proba[pred_idx] * 100)
        except Exception:
            # Fallback if model doesn't support predict_proba
            pred       = self._sklearn_model.predict(X)[0]
            likelihood = str(pred)
            confidence = 70.0

        factors         = self._build_factors(job_role, salary, state, company_size, education)
        recommendations = self._generate_recommendations(job_role, salary, state, company_size, education, confidence)
        return PredictionResult(likelihood, confidence, factors, recommendations, model_used="sklearn")

    # ------------------------------------------------------------------
    # Scoring-model fallback path
    # ------------------------------------------------------------------

    def _predict_scoring(self, job_role, salary, state, company_size, education) -> PredictionResult:
        sal_label = SALARY_LABELS[int(np.digitize(salary, SALARY_BINS[1:-1]))]

        raw  = (
            self._WEIGHTS["job_role"].get(job_role, 15)
            + self._WEIGHTS["salary"].get(sal_label, 15)
            + self._WEIGHTS["state"].get(state, 12)
            + self._WEIGHTS["company_size"].get(company_size, 15)
            + self._WEIGHTS["education"].get(education, 15)
        )
        confidence = (raw / self._MAX_SCORE) * 100

        if confidence >= 70:
            likelihood = "HIGH"
        elif confidence >= 50:
            likelihood = "MEDIUM"
        else:
            likelihood = "LOW"

        factors         = self._build_factors(job_role, salary, state, company_size, education)
        recommendations = self._generate_recommendations(job_role, salary, state, company_size, education, confidence)
        return PredictionResult(likelihood, confidence, factors, recommendations, model_used="scoring")

    # ------------------------------------------------------------------
    # Shared explanation helpers
    # ------------------------------------------------------------------

    def _build_factors(self, job_role, salary, state, company_size, education) -> Dict[str, str]:
        sal_label = SALARY_LABELS[int(np.digitize(salary, SALARY_BINS[1:-1]))]

        salary_msg = {
            "Very High": f"${salary:,.0f} is well above average — strong sponsorship indicator.",
            "High":      f"${salary:,.0f} is above average — competitive for H-1B roles.",
            "Medium":    f"${salary:,.0f} meets typical H-1B prevailing wage requirements.",
            "Low":       f"${salary:,.0f} may face prevailing wage challenges.",
        }
        size_msg = {
            "Enterprise": "Enterprise companies (2000+ filings) have mature, reliable H-1B programs.",
            "Large":      "Large companies (500–2000 filings) regularly sponsor H-1B visas.",
            "Medium":     "Medium companies (100–500 filings) sponsor selectively for key roles.",
            "Small":      "Small companies (<100 filings) sponsor occasionally.",
        }
        edu_msg = {
            "PhD":      "PhD holders qualify for specialised roles and may also pursue O-1 visas.",
            "Masters":  "Master's degree unlocks the extra 20,000 H-1B Master's cap lottery pool.",
            "Bachelors":"Bachelor's degree is eligible for the standard 65,000-visa cap.",
        }

        return {
            "job_role":     f"{job_role} roles are among the most common H-1B categories.",
            "salary":       salary_msg.get(sal_label, "Salary is within typical range."),
            "state":        f"{state} — {'high' if state in ('CA','WA','NY','TX') else 'moderate'} H-1B activity.",
            "company_size": size_msg.get(company_size, "Company size affects sponsorship likelihood."),
            "education":    edu_msg.get(education, "Education level affects visa eligibility."),
        }

    def _generate_recommendations(
        self, job_role, salary, state, company_size, education, score
    ) -> List[str]:
        tips: List[str] = []
        if score < 50:
            tips.append("Target Enterprise or Large companies with established H-1B pipelines.")
        if salary < 100_000:
            tips.append("Roles above $100K have stronger sponsorship indicators (prevailing wage).")
        if company_size in ("Small", "Medium"):
            tips.append("Enterprise / Large companies have more consistent sponsorship track records.")
        if state not in ("CA", "WA", "NY", "TX"):
            tips.append("CA, WA, NY, and TX account for 65%+ of all H-1B filings.")
        if education == "Bachelors":
            tips.append("A Master's degree qualifies you for the extra 20,000-visa lottery pool.")
        if job_role == "Other":
            tips.append("Software Engineering and Data Science roles have the highest sponsorship rates.")
        tips.append("Apply to multiple companies — lottery selection is ~25–30%.")
        tips.append("Have backup plans: STEM OPT extension, O-1, or EB-1.")
        return tips[:5]

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance weights for the bar chart in the UI."""
        total = self._MAX_SCORE
        return {
            "Job Role":     35 / total * 100,
            "Salary":       30 / total * 100,
            "State":        25 / total * 100,
            "Company Size": 30 / total * 100,
            "Education":    25 / total * 100,
        }
