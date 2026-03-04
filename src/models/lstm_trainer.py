"""
H-1B LSTM Training Pipeline
=============================
Trains a PyTorch LSTM on multi-year H-1B company data to predict
sponsorship trends across fiscal years.

Architecture:
    Input:  3-year time-series per company
            Features per year: [total_filings, avg_salary, median_salary, sponsorship_score]
    LSTM:   2 layers, hidden_size=64, dropout=0.2
    Output: Predicted sponsorship_score for the next year → trend direction

Data required:
    data/h1b_multiyear.csv  — produced by scripts/downloader.py

Outputs:
    notebooks/h1b_lstm_model.pt     — saved LSTM weights
    data/lstm_predictions.csv       — per-company trend predictions

Usage (standalone):
    python -m src.models.lstm_trainer

Usage (Python):
    from src.models.lstm_trainer import H1BLSTMTrainer
    trainer = H1BLSTMTrainer()
    predictions = trainer.train_and_predict()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT           = Path(__file__).parent.parent.parent
_MULTIYEAR_CSV  = _ROOT / "data"   / "h1b_multiyear.csv"
_MODEL_OUT      = _ROOT / "notebooks" / "h1b_lstm_model.pt"
_PREDS_OUT      = _ROOT / "data"   / "lstm_predictions.csv"

# Minimum number of years of data a company must appear in to be included
_MIN_YEARS = 2

# ---------------------------------------------------------------------------
# Feature columns used as LSTM inputs
# ---------------------------------------------------------------------------
_FEATURE_COLS     = ["total_filings", "avg_salary", "median_salary", "sponsorship_score"]
_TARGET_COL       = "sponsorship_score"


# ---------------------------------------------------------------------------
# PyTorch model definition
# ---------------------------------------------------------------------------

def _build_model(input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.2):
    """Build and return an LSTM regression model using PyTorch."""
    import torch
    import torch.nn as nn

    class SponsorshipLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):   # x: (batch, seq_len, input_size)
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)  # use last timestep

    return SponsorshipLSTM()


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _prepare_sequences(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    Pivot the multi-year DataFrame into LSTM sequences.

    Returns:
        X:           np.ndarray (N_companies, seq_len, n_features)
        y:           np.ndarray (N_companies,) — last-year sponsorship score as target
        companies:   List[str] of company names in same order as X, y
        feat_mean:   Feature-wise means (for normalisation)
        feat_std:    Feature-wise stds  (for normalisation)
    """
    years = sorted(df["fiscal_year"].unique())
    if len(years) < _MIN_YEARS:
        raise ValueError(
            f"Need at least {_MIN_YEARS} years of data, got {len(years)}: {years}"
        )

    # Keep only companies that appear in ALL years
    company_counts = df.groupby("company")["fiscal_year"].nunique()
    valid_companies = company_counts[company_counts >= _MIN_YEARS].index.tolist()
    df = df[df["company"].isin(valid_companies)].copy()

    sequences, targets, company_names = [], [], []

    for company, grp in df.groupby("company"):
        grp = grp.sort_values("fiscal_year")
        feats = grp[_FEATURE_COLS].values.astype(float)  # (n_years, n_features)

        # Pad / truncate to consistent seq length
        seq_len = len(years)
        if len(feats) < seq_len:
            # Pad at the front with the first known year's values
            pad = np.tile(feats[0], (seq_len - len(feats), 1))
            feats = np.vstack([pad, feats])

        sequences.append(feats[-seq_len:])            # (seq_len, n_features)
        targets.append(float(grp[_TARGET_COL].iloc[-1]))   # last year as label
        company_names.append(company)

    X = np.array(sequences, dtype=np.float32)  # (N, seq_len, n_features)
    y = np.array(targets,   dtype=np.float32)  # (N,)

    # Normalise features
    feat_mean = X.mean(axis=(0, 1), keepdims=True)
    feat_std  = X.std(axis=(0, 1),  keepdims=True) + 1e-8
    X = (X - feat_mean) / feat_std

    return X, y, company_names, feat_mean.squeeze(), feat_std.squeeze()


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class H1BLSTMTrainer:
    """
    Trains a PyTorch LSTM on multi-year H-1B company data.

    Usage:
        trainer = H1BLSTMTrainer()
        predictions = trainer.train_and_predict()
    """

    def __init__(
        self,
        hidden_size: int   = 64,
        num_layers:  int   = 2,
        epochs:      int   = 200,
        lr:          float = 1e-3,
        dropout:     float = 0.2,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.epochs      = epochs
        self.lr          = lr
        self.dropout     = dropout

    # ------------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        """Load the multi-year CSV produced by scripts/downloader.py."""
        if not _MULTIYEAR_CSV.exists():
            raise FileNotFoundError(
                f"Multi-year CSV not found at {_MULTIYEAR_CSV}.\n"
                "Run scripts/downloader.py first:\n"
                "    python scripts/downloader.py --years 2022 2023 2024"
            )
        df = pd.read_csv(_MULTIYEAR_CSV)
        print(f"✅ Loaded {len(df):,} rows from {_MULTIYEAR_CSV.name}")
        print(f"   Fiscal years: {sorted(df['fiscal_year'].unique().tolist())}")
        return df

    # ------------------------------------------------------------------

    def train_and_predict(self, verbose: bool = True) -> pd.DataFrame:
        """
        Train the LSTM on multi-year data and return per-company trend predictions.

        Returns:
            DataFrame with columns:
                company, last_year_score, predicted_next_score, trend
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError(
                "PyTorch is required: pip install torch>=2.1.0"
            )

        df = self.load_data()
        X, y, companies, feat_mean, feat_std = _prepare_sequences(df)

        n_features = X.shape[2]
        model = _build_model(n_features, self.hidden_size, self.num_layers, self.dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X)
        y_t = torch.tensor(y)

        # Normalise targets to 0-1 range
        y_min, y_max = float(y_t.min()), float(y_t.max())
        y_norm = (y_t - y_min) / (y_max - y_min + 1e-8)

        if verbose:
            print(f"\n🧠 Training LSTM")
            print(f"   Companies:   {len(companies)}")
            print(f"   Seq length:  {X.shape[1]} years")
            print(f"   Features:    {_FEATURE_COLS}")
            print(f"   Architecture: LSTM({n_features} → {self.hidden_size} × {self.num_layers}L)")
            print()

        model.train()
        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            pred = model(X_t)
            loss = criterion(pred, y_norm)
            loss.backward()
            optimizer.step()

            if verbose and epoch % 50 == 0:
                print(f"   Epoch {epoch:>4}/{self.epochs}  loss={loss.item():.5f}")

        # Save model
        _MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "hidden_size": self.hidden_size,
            "num_layers":  self.num_layers,
            "n_features":  n_features,
            "feat_mean":   feat_mean.tolist(),
            "feat_std":    feat_std.tolist(),
            "y_min":       y_min,
            "y_max":       y_max,
        }, _MODEL_OUT)
        print(f"\n💾 Saved LSTM model → {_MODEL_OUT}")

        # Generate predictions (predict for "next year" using last-year features)
        model.eval()
        with torch.no_grad():
            pred_norm = model(X_t).numpy()

        pred_scores = pred_norm * (y_max - y_min) + y_min
        actual_scores = y

        records = []
        for company, actual, predicted in zip(companies, actual_scores, pred_scores):
            delta = float(predicted) - float(actual)
            if delta > 2:
                trend = "📈 Rising"
            elif delta < -2:
                trend = "📉 Falling"
            else:
                trend = "➡️ Stable"

            records.append({
                "company":              company,
                "last_year_score":      round(float(actual), 1),
                "predicted_next_score": round(float(predicted), 1),
                "trend":                trend,
            })

        preds_df = pd.DataFrame(records).sort_values("predicted_next_score", ascending=False)
        preds_df.to_csv(_PREDS_OUT, index=False)
        print(f"💾 Saved predictions → {_PREDS_OUT}")

        if verbose:
            print(f"\n📊 Trend Summary:")
            for trend_label, count in preds_df["trend"].value_counts().items():
                print(f"   {trend_label}:  {count} companies")

        return preds_df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    trainer = H1BLSTMTrainer(epochs=200)
    results = trainer.train_and_predict(verbose=True)
    print("\n" + results.to_string(index=False))
