# =========================
# FILE: run_eda.sh  (macOS/Linux/Git Bash)
# =========================
#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_eda.sh --input ecoCart.csv --outdir eda_outputs

VENV_DIR=".venv"
PY_BIN="$VENV_DIR/bin/python"

# Create venv if missing
if [ ! -d "$VENV_DIR" ]; then
  python -m venv "$VENV_DIR"
fi

# Upgrade pip and install requirements (idempotent)
"$PY_BIN" -m pip install --upgrade pip
"$PY_BIN" -m pip install -r requirements.txt

# Run EDA without needing to 'activate' the venv
"$PY_BIN" ecocart_eda.py "$@"


# =========================
# FILE: run_eda.bat  (Windows CMD)
# =========================
@echo off
setlocal enabledelayedexpansion

REM Usage:
REM   run_eda.bat --input ecoCart.csv --outdir eda_outputs

set VENV_DIR=.venv
set PY_BIN=%VENV_DIR%\Scripts\python.exe

if not exist "%VENV_DIR%" (
  py -3.11 -m venv "%VENV_DIR%"
  if errorlevel 1 (
    REM Fallback if py launcher is not available
    python -m venv "%VENV_DIR%"
  )
)

"%PY_BIN%" -m pip install --upgrade pip
"%PY_BIN%" -m pip install -r requirements.txt

"%PY_BIN%" ecocart_eda.py %*
endlocal


# =========================
# FILE: ecocart_eda.py
# =========================
#!/usr/bin/env python3
"""EcoCart EDA (deterministic, reproducible) — Python 3.11+

Why this design (AI-assisted, not AI-generated):
- Deterministic outputs (fixed random seeds, stable cleaning rules)
- Business-friendly plots + tables for a Foundations / AI-for-Business context
- Optional baseline ML to demonstrate feasibility (not production)

Usage:
  python ecocart_eda.py --input ecoCart.csv --outdir eda_outputs

Outputs:
  eda_outputs/
    ecoCart_cleaned.csv
    eda_report.txt
    missingness.csv
    duplicates.csv
    numeric_describe.csv
    categorical_top10.csv
    baseline_ml_sales_metrics.csv (if Sales exists)
    plot_*.png

Dependencies:
  pandas, numpy, matplotlib, scikit-learn
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# ---------------------------
# Deterministic utilities
# ---------------------------

CURRENCY_LIKE_COLS_HINTS = {
    "manufacturing_price",
    "sale_price",
    "gross_sales",
    "discounts",
    "sales",
    "cogs",
    "profit",
    "units_sold",  # in some variants appears as currency-like string
}


def normalize_colname(name: str) -> str:
    """Normalize column names deterministically."""
    s = name.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def parse_currency_to_float(x) -> float:
    """Parse currency-like strings into float. Treat $- as 0.0."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)

    s = str(x).strip().replace('"', "")

    # Treat dash as zero
    if re.fullmatch(r"[$€£]?\s*-+\s*", s) or s in {"-", "$-", "€-", "£-"}:
        return 0.0

    # Remove currency symbols/spaces
    s = re.sub(r"[$€£\s]", "", s)
    s = s.replace(",", "")

    if s == "":
        return np.nan

    try:
        return float(s)
    except ValueError:
        s2 = re.sub(r"[^0-9\.\-]", "", s)
        return float(s2) if s2 else np.nan


def ensure_outdir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir


def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------
# Cleaning + profiling
# ---------------------------

@dataclass
class CleanResult:
    df: pd.DataFrame
    raw_df: pd.DataFrame
    date_col: Optional[str]


def load_and_clean(csv_path: str) -> CleanResult:
    raw = pd.read_csv(csv_path)

    df = raw.copy()
    df.columns = [normalize_colname(c) for c in df.columns]

    # Strip strings, restore NaNs
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype("string").str.strip()
            df.loc[df[c].str.lower().isin(["nan", "none", "null"]), c] = pd.NA

    # Detect a date column
    date_col = None
    for cand in ["date", "order_date", "transaction_date"]:
        if cand in df.columns:
            date_col = cand
            break

    if date_col:
        # Try day-first (common in EU-style CSV)
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)

    # Parse currency-like by hints
    for c in df.columns:
        if c in CURRENCY_LIKE_COLS_HINTS:
            df[c] = df[c].apply(parse_currency_to_float)

    # Important sklearn compatibility: convert pd.NA to np.nan deterministically
    df = df.replace({pd.NA: np.nan})

    return CleanResult(df=df, raw_df=raw, date_col=date_col)


# ---------------------------
# Reporting
# ---------------------------

def write_text_report(df: pd.DataFrame, outdir: str, title: str = "EcoCart EDA Report") -> None:
    path = os.path.join(outdir, "eda_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")
        f.write(f"Generated at: {datetime.now().isoformat(timespec='seconds')}\n\n")

        f.write("Shape\n-----\n")
        f.write(f"Rows: {len(df):,}\nColumns: {df.shape[1]:,}\n\n")

        f.write("Columns & dtypes\n--------------\n")
        f.write(df.dtypes.to_string() + "\n\n")

        f.write("Missing values (top 20)\n-----------------------\n")
        miss = df.isna().sum().sort_values(ascending=False)
        f.write(miss.head(20).to_string() + "\n\n")

        f.write("Duplicate rows\n--------------\n")
        f.write(f"{df.duplicated().sum():,}\n\n")

        num = df.select_dtypes(include=[np.number])
        if not num.empty:
            f.write("Numeric describe\n---------------\n")
            f.write(num.describe().T.to_string() + "\n\n")

        cat = df.select_dtypes(exclude=[np.number])
        if not cat.empty:
            f.write("Categorical top values (top 10 per column)\n------------------------------------------\n")
            for c in cat.columns[:20]:
                f.write(f"\n[{c}]\n")
                f.write(df[c].astype("object").value_counts(dropna=False).head(10).to_string() + "\n")


def export_tables(df: pd.DataFrame, outdir: str) -> None:
    miss = (
        df.isna().sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_pct=lambda x: (x["missing_count"] / len(df)) * 100.0)
        .sort_values("missing_count", ascending=False)
    )
    miss.to_csv(os.path.join(outdir, "missingness.csv"), index=True)

    dup_count = int(df.duplicated().sum())
    pd.DataFrame({"duplicate_rows": [dup_count]}).to_csv(os.path.join(outdir, "duplicates.csv"), index=False)

    num = df.select_dtypes(include=[np.number])
    if not num.empty:
        num.describe().T.to_csv(os.path.join(outdir, "numeric_describe.csv"))

    cat = df.select_dtypes(exclude=[np.number])
    if not cat.empty:
        rows = []
        for c in cat.columns:
            vc = df[c].astype("object").value_counts(dropna=False).head(10)
            for k, v in vc.items():
                rows.append({"column": c, "value": str(k), "count": int(v)})
        pd.DataFrame(rows).to_csv(os.path.join(outdir, "categorical_top10.csv"), index=False)


# ---------------------------
# Plots
# ---------------------------

def plot_missingness(df: pd.DataFrame, outdir: str) -> None:
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]

    plt.figure(figsize=(10, 5))
    if miss.empty:
        plt.text(0.5, 0.5, "No missing values detected.", ha="center", va="center")
        plt.axis("off")
    else:
        plt.bar(miss.index[:30], miss.values[:30])
        plt.xticks(rotation=75, ha="right")
        plt.ylabel("Missing fraction")
        plt.title("Missingness (top columns)")

    save_fig(os.path.join(outdir, "plot_missingness.png"))


def plot_numeric_distributions(df: pd.DataFrame, outdir: str) -> None:
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return

    cols = list(num.columns)[:8]
    for c in cols:
        plt.figure(figsize=(8, 4))
        x = df[c].dropna()
        if x.empty:
            plt.text(0.5, 0.5, f"No data for {c}", ha="center", va="center")
            plt.axis("off")
        else:
            plt.hist(x, bins=40)
            plt.title(f"Distribution: {c}")
            plt.xlabel(c)
            plt.ylabel("Count")
        save_fig(os.path.join(outdir, f"plot_hist_{c}.png"))


def plot_boxplots(df: pd.DataFrame, outdir: str) -> None:
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return

    cols = list(num.columns)[:8]
    plt.figure(figsize=(10, 5))

    # Matplotlib 3.9+ uses tick_labels instead of labels
    plt.boxplot([df[c].dropna().values for c in cols], tick_labels=cols, vert=True)

    plt.xticks(rotation=45, ha="right")
    plt.title("Boxplots (selected numeric columns)")
    save_fig(os.path.join(outdir, "plot_boxplots.png"))


def plot_correlation_heatmap(df: pd.DataFrame, outdir: str) -> None:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return

    corr = num.corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(corr.values, aspect="auto")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=75, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation heatmap (numeric)")
    plt.colorbar()
    save_fig(os.path.join(outdir, "plot_corr_heatmap.png"))


def plot_time_series(df: pd.DataFrame, outdir: str, date_col: Optional[str]) -> None:
    if not date_col or date_col not in df.columns:
        return

    candidates = [c for c in ["sales", "profit", "gross_sales", "cogs", "discounts"] if c in df.columns]
    if not candidates:
        return

    ts = df.dropna(subset=[date_col]).copy()
    if ts.empty:
        return

    ts = ts.sort_values(date_col).set_index(date_col)
    monthly = ts[candidates].resample("MS").sum(min_count=1)

    for c in candidates[:3]:
        plt.figure(figsize=(10, 4))
        plt.plot(monthly.index, monthly[c].values)
        plt.title(f"Monthly {c} (sum)")
        plt.xlabel("Month")
        plt.ylabel(c)
        save_fig(os.path.join(outdir, f"plot_timeseries_monthly_{c}.png"))


def plot_pivot_heatmap(df: pd.DataFrame, outdir: str) -> None:
    if not all(c in df.columns for c in ["country", "segment", "sales"]):
        return

    pivot = (
        df.pivot_table(index="country", columns="segment", values="sales", aggfunc="sum", fill_value=0)
        .sort_index()
    )

    plt.figure(figsize=(10, 5))
    plt.imshow(pivot.values, aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Heatmap: Sales (sum) by Country x Segment")
    plt.colorbar()
    save_fig(os.path.join(outdir, "plot_heatmap_sales_country_segment.png"))


# ---------------------------
# Baseline ML (optional)
# ---------------------------

def baseline_ml_sales(df: pd.DataFrame, outdir: str) -> None:
    """Optional business-facing baseline: predict Sales using a simple RF pipeline."""
    if "sales" not in df.columns:
        return

    # Ensure sklearn-safe missing values
    df2 = df.replace({pd.NA: np.nan}).copy()

    y = pd.to_numeric(df2["sales"], errors="coerce")
    X = df2.drop(columns=["sales"])

    X = X.loc[:, X.notna().any()]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Force categoricals to plain object dtype
    for c in cat_cols:
        X[c] = X[c].astype("object")

    if len(num_cols) + len(cat_cols) == 0:
        return

    mask = y.notna()
    X2, y2 = X.loc[mask], y.loc[mask]

    if len(y2) < 50:
        return

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imp", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X2, y2, test_size=0.25, random_state=42
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    pd.DataFrame(
        [{"mae": mae, "r2": r2, "n_train": len(y_train), "n_test": len(y_test)}]
    ).to_csv(os.path.join(outdir, "baseline_ml_sales_metrics.csv"), index=False)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, s=10)
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Baseline ML: Predicted vs Actual (Sales)")
    save_fig(os.path.join(outdir, "plot_baseline_pred_vs_actual_sales.png"))


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic EDA for EcoCart dataset.")
    parser.add_argument("--input", required=True, help="Path to EcoCart CSV")
    parser.add_argument("--outdir", default="eda_outputs", help="Directory to save outputs")
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)
    res = load_and_clean(args.input)
    df = res.df

    # Save cleaned dataset
    df.to_csv(os.path.join(outdir, "ecoCart_cleaned.csv"), index=False)

    # Reports
    write_text_report(df, outdir)
    export_tables(df, outdir)

    # Plots
    plot_missingness(df, outdir)
    plot_numeric_distributions(df, outdir)
    plot_boxplots(df, outdir)
    plot_correlation_heatmap(df, outdir)
    plot_time_series(df, outdir, res.date_col)
    plot_pivot_heatmap(df, outdir)

    # Optional baseline ML
    baseline_ml_sales(df, outdir)

    print(f"[OK] EDA complete. Outputs saved to: {os.path.abspath(outdir)}")


if __name__ == "__main__":
    main()
