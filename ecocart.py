#!/usr/bin/env python3
"""
ecoCart EDA (deterministic, reproducible) — Python 3.11+

Usage:
  python eda_ecocart.py --input ecoCart.csv --outdir eda_outputs

What it does:
- Deterministic cleaning (column normalization, currency parsing, date parsing)
- Data quality summary (missingness, duplicates)
- Descriptive stats (numeric + categorical)
- Plots exported as PNG (no interactive dependency)
- Quick baseline ML (if feasible): predict Sales from numeric features (R²/MAE)

Dependencies:
  pip install pandas numpy matplotlib scikit-learn
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, List

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
    "units_sold",  # in this dataset it appears as $-formatted string in some rows
}

def normalize_colname(name: str) -> str:
    """Normalize column names deterministically: strip, lower, replace spaces with underscores, remove non-alnum/_."""
    s = name.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def parse_currency_to_float(x) -> float:
    """
    Parse values like:
      ' $1,618.50 ' -> 1618.50
      '$-' or ' $-   ' -> 0.0
      '" $32,370.00 "' -> 32370.0
      888.00 -> 888.0
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)

    s = str(x).strip().replace('"', "")
    # Handle dash-like missing/zero
    if re.fullmatch(r"[$€£]?\s*-+\s*", s) or s in {"-", "$-", "€-", "£-"}:
        return 0.0

    # Remove currency symbols and spaces
    s = re.sub(r"[$€£\s]", "", s)

    # Some rows may have trailing/leading commas
    s = s.replace(",", "")

    # Empty after cleaning
    if s == "":
        return np.nan

    try:
        return float(s)
    except ValueError:
        # last-resort: remove non-numeric except dot and minus
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

    # Normalize column names
    df = raw.copy()
    df.columns = [normalize_colname(c) for c in df.columns]

    # Trim string columns
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()
            # turn literal 'nan' back to NaN if created by astype(str)
            df.loc[df[c].str.lower().isin(["nan", "none", "null"]), c] = np.nan

    # Identify a date column (common in this dataset: "date")
    date_col = None
    for cand in ["date", "order_date", "transaction_date"]:
        if cand in df.columns:
            date_col = cand
            break

    if date_col:
        # Dataset date appears like 01/01/2014 (dd/mm/yyyy in sample)
        # Try dayfirst first, fall back if needed.
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)

    # Parse currency-like columns (by name hints + by content)
    for c in df.columns:
        cname = c
        if cname in CURRENCY_LIKE_COLS_HINTS:
            df[c] = df[c].apply(parse_currency_to_float)

    # Sometimes the file has unexpected spaces in names that become e.g. "product" or "discount_band"
    # Ensure categorical columns are clean strings (strip again)
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype("string").str.strip()

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
        f.write("Shape\n")
        f.write("-----\n")
        f.write(f"Rows: {len(df):,}\nColumns: {df.shape[1]:,}\n\n")

        f.write("Columns & dtypes\n")
        f.write("--------------\n")
        f.write(df.dtypes.to_string() + "\n\n")

        f.write("Missing values (top 20)\n")
        f.write("-----------------------\n")
        miss = df.isna().sum().sort_values(ascending=False)
        f.write(miss.head(20).to_string() + "\n\n")

        f.write("Duplicate rows\n")
        f.write("--------------\n")
        f.write(f"{df.duplicated().sum():,}\n\n")

        # Numeric summary
        num = df.select_dtypes(include=[np.number])
        if not num.empty:
            f.write("Numeric describe\n")
            f.write("---------------\n")
            f.write(num.describe().T.to_string() + "\n\n")

        # Categorical summary
        cat = df.select_dtypes(include=["string", "object"])
        if not cat.empty:
            f.write("Categorical top values (top 10 per column)\n")
            f.write("------------------------------------------\n")
            for c in cat.columns[:20]:
                f.write(f"\n[{c}]\n")
                f.write(cat[c].value_counts(dropna=False).head(10).to_string() + "\n")


def export_tables(df: pd.DataFrame, outdir: str) -> None:
    # Missingness table
    miss = (
        df.isna().sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_pct=lambda x: (x["missing_count"] / len(df)) * 100.0)
        .sort_values("missing_count", ascending=False)
    )
    miss.to_csv(os.path.join(outdir, "missingness.csv"), index=True)

    # Duplicates info
    dup_count = int(df.duplicated().sum())
    pd.DataFrame({"duplicate_rows": [dup_count]}).to_csv(os.path.join(outdir, "duplicates.csv"), index=False)

    # Numeric summary
    num = df.select_dtypes(include=[np.number])
    if not num.empty:
        num.describe().T.to_csv(os.path.join(outdir, "numeric_describe.csv"))

    # Categorical summary (top frequencies)
    cat = df.select_dtypes(include=["string", "object"])
    if not cat.empty:
        rows = []
        for c in cat.columns:
            vc = df[c].value_counts(dropna=False).head(10)
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
    # Limit to top 8 numeric columns to keep output manageable
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
    plt.boxplot([df[c].dropna().values for c in cols], labels=cols, vert=True)
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
    # Prefer "sales" and "profit" if present
    candidates = [c for c in ["sales", "profit", "gross_sales", "cogs", "discounts"] if c in df.columns]
    if not candidates:
        return
    ts = df.dropna(subset=[date_col]).copy()
    if ts.empty:
        return
    ts = ts.sort_values(date_col)
    ts = ts.set_index(date_col)

    # monthly aggregation
    monthly = ts[candidates].resample("MS").sum(min_count=1)
    for c in candidates[:3]:
        plt.figure(figsize=(10, 4))
        plt.plot(monthly.index, monthly[c].values)
        plt.title(f"Monthly {c} (sum)")
        plt.xlabel("Month")
        plt.ylabel(c)
        save_fig(os.path.join(outdir, f"plot_timeseries_monthly_{c}.png"))

def plot_pivot_heatmap(df: pd.DataFrame, outdir: str) -> None:
    # Example: Sales by Country and Segment (common columns in this dataset)
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
    if "sales" not in df.columns:
        return

    # Select features (exclude target + date-ish columns)
    y = df["sales"].astype(float)
    X = df.drop(columns=["sales"])

    # Drop columns that are fully missing
    X = X.loc[:, X.notna().any()]

    # Identify numeric/categorical
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # If there are no features, skip
    if len(num_cols) + len(cat_cols) == 0:
        return

    # Simple pipeline: impute + one-hot + RF
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline(steps=[
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    # Train/test split (deterministic)
    mask = y.notna()
    X2, y2 = X.loc[mask], y.loc[mask]

    if len(y2) < 50:
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X2, y2, test_size=0.25, random_state=42
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Save metrics
    pd.DataFrame([{"mae": mae, "r2": r2, "n_train": len(y_train), "n_test": len(y_test)}]).to_csv(
        os.path.join(outdir, "baseline_ml_sales_metrics.csv"), index=False
    )

    # Plot predicted vs actual
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, s=10)
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Baseline ML: Predicted vs Actual (Sales)")
    save_fig(os.path.join(outdir, "plot_baseline_pred_vs_actual_sales.png"))


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Deterministic EDA for EcoCart dataset.")
    parser.add_argument("--input", required=True, help="Path to ecoCart.csv")
    parser.add_argument("--outdir", default="eda_outputs", help="Directory to save outputs")
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)
    res = load_and_clean(args.input)
    df = res.df

    # Save cleaned dataset for reproducibility
    df.to_csv(os.path.join(outdir, "ecoCart_cleaned.csv"), index=False)

    # Reports/tables
    write_text_report(df, outdir)
    export_tables(df, outdir)

    # Plots
    plot_missingness(df, outdir)
    plot_numeric_distributions(df, outdir)
    plot_boxplots(df, outdir)
    plot_correlation_heatmap(df, outdir)
    plot_time_series(df, outdir, res.date_col)
    plot_pivot_heatmap(df, outdir)

    # Baseline ML (optional)
    baseline_ml_sales(df, outdir)

    print(f"[OK] EDA complete. Outputs saved to: {os.path.abspath(outdir)}")

if __name__ == "__main__":
    main()
