#!/usr/bin/env python3
# exploratory_models.py — EcoCart exploratory modeling + timing
# See previous message for full documentation (kept concise here for performance)

from __future__ import annotations
import argparse, os, time
from datetime import datetime
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def ensure_outdir(d): os.makedirs(d, exist_ok=True); return d
def save_fig(p): plt.tight_layout(); plt.savefig(p, dpi=200, bbox_inches="tight"); plt.close()
def now(): return time.perf_counter()
def run_id(): return datetime.now().strftime("%Y%m%d_%H%M%S")
def tag(t): return str(t).lower().replace(" ", "_")

def append_csv(path, rows):
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)

def load(path):
    return pd.read_csv(path).replace({pd.NA: np.nan})

def split_cols(df, target):
    feats = [c for c in df.columns if c != target]
    num = df[feats].select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in feats if c not in num]
    return num, cat

def rf(df, target, out, rid):
    y = pd.to_numeric(df[target], errors="coerce")
    X = df.drop(columns=[target])
    m = y.notna(); X, y = X[m], y[m]
    num, cat = split_cols(df[m], target)
    pre = ColumnTransformer([
        ("n", Pipeline([("i", SimpleImputer(strategy="median"))]), num),
        ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                        ("o", OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])
    model = Pipeline([("p", pre), ("m", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    t0 = now(); model.fit(Xtr, ytr); ttr = now()-t0
    t1 = now(); pr = model.predict(Xte); tinf = now()-t1
    mae, r2 = mean_absolute_error(yte, pr), r2_score(yte, pr)
    plt.figure(); plt.scatter(yte, pr, s=10); save_fig(os.path.join(out, f"rf_pred_{tag(target)}_{rid}.png"))
    return {"run_id": rid, "model": "RF", "target": target, "mae": mae, "r2": r2,
            "train_time_sec": ttr, "inference_time_sec": tinf}

def mlp(df, target, out, rid):
    y = pd.to_numeric(df[target], errors="coerce")
    X = df.drop(columns=[target])
    m = y.notna(); X, y = X[m], y[m]
    num, cat = split_cols(df[m], target)
    pre = ColumnTransformer([
        ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
        ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                        ("o", OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])
    model = Pipeline([("p", pre), ("m", MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500,
                                                     early_stopping=True, random_state=42))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    t0 = now(); model.fit(Xtr, ytr); ttr = now()-t0
    t1 = now(); pr = model.predict(Xte); tinf = now()-t1
    mae, r2 = mean_absolute_error(yte, pr), r2_score(yte, pr)
    plt.figure(); plt.scatter(yte, pr, s=10); save_fig(os.path.join(out, f"mlp_pred_{tag(target)}_{rid}.png"))
    return {"run_id": rid, "model": "MLP", "target": target, "mae": mae, "r2": r2,
            "train_time_sec": ttr, "inference_time_sec": tinf}

def kmeans(df, out, kmin, kmax, rid):
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    X = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]).fit_transform(df[num])
    rows = []
    for k in range(kmin, kmax+1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        t0 = now(); lbl = km.fit_predict(X); ttr = now()-t0
        sil = silhouette_score(X, lbl) if k > 1 else np.nan
        rows.append({"run_id": rid, "model": "KMeans", "k": k, "silhouette": sil, "train_time_sec": ttr})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="exploratory_outputs")
    ap.add_argument("--target", default="units_sold")
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=8)
    a = ap.parse_args()

    out = ensure_outdir(a.outdir)
    rid = run_id()
    df = load(a.input)

    rows = []
    rows.append(rf(df, a.target, out, rid))
    rows.append(mlp(df, a.target, out, rid))
    rows.extend(kmeans(df, out, a.kmin, a.kmax, rid))

    append_csv(os.path.join(out, "model_timing.csv"), rows)
    print(f"[OK] Run {rid} appended to model_timing.csv")

if __name__ == "__main__":
    main()
