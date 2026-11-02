"""
Reproduce the benchmark's published protocol (Random Forest / bagging, random split)
and compare feature-set choices. Goal: confirm we match literature (~99%) under the
same protocol they use, and quantify honest generalization under a time-based split.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from data_loader import load_data, FEATURE_GROUPS
from train_detection_epoch import build_epoch_features

def rf_eval(X_tr, y_tr, X_te, y_te, tag):
    m = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42)
    m.fit(X_tr, y_tr)
    yp = m.predict(X_te)
    acc = accuracy_score(y_te, yp); f1 = f1_score(y_te, yp, average="macro")
    print(f"  [{tag}] acc={acc*100:.2f}%  macroF1={f1:.4f}")
    return acc, f1

if __name__ == "__main__":
    df = load_data()

    print("\n### PER-CHANNEL (matches literature's 510K-sample, 13-feature setup) ###")
    le = LabelEncoder().fit(df["class"])
    X = df[FEATURE_GROUPS].values; y = le.transform(df["class"].values)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    rf_eval(Xtr, ytr, Xte, yte, "RandomForest, per-channel, all 13 feats, RANDOM split")

    print("\n### EPOCH-LEVEL multi-channel fusion ###")
    epoch = build_epoch_features(df)
    feat_cols = [c for c in epoch.columns if c not in ("epoch_id", "Output", "class")]
    le2 = LabelEncoder().fit(epoch["class"])
    Xe = epoch[feat_cols].values; ye = le2.transform(epoch["class"].values)
    Xtr, Xte, ytr, yte = train_test_split(Xe, ye, test_size=0.25, random_state=42, stratify=ye)
    rf_eval(Xtr, ytr, Xte, yte, "RandomForest, epoch-level, RANDOM split")

    # honest time-based split for reference
    es = epoch.sort_values("epoch_id").reset_index(drop=True)
    n = len(es); block = (np.arange(n) * 5 // n); tm = block == 4
    Xa = es[feat_cols].values; ya = le2.transform(es["class"].values)
    rf_eval(Xa[~tm], ya[~tm], Xa[tm], ya[tm], "RandomForest, epoch-level, TIME-based split (honest)")
