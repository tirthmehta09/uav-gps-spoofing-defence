"""
Push detection accuracy as high as LEGITIMATELY possible, and diagnose the gap to
the literature's 99.99%. Every experiment is labelled with its evaluation protocol
so we can see which numbers are honest generalisation vs which rely on leakage.

Experiments:
  A. Enhanced epoch features (mean/std/min/max/range/cv per group + n_active).
  B. Multiclass: tuned XGBoost, ExtraTrees, and a soft-vote ensemble.
  C. Binary (spoofed vs authentic): the operational detection question.
  D. Protocol probe: random hold-out vs 5-fold CV vs GROUPED-by-epoch CV.
     If random/CV >> grouped, the high number is leakage from temporal autocorrelation.
"""
import os
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

from data_loader import load_data, FEATURE_GROUPS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RS = 42


def build_features(df):
    g = df.groupby("epoch_id")
    agg = g[FEATURE_GROUPS].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{f}_{a}" for f, a in agg.columns]
    for f in FEATURE_GROUPS:                       # physically-meaningful spread signatures
        agg[f"{f}_range"] = agg[f"{f}_max"] - agg[f"{f}_min"]
        agg[f"{f}_cv"] = agg[f"{f}_std"] / (agg[f"{f}_mean"].abs() + 1e-6)
    agg["n_active"] = g["channel"].count()
    agg = agg.fillna(0.0)
    cmap = {0: "authentic", 1: "simplistic", 2: "intermediate", 3: "sophisticated"}
    agg["class"] = g["Output"].max().map(cmap)
    agg = agg.reset_index()
    feats = [c for c in agg.columns if c not in ("epoch_id", "class")]
    return agg, feats


def xgb(nclass):
    # Identical configuration to run_framework.py so hold-out numbers are consistent
    # across the whole manuscript (same features, same split, same model -> same result).
    return XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.15,
                         subsample=0.9, colsample_bytree=0.9,
                         tree_method="hist", n_jobs=-1, random_state=RS,
                         eval_metric=("logloss" if nclass == 2 else "mlogloss"))


def report(y_true, y_pred, tag):
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    print(f"  [{tag}] acc={acc*100:.2f}%  macroF1={mf1:.4f}")
    return {"acc": float(acc), "macro_f1": float(mf1)}


def main():
    out = {}
    df = load_data()
    epoch, feats = build_features(df)
    print(f"\nEpochs={len(epoch)}  enhanced-features={len(feats)}")
    # Time-block groups (5 contiguous chronological blocks) for a leakage-free CV:
    # grouping by epoch_id would be pointless since each epoch is already one sample.
    order = np.argsort(epoch["epoch_id"].values)
    tblock = np.empty(len(epoch), dtype=int)
    tblock[order] = (np.arange(len(epoch)) * 5 // len(epoch))
    groups = tblock

    # ---- MULTICLASS ----
    print("\n### MULTICLASS (4 classes) ###")
    ym = LabelEncoder().fit_transform(epoch["class"].values)
    X = epoch[feats].values
    Xtr, Xte, ytr, yte = train_test_split(X, ym, test_size=0.25, random_state=RS, stratify=ym)

    m = xgb(4); m.fit(Xtr, ytr)
    out["mc_xgb_holdout"] = report(yte, m.predict(Xte), "XGBoost enhanced, holdout")

    et = ExtraTreesClassifier(n_estimators=400, n_jobs=-1, random_state=RS)
    et.fit(Xtr, ytr)
    out["mc_extratrees_holdout"] = report(yte, et.predict(Xte), "ExtraTrees, holdout")

    ens = VotingClassifier([("xgb", xgb(4)),
                            ("et", ExtraTreesClassifier(n_estimators=400, n_jobs=-1, random_state=RS)),
                            ("rf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RS))],
                           voting="soft", n_jobs=-1)
    ens.fit(Xtr, ytr)
    out["mc_ensemble_holdout"] = report(yte, ens.predict(Xte), "Soft-vote ensemble, holdout")

    # protocol probe (multiclass, XGBoost)
    skf = StratifiedKFold(5, shuffle=True, random_state=RS)
    cv = cross_val_score(xgb(4), X, ym, cv=skf, scoring="accuracy", n_jobs=1)
    out["mc_xgb_5fold_random"] = {"acc_mean": float(cv.mean()), "acc_std": float(cv.std())}
    print(f"  [XGBoost 5-fold RANDOM CV] acc={cv.mean()*100:.2f}% +/- {cv.std()*100:.2f}")

    gkf = GroupKFold(5)
    cvg = cross_val_score(xgb(4), X, ym, cv=gkf, groups=groups, scoring="accuracy", n_jobs=1)
    out["mc_xgb_5fold_timeblock"] = {"acc_mean": float(cvg.mean()), "acc_std": float(cvg.std())}
    print(f"  [XGBoost 5-fold TIME-BLOCK CV] acc={cvg.mean()*100:.2f}% +/- {cvg.std()*100:.2f}")

    # ---- BINARY ----
    print("\n### BINARY (spoofed vs authentic) ###")
    yb = (epoch["class"] != "authentic").astype(int).values
    Xtrb, Xteb, ytrb, yteb = train_test_split(X, yb, test_size=0.25, random_state=RS, stratify=yb)
    b = xgb(2); b.fit(Xtrb, ytrb)
    ypb = b.predict(Xteb)
    out["bin_xgb_holdout"] = report(yteb, ypb, "Binary XGBoost, holdout")
    tp = int(((ypb == 1) & (yteb == 1)).sum()); fn = int(((ypb == 0) & (yteb == 1)).sum())
    fp = int(((ypb == 1) & (yteb == 0)).sum()); tn = int(((ypb == 0) & (yteb == 0)).sum())
    out["bin_xgb_holdout"]["detection_rate"] = tp / (tp + fn) if (tp + fn) else 0.0
    out["bin_xgb_holdout"]["false_alarm"] = fp / (fp + tn) if (fp + tn) else 0.0

    ensb = VotingClassifier([("xgb", xgb(2)),
                             ("et", ExtraTreesClassifier(n_estimators=400, n_jobs=-1, random_state=RS)),
                             ("rf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RS))],
                            voting="soft", n_jobs=-1)
    ensb.fit(Xtrb, ytrb)
    out["bin_ensemble_holdout"] = report(yteb, ensb.predict(Xteb), "Binary ensemble, holdout")

    cvb = cross_val_score(xgb(2), X, yb, cv=StratifiedKFold(5, shuffle=True, random_state=RS),
                          scoring="accuracy", n_jobs=1)
    out["bin_xgb_5fold_random"] = {"acc_mean": float(cvb.mean()), "acc_std": float(cvb.std())}
    print(f"  [Binary 5-fold RANDOM CV] acc={cvb.mean()*100:.2f}% +/- {cvb.std()*100:.2f}")

    cvbg = cross_val_score(xgb(2), X, yb, cv=GroupKFold(5), groups=groups,
                           scoring="accuracy", n_jobs=1)
    out["bin_xgb_5fold_timeblock"] = {"acc_mean": float(cvbg.mean()), "acc_std": float(cvbg.std())}
    print(f"  [Binary 5-fold TIME-BLOCK CV] acc={cvbg.mean()*100:.2f}% +/- {cvbg.std()*100:.2f}")

    with open(os.path.join(RESULTS_DIR, "optimization.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved -> {os.path.join(RESULTS_DIR, 'optimization.json')}")


if __name__ == "__main__":
    main()
