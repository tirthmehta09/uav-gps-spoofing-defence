"""
Detection Agent training + LEAKAGE AUDIT for the UAV GPS spoofing dataset.

We deliberately test two feature sets:
  (A) ALL 13 features (incl. time features RX_time, TOW)
  (B) PHYSICAL-ONLY (drop time/index features that could act as session proxies)

If (A) >> (B) and time features dominate importance, that's leakage.
If (B) still clears ~95%, the accuracy is a genuine spoofing signature.
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

from data_loader import load_data, FEATURE_GROUPS

# Time / index-like features that risk being "which session" proxies
TIME_LIKE = ["RX_time", "TOW_at_current_symbol_s"]
PHYSICAL = [f for f in FEATURE_GROUPS if f not in TIME_LIKE]


def evaluate_feature_set(df, feature_cols, label_col="class", tag=""):
    print(f"\n{'='*70}\n[{tag}] Features ({len(feature_cols)}): {feature_cols}\n{'='*70}")
    le = LabelEncoder()
    X = df[feature_cols].values
    y = le.fit_transform(df[label_col].values)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.15,
        subsample=0.9, colsample_bytree=0.9,
        objective="multi:softprob", num_class=len(le.classes_),
        tree_method="hist", n_jobs=-1, eval_metric="mlogloss",
    )
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_s = time.perf_counter() - t0

    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    macro_f1 = f1_score(y_te, y_pred, average="macro")
    weighted_f1 = f1_score(y_te, y_pred, average="weighted")

    print(f"Train time: {train_s:.1f}s")
    print(f"Accuracy:      {acc*100:.2f}%")
    print(f"Macro-F1:      {macro_f1:.4f}")
    print(f"Weighted-F1:   {weighted_f1:.4f}")
    print("\nPer-class report:")
    print(classification_report(y_te, y_pred, target_names=le.classes_, digits=4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(confusion_matrix(y_te, y_pred),
                       index=le.classes_, columns=le.classes_))

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nFeature importance:")
    print(imp.to_string())
    return acc, macro_f1, imp


def single_feature_leakage_probe(df, feature_cols, label_col="class"):
    """Train on ONE feature at a time; if any single feature trivially predicts class, suspect leakage."""
    print(f"\n{'='*70}\nSINGLE-FEATURE LEAKAGE PROBE (accuracy using only 1 feature)\n{'='*70}")
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].values)
    results = {}
    for f in feature_cols:
        X = df[[f]].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        m = XGBClassifier(n_estimators=60, max_depth=6, tree_method="hist", n_jobs=-1,
                          objective="multi:softprob", num_class=len(le.classes_), eval_metric="mlogloss")
        m.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, m.predict(X_te))
        results[f] = acc
    s = pd.Series(results).sort_values(ascending=False)
    print(s.to_string())
    return s


if __name__ == "__main__":
    df = load_data()

    # (A) all features
    accA, f1A, impA = evaluate_feature_set(df, FEATURE_GROUPS, tag="A: ALL 13 FEATURES")

    # (B) physical-only (leakage-safe)
    accB, f1B, impB = evaluate_feature_set(df, PHYSICAL, tag="B: PHYSICAL-ONLY (no time features)")

    # Single-feature probe on all
    probe = single_feature_leakage_probe(df, FEATURE_GROUPS)

    print(f"\n{'#'*70}")
    print("SUMMARY")
    print(f"  (A) All 13 features:      acc={accA*100:.2f}%  macroF1={f1A:.4f}")
    print(f"  (B) Physical-only:        acc={accB*100:.2f}%  macroF1={f1B:.4f}")
    print(f"  Max single-feature acc:   {probe.max()*100:.2f}%  ({probe.idxmax()})")
    print(f"{'#'*70}")
