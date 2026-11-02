"""
Epoch-level (multi-channel fusion) Detection Agent + temporal-leakage audit.

Each epoch = one time step of the GPS receiver with up to 8 active channels.
We fuse the 8 channels per epoch into aggregate features (mean/std/min/max/spread
per measurement + active-channel count). Spoofing distorts the whole constellation,
so cross-channel statistics are strong, physically-grounded signatures.

AUDIT: we compare
  (1) random stratified split  (the accepted literature protocol)
  (2) contiguous block split    (train early time segments, test later ones)
If (2) collapses vs (1), temporal autocorrelation is inflating (1) -> report honestly.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

from data_loader import load_data, FEATURE_GROUPS

AGGS = ["mean", "std", "min", "max"]


def build_epoch_features(df):
    # Label per epoch: max Output across active channels (0=authentic, else attack type)
    epoch_label = df.groupby("epoch_id")["Output"].max()

    # Consistency check: within attacked epochs, are channel labels consistent?
    def nonzero_uniq(s):
        v = s[s != 0].unique()
        return len(v)
    n_distinct = df.groupby("epoch_id")["Output"].apply(nonzero_uniq)
    mixed = (n_distinct > 1).sum()
    print(f"Epochs with >1 distinct nonzero label (mixed attacks): {mixed} / {len(n_distinct)} "
          f"({100*mixed/len(n_distinct):.2f}%)")

    # Aggregate features across channels per epoch
    agg = df.groupby("epoch_id")[FEATURE_GROUPS].agg(AGGS)
    agg.columns = [f"{f}_{a}" for f, a in agg.columns]
    agg = agg.fillna(0.0)
    agg["n_active"] = df.groupby("epoch_id")["channel"].count()

    agg["Output"] = epoch_label
    CLASS_MAP = {0: "authentic", 1: "simplistic", 2: "intermediate", 3: "sophisticated"}
    agg["class"] = agg["Output"].map(CLASS_MAP)
    agg = agg.reset_index()  # keep epoch_id for chronological split
    return agg


def train_eval(X_tr, y_tr, X_te, y_te, le, tag):
    print(f"\n{'='*70}\n[{tag}]\n{'='*70}")
    model = XGBClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.15,
        subsample=0.9, colsample_bytree=0.9,
        objective="multi:softprob", num_class=len(le.classes_),
        tree_method="hist", n_jobs=-1, eval_metric="mlogloss",
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    macro_f1 = f1_score(y_te, y_pred, average="macro")
    print(f"Accuracy:    {acc*100:.2f}%")
    print(f"Macro-F1:    {macro_f1:.4f}")
    print(classification_report(y_te, y_pred, target_names=le.classes_, digits=4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(confusion_matrix(y_te, y_pred), index=le.classes_, columns=le.classes_))
    return acc, macro_f1, model


if __name__ == "__main__":
    df = load_data()
    epoch = build_epoch_features(df)
    print(f"\nEpoch-level dataset: {epoch.shape}")
    print("Epoch class distribution:")
    print(epoch["class"].value_counts())

    feat_cols = [c for c in epoch.columns if c not in ("epoch_id", "Output", "class")]
    print(f"\n# epoch features: {len(feat_cols)}")

    # --- temporal layout audit: class distribution across 10 chronological bins ---
    print(f"\n{'='*70}\nTEMPORAL LAYOUT (class share per chronological decile)\n{'='*70}")
    epoch_sorted = epoch.sort_values("epoch_id").reset_index(drop=True)
    bins = np.array_split(epoch_sorted, 10)
    for i, b in enumerate(bins):
        dist = (b["class"].value_counts(normalize=True) * 100).round(1).to_dict()
        print(f"  decile {i}: n={len(b):6d}  {dist}")

    le = LabelEncoder().fit(epoch["class"])

    # (1) random stratified split
    Xr = epoch[feat_cols].values
    yr = le.transform(epoch["class"].values)
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.25, random_state=42, stratify=yr)
    acc_rand, f1_rand, _ = train_eval(Xtr, ytr, Xte, yte, le, "RANDOM STRATIFIED SPLIT (literature protocol)")

    # (2) contiguous block split: 5 folds over time, round-robin so each class-run is split
    # Assign each epoch (in time order) to one of 5 time-blocks; test on block 4 (latest quintile)
    n = len(epoch_sorted)
    block = (np.arange(n) * 5 // n)  # 0..4 chronological blocks
    test_mask = block == 4
    Xall = epoch_sorted[feat_cols].values
    yall = le.transform(epoch_sorted["class"].values)
    acc_blk, f1_blk, _ = train_eval(
        Xall[~test_mask], yall[~test_mask], Xall[test_mask], yall[test_mask], le,
        "CONTIGUOUS BLOCK SPLIT (train first 80% of time, test last 20%)"
    )

    print(f"\n{'#'*70}\nSUMMARY (epoch-level multi-channel fusion)")
    print(f"  Random stratified:   acc={acc_rand*100:.2f}%  macroF1={f1_rand:.4f}")
    print(f"  Contiguous block:    acc={acc_blk*100:.2f}%  macroF1={f1_blk:.4f}")
    print(f"{'#'*70}")
