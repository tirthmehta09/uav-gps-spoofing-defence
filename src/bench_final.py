"""
Decisive test to clear 95% under the standard (random-split) protocol:
  1. Per-channel + epoch cross-channel CONTEXT features (multi-class)
  2. Binary (authentic vs spoofed) ceiling
Each also reported under the honest time-based split for transparency.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

from data_loader import load_data, FEATURE_GROUPS

def xgb(nclass):
    return XGBClassifier(n_estimators=400, max_depth=9, learning_rate=0.15,
        subsample=0.9, colsample_bytree=0.9, tree_method="hist", n_jobs=-1,
        objective=("binary:logistic" if nclass==2 else "multi:softprob"),
        num_class=(1 if nclass==2 else nclass), eval_metric="logloss")

def run(X, y, tag, groups=None):
    print(f"\n{'='*66}\n[{tag}]  n={len(y)}  classes={len(np.unique(y))}")
    # random split
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
    m = xgb(len(np.unique(y))); m.fit(Xtr,ytr); yp=m.predict(Xte)
    print(f"  RANDOM split : acc={accuracy_score(yte,yp)*100:.2f}%  macroF1={f1_score(yte,yp,average='macro'):.4f}")
    # time-based split (honest) if group order provided
    if groups is not None:
        order = np.argsort(groups)
        Xs, ys = X[order], y[order]
        n=len(ys); tm = (np.arange(n)*5//n)==4
        m2 = xgb(len(np.unique(y))); m2.fit(Xs[~tm],ys[~tm]); yp2=m2.predict(Xs[tm])
        print(f"  TIME split   : acc={accuracy_score(ys[tm],yp2)*100:.2f}%  macroF1={f1_score(ys[tm],yp2,average='macro'):.4f}")

if __name__ == "__main__":
    df = load_data()

    # Build epoch context (mean/std across channels per epoch) and merge back per-channel
    ctx = df.groupby("epoch_id")[FEATURE_GROUPS].agg(["mean","std"]).fillna(0.0)
    ctx.columns = [f"{f}_ep_{a}" for f,a in ctx.columns]
    ctx["ep_nactive"] = df.groupby("epoch_id")["channel"].count()
    merged = df.merge(ctx, on="epoch_id", how="left")

    ctx_cols = list(ctx.columns)
    all_feats = FEATURE_GROUPS + ctx_cols
    groups = merged["epoch_id"].values

    # 1. multi-class per-channel + context
    le = LabelEncoder().fit(merged["class"])
    run(merged[all_feats].values, le.transform(merged["class"].values),
        "MULTI-CLASS: per-channel + epoch context", groups=groups)

    # 2. binary per-channel + context
    ybin = (merged["class"] != "authentic").astype(int).values
    run(merged[all_feats].values, ybin,
        "BINARY: spoofed vs authentic, per-channel + context", groups=groups)

    # 3. binary per-channel base features only (matches literature simplest form)
    run(df[FEATURE_GROUPS].values, (df["class"]!="authentic").astype(int).values,
        "BINARY: spoofed vs authentic, base 13 feats", groups=df["epoch_id"].values)
