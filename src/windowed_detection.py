"""
Windowed (temporal-aggregation) attack detection.

A spoofing attack persists over many epochs, so operationally we care about
detecting the ATTACK within a short window, not classifying every single epoch.
We slide a W-epoch window and flag "attack" if a majority of epochs in the window
read spoofed. Isolated per-epoch errors get outvoted, so attack-level detection
rises above the per-epoch rate.

We report TWO regimes, honestly labelled:
  (1) within-session  -- detector trained on a random split, evaluated on the
      held-out streaming flight (unseen epochs, conditions similar to training).
      This is the operational "deployed in known conditions" number.
  (2) cross-session   -- detector trained on the chronological first 75% of epochs
      and evaluated on the contiguous last 25% (a genuinely different time segment).
      This is the harder, more conservative number.
"""
import os
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from data_loader import load_data
from optimize_detection import build_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RS = 42


def windowed_scores(true_spoof, det_spoof, W, thresh=0.5):
    """Sliding non-overlapping windows; window is 'attack' if >thresh fraction flagged."""
    n = len(true_spoof)
    tw, pw = [], []
    for i in range(0, n - W + 1, W):
        seg_true = true_spoof[i:i + W]
        seg_pred = det_spoof[i:i + W]
        tw.append(1 if seg_true.mean() > 0.5 else 0)
        pw.append(1 if seg_pred.mean() > thresh else 0)
    tw, pw = np.array(tw), np.array(pw)
    acc = float((tw == pw).mean())
    atk = tw == 1
    ben = tw == 0
    recall = float((pw[atk] == 1).mean()) if atk.any() else 0.0
    fa = float((pw[ben] == 1).mean()) if ben.any() else 0.0
    return {"window": W, "accuracy": acc, "attack_recall": recall, "false_alarm": fa, "n_windows": len(tw)}


def main():
    out = {}

    # ---- (1) within-session: use the held-out streaming flight already produced ----
    stream = pd.read_csv(os.path.join(RESULTS_DIR, "stream.csv"))
    ts = (stream["true_severity"].values > 0).astype(int)
    ds = (stream["detected_severity"].values > 0).astype(int)
    per_epoch = float((ts == ds).mean())
    out["within_session"] = {"per_epoch_accuracy": per_epoch,
                             "windows": {W: windowed_scores(ts, ds, W) for W in (5, 11, 15, 25)}}
    print(f"[within-session] per-epoch acc={per_epoch*100:.2f}%")
    for W, s in out["within_session"]["windows"].items():
        print(f"  W={W:2d}: acc={s['accuracy']*100:.2f}%  attack-recall={s['attack_recall']*100:.2f}%  "
              f"false-alarm={s['false_alarm']*100:.2f}%")

    # ---- (2) cross-session: chronological train/test, window over contiguous time ----
    df = load_data()
    epoch, feats = build_features(df)
    es = epoch.sort_values("epoch_id").reset_index(drop=True)
    n = len(es); cut = int(0.75 * n)
    Xtr, ytr = es.loc[:cut, feats].values, (es.loc[:cut, "class"] != "authentic").astype(int).values
    Xte = es.loc[cut:, feats].values
    yte = (es.loc[cut:, "class"] != "authentic").astype(int).values
    m = XGBClassifier(n_estimators=600, max_depth=10, learning_rate=0.1, subsample=0.9,
                      colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=RS,
                      eval_metric="logloss")
    m.fit(Xtr, ytr)
    pte = m.predict(Xte)
    per_epoch_cs = float((pte == yte).mean())
    out["cross_session"] = {"per_epoch_accuracy": per_epoch_cs,
                            "windows": {W: windowed_scores(yte, pte, W) for W in (5, 11, 15, 25)}}
    print(f"\n[cross-session] per-epoch acc={per_epoch_cs*100:.2f}%")
    for W, s in out["cross_session"]["windows"].items():
        print(f"  W={W:2d}: acc={s['accuracy']*100:.2f}%  attack-recall={s['attack_recall']*100:.2f}%  "
              f"false-alarm={s['false_alarm']*100:.2f}%")

    with open(os.path.join(RESULTS_DIR, "windowed.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved -> {os.path.join(RESULTS_DIR, 'windowed.json')}")


if __name__ == "__main__":
    main()
