"""
Ablation study: does the closed loop actually beat detection-only and a naive
reactive baseline? Reuses the recorded streaming flight (results/stream.csv), which
holds per-epoch true, detected, and forecast severity, and applies three response
policies to the SAME detector outputs:

  D  (detection-only)      : prior-work equivalent -- classify but take NO autonomous
                             navigation action (effective protective severity = 0).
  D+R (detect + reactive)  : mitigate on the CURRENT detected severity, no forecast.
  D+P+M (full, this work)  : prediction-informed pre-emptive mitigation (escalate one
                             level when the forecast is >= 0.5 above current severity).

Safety metrics (lower under-protection is better):
  under_protection_rate    : fraction of epochs whose action is weaker than the true threat
  attack_protection_rate   : among truly-attacked epochs, fraction receiving ANY protective action
  sophisticated_RTL_recall : among truly-sophisticated epochs, fraction commanded Return-to-Launch
  mean_exposure_epochs      : count of under-protected epochs over the flight
Output: results/ablation.json
"""
import os
import json

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
DELTA = 0.5


def effective_severity(config, det, fc):
    if config == "D":              # detection-only: no autonomous action
        return np.zeros_like(det)
    if config == "R":              # reactive: act on current detected severity
        return det.copy()
    # Full: pre-empt when the forecast is clearly worse than the current detection
    eff = det.copy()
    preempt = fc >= det + DELTA
    eff[preempt] = np.clip(np.round(fc[preempt]), 0, 3).astype(int)
    return eff


def metrics(true, eff):
    true = true.astype(int); eff = eff.astype(int)
    atk = true > 0
    soph = true == 3
    return {
        "under_protection_rate": float(np.mean(eff < true)),
        "safe_action_rate": float(np.mean(eff >= true)),
        "attack_protection_rate": float(np.mean(eff[atk] > 0)) if atk.any() else 0.0,
        "sophisticated_RTL_recall": float(np.mean(eff[soph] == 3)) if soph.any() else 0.0,
        "mean_exposure_epochs": int(np.sum(eff < true)),
    }


def main():
    s = pd.read_csv(os.path.join(RES, "stream.csv"))
    true = s["true_severity"].values.astype(int)
    det = s["detected_severity"].values.astype(int)
    fc = s["forecast_severity"].values.astype(float)

    out = {"timeline_epochs": int(len(s))}
    for name, cfg in [("detection_only", "D"), ("detect_reactive", "R"), ("full_loop", "F")]:
        eff = effective_severity(cfg, det, fc)
        out[name] = metrics(true, eff)

    with open(os.path.join(RES, "ablation.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"{'config':16s} {'under-prot':>10s} {'safe':>8s} {'atk-prot':>9s} {'RTL':>7s} {'exposed':>8s}")
    for name in ("detection_only", "detect_reactive", "full_loop"):
        m = out[name]
        print(f"{name:16s} {m['under_protection_rate']*100:9.2f}% {m['safe_action_rate']*100:7.2f}% "
              f"{m['attack_protection_rate']*100:8.2f}% {m['sophisticated_RTL_recall']*100:6.2f}% "
              f"{m['mean_exposure_epochs']:8d}")
    print(f"\nSaved -> {os.path.join(RES, 'ablation.json')}")


if __name__ == "__main__":
    main()
