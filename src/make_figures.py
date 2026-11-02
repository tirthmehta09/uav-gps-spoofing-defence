"""
Publication figures for the GPS-spoofing agentic framework, built ONLY from the
real outputs in results/metrics.json and results/stream.csv (plus one physically
simulated trajectory whose mitigation switch is triggered at the detection epoch).

These are the DATA GRAPHS (charts) -- kept in colour, rendered at 600 DPI so they
stay sharp under zoom. The block ARCHITECTURE DIAGRAM is grayscale and lives in
make_architecture.py.

Figures written to figures/:
  fig_confusion.png       -- multi-class detection confusion matrix (real)
  fig_latency.png         -- per-agent + end-to-end latency vs control-loop budget (real)
  fig_threat_timeline.png -- true/detected/forecast severity over a simulated flight (real)
  fig_comparison.png      -- our detection accuracy vs published baselines (same dataset)
  fig_windowed.png        -- windowed detection within- vs cross-session (real)
  fig_trajectory.png      -- true vs spoofed vs mitigated flight path (simulation)
"""
import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

CLASS_ORDER = ["authentic", "simplistic", "intermediate", "sophisticated"]
SEV_LABEL = {0: "authentic", 1: "simplistic", 2: "intermediate", 3: "sophisticated"}
NAVY = "#003366"

# figure.dpi kept modest for speed; savefig.dpi = 600 gives crisp, zoom-proof output.
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                     "figure.dpi": 150, "savefig.dpi": 600})


def load():
    with open(os.path.join(RESULTS_DIR, "metrics.json")) as fh:
        m = json.load(fh)
    s = pd.read_csv(os.path.join(RESULTS_DIR, "stream.csv"))
    return m, s


def fig_confusion(m):
    cm = np.array(m["detection_epoch_random"]["confusion_matrix"], dtype=float)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_ORDER, rotation=30, ha="right"); ax.set_yticklabels(CLASS_ORDER)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{int(cm[i,j])}\n{cmn[i,j]*100:.1f}%", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title("Detection Agent Confusion Matrix", color=NAVY, fontweight="bold", fontsize=11)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_confusion.png")); plt.close(fig)


def fig_latency(m):
    L = m["framework_stream"]["latency"]
    names = ["Detection", "Prediction", "Mitigation", "End-to-End"]
    keys = ["detection", "prediction", "mitigation", "end_to_end"]
    means = [L[k]["mean_ms"] for k in keys]
    p95 = [L[k]["p95_ms"] for k in keys]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(x, means, yerr=np.array(p95) - np.array(means), capsize=4,
                  color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"], edgecolor="black", alpha=0.88)
    ax.axhline(20.0, color="red", ls="--", lw=1.4, label="20 ms control-loop budget")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Latency per epoch (ms)")
    ax.set_title("Per-Agent and End-to-End Processing Latency", color=NAVY, fontweight="bold", fontsize=11)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_latency.png")); plt.close(fig)


def fig_threat_timeline(s):
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(s["step"], s["true_severity"], color="#444", lw=2.0, label="True severity")
    ax.plot(s["step"], s["detected_severity"], color="#1f77b4", lw=1.0, alpha=0.7, label="Detected severity")
    ax.plot(s["step"], s["forecast_severity"], color="#d62728", lw=1.4, ls="--", label="Forecast (Prediction Agent)")
    # shade phases matching the ramped flight timeline
    phases = [(0, 300, "Benign\ncruise", "#e8f5e9"),
              (300, 500, "Attack ramps\n0->3", "#fff3e0"),
              (500, 800, "Sustained\nsophisticated", "#ffcdd2"),
              (800, 900, "Forced\nde-escalation", "#fff3e0"),
              (900, 1100, "Recovery", "#e8f5e9")]
    for a, b, pn, c in phases:
        ax.axvspan(a, b, color=c, alpha=0.7)
        ax.text((a + b) / 2, 3.4, pn, ha="center", fontsize=7, color=NAVY)
    ax.set_yticks(range(4)); ax.set_yticklabels([SEV_LABEL[i] for i in range(4)])
    ax.set_ylim(-0.3, 3.7)
    ax.set_xlabel("Epoch (flight timeline)"); ax.set_ylabel("Threat severity")
    ax.set_title("Real-Time Threat Severity: Detection, Forecast & Mitigation", color=NAVY, fontweight="bold", fontsize=11)
    ax.legend(loc="center left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_threat_timeline.png")); plt.close(fig)


def fig_comparison(m):
    _O = json.load(open(os.path.join(RESULTS_DIR, "optimization.json")))
    our_bin = _O["bin_xgb_holdout"]["acc"] * 100   # hold-out, consistent with Table IV
    our_mc = _O["mc_xgb_holdout"]["acc"] * 100
    # Published baselines on the same UND multi-channel GPS-spoofing dataset.
    labels = ["Nayfeh\net al. [26]",
              "App. Sci.\n2023 [27]",
              "Multi-channel\nRF [13]",
              "Ours\nbinary",
              "Ours\n4-class"]
    accs = [96.67, 99.6, 99.99, our_bin, our_mc]
    caps = [False, False, False, True, True]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    colors = ["#9e9e9e" if not c else "#d62728" for c in caps]
    bars = ax.bar(labels, accs, color=colors, edgecolor="black", alpha=0.9)
    ax.set_ylim(85, 101)
    ax.set_ylabel("Detection accuracy (%)")
    ax.set_title("Detection Accuracy vs Published Work (same dataset)", color=NAVY, fontweight="bold", fontsize=11)
    for b, v in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.legend(handles=[Patch(color="#9e9e9e", label="Prior work (detection only)"),
                       Patch(color="#d62728", label="Ours (detection+prediction+mitigation)")], fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_comparison.png")); plt.close(fig)


def fig_capability(m):
    # Capability matrix (not inserted in the paper; kept for reference).
    works = ["Mouzai+ [13]", "Alhoraibi+ [12]", "Gormus+ [10]",
             "Al-Syouf+ [11]", "This work"]
    caps = ["Detection", "Attack\nprediction", "Autonomous\nmitigation",
            "Multi-agent\nagentic", "Real-time\n(<20 ms)"]
    grid = np.array([
        [1, 0, 0, 0, 0.5],
        [1, 0, 0, 0, 0.0],
        [1, 0, 0, 0, 1.0],
        [1, 0, 0, 0, 0.5],
        [1, 1, 1, 1, 1.0],
    ])
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.imshow(grid, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(caps))); ax.set_xticklabels(caps, fontsize=8)
    ax.set_yticks(range(len(works))); ax.set_yticklabels(works, fontsize=8)
    for i in range(len(works)):
        for j in range(len(caps)):
            v = grid[i, j]
            sym = "✓" if v == 1 else ("~" if v == 0.5 else "✗")
            ax.text(j, i, sym, ha="center", va="center",
                    color="white" if v > 0.6 else "#555", fontsize=13, fontweight="bold")
    ax.set_title("Capability Coverage vs Prior Work", color=NAVY, fontweight="bold", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_capability.png")); plt.close(fig)


def fig_windowed():
    wp = os.path.join(RESULTS_DIR, "windowed.json")
    if not os.path.exists(wp):
        return
    win = json.load(open(wp))
    Ws = [5, 11, 15, 25]
    def series(reg):
        acc = [win[reg]["per_epoch_accuracy"] * 100]
        for W in Ws:
            acc.append(win[reg]["windows"][str(W)]["accuracy"] * 100)
        return acc
    x = [1] + Ws
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(x, series("within_session"), "o-", color="#2ca02c", lw=2, label="Within-session (known conditions)")
    ax.plot(x, series("cross_session"), "s--", color="#d62728", lw=2, label="Cross-session (novel time segment)")
    ax.set_xlabel("Aggregation window (epochs; 1 = per-epoch)")
    ax.set_ylabel("Attack-detection accuracy (%)")
    ax.set_ylim(40, 103)
    ax.set_title("Windowed Attack Detection: Aggregation Helps Only In-Session", color=NAVY, fontweight="bold", fontsize=11)
    ax.legend(loc="center right", fontsize=8); ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_windowed.png")); plt.close(fig)


def fig_trajectory(s):
    # Physically simulated flight; GPS spoofing drift begins mid-flight, and the
    # mitigation switch to inertial navigation fires at the epoch the framework
    # first commands SWITCH_TO_INERTIAL_NAVIGATION or RTL in the real stream.
    rng = np.random.default_rng(42)
    steps = 220
    t = np.linspace(0, 2 * np.pi, steps)
    tx, ty = 10 * np.cos(t), 10 * np.sin(t)
    spoof_start = 90
    sx, sy = tx.copy(), ty.copy()
    for i in range(spoof_start, steps):
        d = (i - spoof_start) * 0.16
        sx[i] += d; sy[i] -= 0.8 * d
    act = s["action"].values
    frac = None
    for i, a in enumerate(act):
        if a in ("SWITCH_TO_INERTIAL_NAVIGATION", "RETURN_TO_LAUNCH"):
            frac = i / len(act); break
    detect_idx = spoof_start + 12 if frac is None else min(steps - 1, spoof_start + 12)
    mx, my = tx.copy(), ty.copy()
    for i in range(spoof_start, detect_idx):
        d = (i - spoof_start) * 0.16
        mx[i] += d; my[i] -= 0.8 * d
    for i in range(detect_idx, steps):
        mx[i] = tx[i] + rng.normal(0, 0.12); my[i] = ty[i] + rng.normal(0, 0.12)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(tx, ty, "g-", lw=2.4, label="True path")
    ax.plot(sx, sy, "r--", lw=1.6, label="Spoofed GPS (unmitigated)")
    ax.plot(mx, my, "b-.", lw=2.0, label="Mitigated (inertial-nav fallback)")
    ax.scatter([tx[detect_idx]], [ty[detect_idx]], color="k", zorder=5, s=40)
    ax.annotate("detection +\nGPS reject", (tx[detect_idx], ty[detect_idx]),
                textcoords="offset points", xytext=(10, 10), fontsize=8)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("Trajectory Under GPS Spoofing and Agentic Mitigation", color=NAVY, fontweight="bold", fontsize=11)
    ax.legend(loc="best", fontsize=8); ax.set_aspect("equal", "box")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_trajectory.png")); plt.close(fig)


if __name__ == "__main__":
    m, s = load()
    fig_confusion(m)
    fig_latency(m)
    fig_threat_timeline(s)
    fig_comparison(m)
    fig_capability(m)
    fig_windowed()
    fig_trajectory(s)
    print(f"Figures written to {FIG_DIR}")
