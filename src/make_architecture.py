"""
System / threat-model architecture diagram.
Grayscale, high-resolution, professional block diagram with STRAIGHT / right-angle
(orthogonal) connectors only -- no slanted or curved arrows.
Written to figures/fig_architecture.png at 600 DPI.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

BLACK = "#000000"; GREY = "#555555"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                     "mathtext.fontset": "dejavusans"})


def box(ax, x, y, w, h, text, fc="#ffffff", fs=9, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.6",
                       linewidth=1.5, edgecolor=BLACK, facecolor=fc, zorder=3)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=BLACK,
            fontsize=fs, fontweight="bold" if bold else "normal", zorder=4,
            linespacing=1.4)
    return dict(cx=x + w / 2, cy=y + h / 2, l=x, r=x + w, t=y + h, b=y)


def ortho(ax, pts, dashed=False, color=BLACK, lw=1.8):
    """Draw an orthogonal connector through pts (each segment horizontal or
    vertical only) with an arrowhead on the final segment."""
    ls = (0, (5, 3)) if dashed else "-"
    if len(pts) > 2:
        xs = [p[0] for p in pts[:-1]]; ys = [p[1] for p in pts[:-1]]
        ax.add_line(Line2D(xs, ys, color=color, lw=lw, linestyle=ls,
                           solid_capstyle="projecting", zorder=2))
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                                shrinkA=0, shrinkB=0, mutation_scale=16), zorder=2)


def main():
    fig, ax = plt.subplots(figsize=(8.0, 9.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # ---- zone bands (light greys) with labels that clear the boxes ----
    zones = [(81, 99, "#f5f5f5", "Threat environment"),
             (63.5, 80.5, "#ededed", "UAV sensing / data pipeline"),
             (28.5, 63, "#e4e4e4", "Safety-aware AI framework (this work)"),
             (4, 19, "#ededed", "Actuation & recovery")]
    for y0, y1, c, label in zones:
        ax.add_patch(Rectangle((1.5, y0), 93, y1 - y0, facecolor=c, edgecolor="none", zorder=0))
        ax.text(3, y1 - 1.8, label, fontsize=8.5, style="italic", color=GREY, zorder=1,
                va="top")

    # ---- threat environment ----
    sat = box(ax, 10, 83, 32, 9, "GNSS satellites\n(legitimate)", "#ffffff", 9.5, True)
    atk = box(ax, 58, 83, 32, 9, "Spoofing attacker\n(RF transmitter)", "#dcdcdc", 9.5, True)

    # ---- receiver (wide: spans under both sources so arrows stay vertical) ----
    rec = box(ax, 15, 66, 70, 9, "UAV 8-channel GPS receiver  (software-defined radio)", "#ffffff", 9.5, True)
    ortho(ax, [(sat["cx"], sat["b"]), (sat["cx"], rec["t"])])                 # authentic (V)
    ortho(ax, [(atk["cx"], atk["b"]), (atk["cx"], rec["t"])], dashed=True)    # spoofed (V, dashed)
    _wb = dict(facecolor="white", edgecolor="none", pad=1.2)
    ax.text(sat["cx"], 79.5, "authentic", fontsize=7.2, style="italic", color=GREY,
            ha="center", va="center", bbox=_wb, zorder=5)
    ax.text(atk["cx"], 79.5, "spoofed", fontsize=7.2, style="italic", color=GREY,
            ha="center", va="center", bbox=_wb, zorder=5)

    # ---- feature extraction (wide, centered) ----
    feat = box(ax, 15, 49, 70, 9,
               "Data acquisition  →  Feature extraction\n"
               "79 cross-channel features / epoch  (mean, std, min, max, range, CV; + active-channel count)",
               "#ffffff", 8)
    ortho(ax, [(rec["cx"], rec["b"]), (feat["cx"], feat["t"])])               # V

    # ---- AI framework: three stage boxes in a row ----
    det = box(ax, 6, 33.5, 27, 12.5,
              "Detection\n(XGBoost · 4-class)\n" r"$x_t \rightarrow$ severity $s_t$" "\n+ confidence",
              "#d6d6d6", 8, True)
    pred = box(ax, 36.5, 33.5, 27, 12.5,
               "Prediction\n(sliding-window trend)\n" r"$s_t \rightarrow$ forecast $\hat{s}$" "\nescalation warning",
               "#d6d6d6", 8, True)
    mit = box(ax, 67, 33.5, 27, 12.5,
              "Mitigation\n(finite-state policy)\n" r"$(s_t,\hat{s}) \rightarrow$ action $a_t$" "\n"
              r"pre-empt if $\hat{s} \geq s_t{+}\delta$",
              "#d6d6d6", 8, True)

    # feature -> Detection: down, left, down  (three straight segments)
    ortho(ax, [(feat["cx"], feat["b"]), (feat["cx"], 47.5), (det["cx"], 47.5), (det["cx"], det["t"])])
    ortho(ax, [(det["r"], det["cy"]), (pred["l"], pred["cy"])])               # H
    ortho(ax, [(pred["r"], pred["cy"]), (mit["l"], mit["cy"])])               # H
    ax.text(50, 30.4, "coordinator applies Detection → Prediction → Mitigation every epoch",
            ha="center", fontsize=7.4, style="italic", color=GREY)

    # ---- actuation / recovery ----
    act = box(ax, 10, 6, 80, 9,
              r"Navigation action  $a_t \in$ { accept GPS  |  GPS integrity check  |" "\n"
              "reject GPS / inertial-nav fallback  |  return-to-launch }",
              "#ffffff", 8.3, True)
    ortho(ax, [(mit["cx"], mit["b"]), (mit["cx"], act["t"])])                 # V

    # ---- recovery / feedback: orthogonal loop on the right (right, up, left) ----
    ortho(ax, [(act["r"], act["cy"]), (96.5, act["cy"]), (96.5, rec["cy"]), (rec["r"], rec["cy"])],
          dashed=True, color=GREY, lw=1.5)
    ax.text(98.2, (act["cy"] + rec["cy"]) / 2, "recovery / feedback", fontsize=7.2, style="italic",
            color=GREY, rotation=90, ha="center", va="center")

    ax.set_title("System and Threat-Model Architecture", color=BLACK, fontweight="bold", fontsize=13, pad=10)
    fig.savefig(os.path.join(FIG, "fig_architecture.png"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", os.path.join(FIG, "fig_architecture.png"))


if __name__ == "__main__":
    main()
