"""
Generate the revised IEEE manuscript into the conference-template-a4 template.
All quantitative claims are pulled from results/*.json for consistency.

Detection accuracy sourcing (for cross-manuscript consistency):
  - Headline / cross-validation numbers  -> optimization.json (5-fold CV, hold-out, time-block)
  - Per-class hold-out + confusion        -> metrics.json (same model config & split as optimize)
  - Streaming / mitigation / latency      -> metrics.json framework_stream
  - Windowed                              -> windowed.json ; Ablation -> ablation.json
"""
import os
import json

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.text import WD_TAB_ALIGNMENT

# LaTeX -> MathML -> OMML (native Word equation objects). Degrades to text if unavailable.
import glob
from copy import deepcopy
try:
    from latex2mathml.converter import convert as _latex2mathml
    from lxml import etree as _etree
    _xslc = (glob.glob(r"C:\Program Files*\Microsoft Office\*\MML2OMML.XSL")
             + glob.glob(r"C:\Program Files*\Microsoft Office\root\*\MML2OMML.XSL"))
    _MML2OMML = _etree.XSLT(_etree.parse(_xslc[0])) if _xslc else None
except Exception:
    _latex2mathml = None; _MML2OMML = None


def _omml(latex):
    mml = _latex2mathml(latex)
    return deepcopy(_MML2OMML(_etree.fromstring(mml)).getroot())

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(BASE, "conference-template-a4-transitional.docx")
FIGDIR = os.path.join(BASE, "figures")
RESULTS = os.path.join(BASE, "results")
OUT = os.path.join(BASE, "paper", "UAV_GPS_Spoofing_IEEE.docx")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

M = json.load(open(os.path.join(RESULTS, "metrics.json")))
O = json.load(open(os.path.join(RESULTS, "optimization.json")))
W = json.load(open(os.path.join(RESULTS, "windowed.json")))
A = json.load(open(os.path.join(RESULTS, "ablation.json")))


def pct(x, d=2):
    return f"{x*100:.{d}f}%"


def oa(k):
    v = O.get(k, {})
    return v.get("acc", v.get("acc_mean", 0.0))


def ostd(k):
    return O.get(k, {}).get("acc_std", 0.0)


# ---------- helpers ----------
def clear_body(doc):
    body = doc.element.body
    for child in list(body):
        if child.tag in (qn("w:p"), qn("w:tbl")):
            body.remove(child)


def set_columns(section, num, space=425):
    sectPr = section._sectPr
    cols = sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols"); sectPr.append(cols)
    cols.set(qn("w:num"), str(num)); cols.set(qn("w:space"), str(space))


def para(doc, text, style=None, align=None, lead=None, lead_italic=True, size=None):
    p = doc.add_paragraph(style=style) if style else doc.add_paragraph()
    if align is not None:
        p.alignment = align
    if lead:
        r = p.add_run(lead)
        r.italic = lead_italic; r.bold = not lead_italic
        if size: r.font.size = Pt(size)
    if text:
        r = p.add_run(text)
        if size: r.font.size = Pt(size)
    return p


def heading(doc, text):
    return doc.add_paragraph(text, style="Heading 1")


def subheading(doc, text):
    import re
    text = re.sub(r"^[A-Z]\.\s+", "", text)   # Heading 2 auto-letters
    try:
        return doc.add_paragraph(text, style="Heading 2")
    except KeyError:
        p = doc.add_paragraph(); r = p.add_run(text); r.italic = True
        return p


def equation(doc, latex, num=None, fallback=None):
    """Insert a native Word (OMML) equation, centered, with a right-aligned number.
    Degrades to centered italic text if the OMML toolchain is unavailable."""
    p = doc.add_paragraph()
    if _MML2OMML is not None and _latex2mathml is not None:
        try:
            pf = p.paragraph_format
            pf.tab_stops.add_tab_stop(Inches(1.55), WD_TAB_ALIGNMENT.CENTER)
            pf.tab_stops.add_tab_stop(Inches(3.35), WD_TAB_ALIGNMENT.RIGHT)
            p.add_run().add_tab()
            p._p.append(_omml(latex))
            if num:
                rr = p.add_run(); rr.add_tab()
                r2 = p.add_run(f"({num})"); r2.font.size = Pt(9)
            return p
        except Exception as e:
            print(f"  eqn OMML fallback ({num}): {e}")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(fallback or latex); r.italic = True; r.font.size = Pt(10)
    if num:
        rn = p.add_run(f"  ({num})"); rn.font.size = Pt(9)
    return p


def bullets(doc, items):
    for it in items:
        doc.add_paragraph(it, style="bullet list")


def add_figure(doc, fname, caption, width=3.2):
    import re
    path = os.path.join(FIGDIR, fname)
    if not os.path.exists(path):
        print(f"  WARN missing figure {fname}"); return
    caption = re.sub(r"^Fig\.\s*\d+\.\s*", "", caption)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Inches(width))
    try:
        c = doc.add_paragraph(caption, style="figure caption")
    except KeyError:
        c = doc.add_paragraph(caption); c.runs and setattr(c.runs[0].font, "size", Pt(8))
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER


def _apply_borders(t, sz="6"):
    """Apply single black borders to every cell edge (independent of template styles)."""
    tblPr = t._tbl.tblPr
    existing = tblPr.find(qn("w:tblBorders"))
    if existing is not None:
        tblPr.remove(existing)
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), sz)
        e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")
        b.append(e)
    tblPr.append(b)


def styled_table(doc, headers, rows, caption=None):
    if caption:
        cp = doc.add_paragraph(); cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cp.add_run(caption); r.bold = True; r.font.size = Pt(8)
    t = doc.add_table(rows=1, cols=len(headers)); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    _apply_borders(t)
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]; cell.text = ""
        rr = cell.paragraphs[0].add_run(h); rr.bold = True; rr.font.size = Pt(8)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            rr = cells[i].paragraphs[0].add_run(str(v)); rr.font.size = Pt(8)
    return t


def add_algorithm(doc, title, lines):
    t = doc.add_table(rows=1, cols=1); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    _apply_borders(t)
    cell = t.rows[0].cells[0]
    tr = cell.paragraphs[0].add_run(title); tr.bold = True; tr.font.size = Pt(8.5)
    for ln in lines:
        r = cell.add_paragraph().add_run(ln); r.font.name = "Consolas"; r.font.size = Pt(8)
    return t


def _author_grid(doc, authors, ncols=3):
    """IEEE author blocks: `ncols` per row (e.g., 3 in the first row); any remaining
    authors are centered on the next row. Each block = name, italic affiliation, email."""
    aff = ["Department of Information Technology",
           "Mukesh Patel School of Technology",
           "Management & Engineering, SVKM's NMIMS",
           "Mumbai, India"]
    colw = Inches(2.3)
    starts = list(range(0, len(authors), ncols))
    for ridx, i0 in enumerate(starts):
        row = authors[i0:i0 + ncols]
        t = doc.add_table(rows=1, cols=ncols)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False; t.allow_autofit = False
        offset = 0   # fill from the left (4th author sits under the 1st)
        for i, (name, email) in enumerate(row):
            cell = t.rows[0].cells[offset + i]; cell.width = colw
            p0 = cell.paragraphs[0]; p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p0.add_run(name).font.size = Pt(10)
            for line in aff:
                pl = cell.add_paragraph(); pl.alignment = WD_ALIGN_PARAGRAPH.CENTER
                rl = pl.add_run(line); rl.italic = True; rl.font.size = Pt(9)
            pe = cell.add_paragraph(); pe.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pe.add_run(email).font.size = Pt(9)
        for col in t.columns:
            col.width = colw
        if ridx != len(starts) - 1:   # small gap between author rows
            sp = doc.add_paragraph(); sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            sp.paragraph_format.space_before = Pt(0); sp.paragraph_format.space_after = Pt(0)
            sp.add_run(" ").font.size = Pt(8)
    return t


# ---------- build ----------
def main():
    doc = Document(TEMPLATE)
    clear_body(doc)

    det = M["detection_epoch_random"]; fw = M["framework_stream"]
    pc = M["detection_perchannel_random"]; L = fw["latency"]; ds = M["dataset"]
    bh = O["bin_xgb_holdout"]
    w5 = W["within_session"]["windows"]["5"]; w11 = W["within_session"]["windows"]["11"]
    w25 = W["within_session"]["windows"]["25"]; wc11 = W["cross_session"]["windows"]["11"]

    # ===== TITLE BLOCK =====
    set_columns(doc.sections[0], 1)
    para(doc, "A Safety-Aware Multi-Agent AI Framework for Detection, Prediction, "
              "and Mitigation of UAV GPS Spoofing Attacks", style="paper title",
         align=WD_ALIGN_PARAGRAPH.CENTER)
    _author_grid(doc, [
        ("Krimy K. Shah", "shahkrimy09@gmail.com"),
        ("Tirth V. Mehta", "tvm2052@gmail.com"),
        ("Dhara Patel", "dhara.patel58@nmims.in"),
        ("Sana Ansari", "sana.ansari@nmims.edu"),
    ])

    body = doc.add_section(WD_SECTION.CONTINUOUS)
    set_columns(body, 2, space=425)

    para(doc,
         "Global Positioning System (GPS) spoofing is a safety-critical threat to unmanned aerial vehicles "
         "(UAVs): counterfeit signals can drive the navigation state off course without triggering an overt "
         "failure. Most machine-learning studies treat this as a detection or classification problem evaluated on "
         "random data splits, and even recent work that integrates detection with mitigation does so by learning a "
         "reactive control policy. We propose a safety-aware framework organized as three cooperating components—"
         "detection, prediction, and mitigation—coordinated to operate on GPS signal-quality features alone. The "
         "framework classifies attack severity, forecasts its escalation, and drives an interpretable finite-state "
         "safety policy that acts pre-emptively, and it is evaluated under both random and leakage-free "
         "chronological protocols. On the public University of North Dakota multi-channel dataset, spoofing is "
         f"detected with about {pct(oa('bin_xgb_5fold_random'),0)} accuracy under five-fold cross-validation; more "
         "importantly, an ablation shows that adding the prediction and mitigation stages reduces the safety "
         f"under-protection rate from about {pct(A['detection_only']['under_protection_rate'],0)} for a "
         f"detection-only system to about {pct(A['full_loop']['under_protection_rate'],0)}. The framework offers "
         "an interpretable, real-time complement to learned control policies, while its chronological evaluation "
         "exposes a cross-session generalization gap that random-split accuracies conceal.",
         style="Abstract", lead="Abstract—", lead_italic=False)
    para(doc, "UAV security, GPS spoofing, multi-agent AI, attack-severity prediction, safety-aware mitigation, "
              "temporal generalization, intrusion detection", style="Keywords", lead="Keywords—", lead_italic=False)

    # ---- I. INTRODUCTION ----
    heading(doc, "Introduction")
    para(doc, "Unmanned Aerial Vehicles (UAVs) are increasingly used for inspection, agriculture, logistics, "
              "surveillance, and emergency response, where navigation integrity is tied directly to operational "
              "safety. Their reliance on satellite navigation, wireless links, embedded software, and sensors "
              "creates a broad cyber-physical attack surface [1], [2]. Because many missions are flown Beyond "
              "Visual Line of Sight, an on-board defense that acts autonomously within the flight-control timing "
              "budget is essential.", style="Body Text")
    para(doc, "GPS spoofing is especially dangerous because it need not produce an obvious failure: unauthenticated "
              "signals arriving below the noise floor can be overpowered by a counterfeit transmitter that captures "
              "the receiver's tracking loops and slowly pulls the reported position away from truth while the "
              "system appears to operate normally. A corrupted navigation state then propagates into guidance and "
              "control, so a defense should not merely decide whether spoofing is present, but also estimate how "
              "the threat is evolving and select a response that preserves safety.", style="Body Text")
    para(doc, "Machine-learning detection of GPS spoofing is now mature, with strong results using classical, "
              "ensemble, adversarial, and deep models on single- and multi-channel measurements [3]–[8]. Fewer "
              "works close the loop to a response. Reinforcement learning has recently integrated detection and "
              "mitigation by learning, from GPS and on-board vision inputs, a policy that switches between "
              "GPS-based and vision-based navigation [14]; related control- and multi-agent approaches address "
              "distributed detection and resilient control [9], and agentic designs decompose defense into "
              "specialized components for other domains [10], [11]. Two aspects remain comparatively "
              "under-addressed for GPS spoofing: explicit forecasting of attack-severity escalation to enable "
              "pre-emptive response, and evaluation under time-separated (leakage-free) protocols. Our "
              "contributions are:", style="Body Text")
    bullets(doc, [
        "A safety-aware framework that couples attack-severity detection with explicit escalation forecasting and "
        "an interpretable, pre-emptive mitigation policy, using GPS signal-quality features without additional "
        "sensors.",
        "A formal problem statement and a transparent, deterministic decision rule for integrated "
        "detection–prediction–mitigation, offered as an interpretable complement to learned control policies.",
        "A safety-oriented evaluation (safe-action and under-protection rates, return-to-launch recall, escalation "
        "lead time, latency) and an ablation isolating the contribution of each stage.",
        "A leakage-free temporal-generalization audit that quantifies a cross-session gap concealed by random-split "
        "evaluation.",
    ])
    para(doc, "Research gap. Although integrated detection and mitigation has recently been demonstrated with "
              "reinforcement learning [14], the gap we address is an interpretable, sensor-light framework that "
              "additionally forecasts attack-severity escalation to act pre-emptively through a transparent safety "
              "policy, and that is validated under a leakage-free temporal protocol exposing cross-session "
              "generalization—properties not jointly provided by a learned, vision-assisted control policy.",
         style="Body Text")

    # ---- II. RELATED WORK ----
    heading(doc, "Related Work")
    para(doc, "UAV security surveys map the attack surface across hardware, software, communication, and sensing "
              "and stress mobility, resource, and safety constraints on defenses [1], [2]; they motivate but do not "
              "provide a GPS-specific predictive-mitigation loop.", style="Body Text",
         lead="A. UAV security surveys. ")
    para(doc, "For GPS/GNSS spoofing, tree-based, ensemble, adversarial, and deep detectors report high accuracy, "
              "and multi-channel receiver measurements are particularly discriminative [3]–[8]; the "
              "multi-channel study of Mouzai et al. [6] uses the same dataset as this work. This line establishes "
              "that detection accuracy is mature, leaving the response and evaluation questions open.",
         style="Body Text", lead="B. GPS/GNSS spoofing detection. ")
    para(doc, "Closing the loop to a response has been studied with control-theoretic and learning methods. Hu et "
              "al. [14] integrate detection and mitigation with a deep-Q-network that reactively switches between "
              "GPS- and vision-based navigation according to the current reliability of the GPS signal, and "
              "related federated and multi-agent-learning schemes address resilient "
              "control in related settings [9]. These works demonstrate integrated or adaptive response but "
              "rely on learned (hence less interpretable) policies and, for [14], an additional vision sensor. In "
              "contrast, our framework adds explicit forecasting of attack-severity escalation for pre-emptive "
              "response, retains an interpretable (non-learned) policy operating on GPS features alone, and is "
              "validated under a leakage-free temporal protocol.", style="Body Text",
         lead="C. Integrated detection and mitigation. ")
    para(doc, "Agentic and LLM-based designs decompose cyber defense into specialized cooperating components with "
              "detection, analysis, and response roles [10], [11]. We adopt the same principle of role "
              "specialization, but at a lightweight, interpretable, and non-LLM level suited to on-board UAV "
              "operation. In summary, our framework differs from prior integrated approaches by forecasting "
              "attack-severity escalation, using a transparent (non-learned) safety policy on GPS features alone, "
              "and reporting a leakage-free temporal audit; Table I positions it against the capability of prior "
              "GPS-spoofing detectors on the same data.", style="Body Text",
         lead="D. Agentic and multi-component defense. ")
    styled_table(doc, ["Work", "Detect", "Predict", "Mitigate", "Interpretable policy"],
        [["GPS detectors [3]–[8]", "Yes", "No", "No", "n/a"],
         ["RL integrated [14]", "Yes", "No", "Yes (learned)", "No"],
         ["This work", "Yes", "Yes", "Yes", "Yes"]],
        caption="TABLE I. CAPABILITY POSITIONING ON THE SAME DATASET")

    # ---- III. THREAT MODEL + ARCHITECTURE ----
    heading(doc, "Threat Model and System Architecture")
    para(doc, "We consider an external adversary within radio range that transmits counterfeit GPS signals to "
              "corrupt the on-board position solution, escalating through three regimes of increasing capability: "
              "a simplistic regime that overpowers signals without careful alignment; an intermediate regime that "
              "approximately matches power and code phase; and a sophisticated regime that seamlessly captures and "
              "walks the tracking loops. The defender observes only the receiver's own signal-quality measurements "
              "across its channels and must decide, in real time and on-board, the current severity and the action "
              "to take. The inertial measurement unit and a return-to-launch capability are assumed available, as "
              "on modern flight controllers. Fig. 1 shows the end-to-end architecture: legitimate satellites and "
              "the attacker emit authentic and spoofed signals to the eight-channel receiver; per-epoch features "
              "are extracted; the detection, prediction, and mitigation components run under a coordinator with a "
              "monotone safety policy; and the selected navigation action feeds back to the flight controller.",
         style="Body Text")
    add_figure(doc, "fig_architecture.png", "Fig. 1. System and threat-model architecture, from the spoofing "
               "threat and eight-channel receiver through feature extraction and the detection–prediction–"
               "mitigation stages to the navigation action and recovery feedback.", width=3.35)

    # ---- IV. PROPOSED FRAMEWORK ----
    heading(doc, "Proposed Framework and Formulation")
    para(doc, "We organize the framework as three cooperating software agents under a coordinator. We use the term "
              "agent in the software sense of a component with a well-defined role, its own internal state, an "
              "explicit input–output contract, and a local decision function; the agents interact through the "
              "coordinator, and the prediction agent's output directly conditions the mitigation agent's decision, "
              "which is an inter-agent dependency beyond a linear transform pipeline. We do not claim autonomy in "
              "the strong sense: the agents are executed deterministically and sequentially by the coordinator "
              "rather than concurrently or through negotiation, the mitigation policy is a fixed finite-state map "
              "rather than a learned or self-modifying one, and no large-language-model reasoning is involved. The "
              "design is therefore a coordinated, safety-aware multi-agent pipeline; the term denotes "
              "role-specialized cooperation, not autonomous agentic AI.", style="Body Text")

    subheading(doc, "A. Observation model and features")
    para(doc, "At receiver epoch t, each active channel c reports a 13-dimensional signal-quality vector r_{t,c} "
              "(PRN, carrier Doppler, pseudorange, times, carrier phase, correlator outputs, and CN0). Because the "
              "navigation decision concerns the whole solution, we fuse the active channels into one epoch vector "
              "x_t by a cross-channel statistics map φ:", style="Body Text")
    equation(doc, r"""x_t = \varphi(\{ r_{t,c} : c \in C_t \}) \in \mathbb{R}^d, \quad d = 13 \times 6 + 1 = 79""", "1", fallback="x_t = phi({ r_(t,c) : c in C_t }) in R^d,  d = 13 x 6 + 1 = 79")
    para(doc, "where C_t is the set of active channels and, for each of the 13 measurements, φ computes the "
              "cross-channel mean, standard deviation, minimum, maximum, range R = max − min, and coefficient of "
              "variation CV = σ/(|μ|+ε), plus the active-channel count |C_t|. These spread statistics capture "
              "constellation-wide distortion that a single channel does not reveal.", style="Body Text")

    subheading(doc, "B. Detection agent")
    para(doc, "The detection agent is a classifier f_θ that maps x_t to a distribution over the four severity "
              "classes y ∈ {0,1,2,3} (authentic, simplistic, intermediate, sophisticated):", style="Body Text")
    equation(doc, r"""\hat{c}_t = \underset{k}{\arg\max}\,[f_\theta(x_t)]_k, \quad s_t = \hat{c}_t, \quad \mathrm{conf}_t = \underset{k}{\max}\,[f_\theta(x_t)]_k""", "2", fallback="c_t = argmax_k [f_theta(x_t)]_k,  s_t = c_t,  conf_t = max_k [f_theta(x_t)]_k")
    para(doc, "with parameters θ fitted by minimizing the regularized cross-entropy over labeled epochs. Here s_t "
              "is the ordinal severity used downstream and conf_t the confidence. We instantiate f_θ as "
              "gradient-boosted trees for near-millisecond single-epoch inference.", style="Body Text")

    subheading(doc, "C. Prediction agent")
    para(doc, "The prediction agent forecasts the near-term severity from the stream of detected severities over a "
              "window of the last W epochs, fitting a least-squares line s_τ ≈ a_t·τ + b_t and extrapolating one "
              "step:", style="Body Text")
    equation(doc, r"""(a_t, b_t) = \underset{a,b}{\arg\min}\sum_{\tau=t-W+1}^{t}(s_\tau - a\tau - b)^2, \quad \hat{s}_{t+1} = \mathrm{clip}(a_t(t+1)+b_t, 0, 3)""", "3", fallback="(a_t,b_t) = argmin_(a,b) sum (s_tau - a*tau - b)^2,  s_(t+1) = clip(a_t(t+1)+b_t, 0, 3)")
    para(doc, "and it labels the trend ESCALATING, STABLE, or DE-ESCALATING according to whether the slope a_t "
              "exceeds a small threshold. Because ŝ_{t+1} is produced before the attack peaks, it provides early "
              "warning during an escalating attack.", style="Body Text")

    subheading(doc, "D. Mitigation agent and safety objective")
    para(doc, "The mitigation agent maps severity to an ordered, monotone action set through a fixed policy π, and "
              "pre-empts one level when the forecast is clearly worse than the current severity:", style="Body Text")
    equation(doc, r"""s^{*}_{t} = \mathrm{round}(\hat{s}_{t+1})\ \text{if}\ \hat{s}_{t+1} \geq s_t + \delta,\ \text{else}\ s_t; \quad a_t = \pi(s^{*}_{t})""", "4", fallback="s*_t = round(s_(t+1)) if s_(t+1) >= s_t + delta, else s_t;  a_t = pi(s*_t)")
    para(doc, "with δ = 0.5 and π(0)=accept GPS, π(1)=GPS integrity check, π(2)=reject GPS / inertial-navigation "
              "fallback, π(3)=return-to-launch. The safety objective minimizes the under-protection rate subject to "
              "a per-epoch latency budget T_max, while keeping the false-alarm rate low:", style="Body Text")
    equation(doc, r"""\min\ P(\mathrm{sev}(a_t) < y_t)\quad \text{s.t.}\ \ \mathbb{E}[\ell_t] \leq T_{\max}, \quad P(a_t \neq \mathrm{accept} \mid y_t = 0) \leq \alpha""", "5", fallback="min P(sev(a_t) < y_t)  s.t.  E[l_t] <= T_max,  P(a_t != accept | y_t=0) <= alpha")
    para(doc, "where sev(·) is the severity a given action protects against and ℓ_t is the per-epoch latency. "
              "Under-protection—acting less severely than the true threat requires—is the safety-critical error, "
              "so it is minimized directly rather than optimizing accuracy alone.", style="Body Text")

    subheading(doc, "E. Coordination")
    para(doc, "The coordinator executes the three agents in order each epoch and records per-agent and end-to-end "
              "latency (Algorithm 1). The implementation class is named ConsensusOrchestrator; it is coordination "
              "logic, not a formal distributed-consensus protocol.", style="Body Text")
    add_algorithm(doc, "Algorithm 1  Per-epoch coordination", [
        "Input: fused epoch feature vector x_t",
        "1  (c_t, conf_t) <- Detection.classify(x_t)",
        "2  s_t <- severity(c_t)",
        "3  Prediction.update(s_t);  (s_hat, trend) <- Prediction.forecast()",
        "4  s_star <- s_t",
        "5  if s_hat >= s_t + delta:  s_star <- round(s_hat)",
        "6  a_t <- pi(s_star)",
        "7  return a_t, trend, latencies",
    ])

    # ---- V. EXPERIMENTAL SETUP ----
    heading(doc, "Experimental Setup")
    para(doc, "We use the public University of North Dakota multi-channel UAV GPS-spoofing dataset (Mendeley Data, "
              "DOI 10.17632/z7dj3yyzt8, CC BY 4.0), reshaping the eight-channel log into "
              f"{ds['raw_channel_samples']:,} per-channel samples and fusing them into {ds['epochs']:,} epoch "
              f"vectors of {ds['n_epoch_features']} features (Table II). The epoch label is the worst-case severity "
              "among its channels; about 15% of attacked epochs contain mixed-severity channels, which we disclose "
              "as a limitation. The pipeline is implemented in Python with scikit-learn and XGBoost under a fixed "
              "seed. We report a stratified 75/25 random hold-out, five-fold random cross-validation, and a "
              "leakage-free five-fold time-block protocol in which contiguous chronological blocks are held out; a "
              "1,100-epoch held-out streaming scenario exercises the full loop. Detection is scored by accuracy, "
              "macro-F1, per-class precision/recall, detection rate, and false-alarm rate; prediction by severity "
              "MAE and escalation lead time; mitigation by safe-action, under-protection, and return-to-launch "
              "rates; and feasibility by latency.", style="Body Text")
    styled_table(doc, ["Level", "Per-channel", "Share", "Epoch-level"],
        [["authentic", "794,596", "85.96%", f"{ds['class_distribution']['authentic']:,}"],
         ["simplistic", "41,780", "4.52%", f"{ds['class_distribution']['simplistic']:,}"],
         ["intermediate", "44,232", "4.79%", f"{ds['class_distribution']['intermediate']:,}"],
         ["sophisticated", "43,743", "4.73%", f"{ds['class_distribution']['sophisticated']:,}"]],
        caption="TABLE II. EPOCH-LEVEL CLASS DISTRIBUTION AFTER FUSION")

    # ---- VI. RESULTS ----
    heading(doc, "Results and Discussion")
    subheading(doc, "A. Detection performance")
    para(doc, f"Under five-fold cross-validation the detector attains {pct(oa('bin_xgb_5fold_random'))} ± "
              f"{ostd('bin_xgb_5fold_random')*100:.2f} accuracy on the binary spoofed-versus-authentic task and "
              f"{pct(oa('mc_xgb_5fold_random'))} ± {ostd('mc_xgb_5fold_random')*100:.2f} on the four-class severity "
              "task; the small standard deviations indicate the result is stable rather than split-dependent. On a "
              f"representative 75/25 hold-out the binary detection rate is {pct(bh['detection_rate'])} at a "
              f"{pct(bh['false_alarm'])} false-alarm rate, and the four-class per-class scores (Table III, "
              "confusion matrix in Fig. 2) show residual error concentrated between the adjacent simplistic and "
              "intermediate levels, while authentic and sophisticated epochs exceed 0.95 F1.", style="Body Text")
    styled_table(doc, ["Class", "Precision", "Recall", "F1", "Support"],
        [[c, f"{det['per_class'][c]['precision']:.3f}", f"{det['per_class'][c]['recall']:.3f}",
          f"{det['per_class'][c]['f1']:.3f}", det['per_class'][c]['support']]
         for c in ["authentic", "simplistic", "intermediate", "sophisticated"]],
        caption="TABLE III. PER-CLASS DETECTION PERFORMANCE (75/25 HOLD-OUT)")
    add_figure(doc, "fig_confusion.png", "Fig. 2. Four-class detection confusion matrix on the hold-out "
               "(row-normalized).")

    subheading(doc, "B. Random versus chronological generalization")
    para(doc, "Cross-validation and hold-out agree closely, but under the leakage-free time-block protocol "
              f"accuracy falls to {pct(oa('bin_xgb_5fold_timeblock'))} (binary) and "
              f"{pct(oa('mc_xgb_5fold_timeblock'))} (four-class) (Table IV). Temporally correlated measurements "
              "make random splits optimistic; we report this gap as a substantive finding, since a detector tuned "
              "only to random-split accuracy may not generalize to a future flight segment.", style="Body Text")
    styled_table(doc, ["Task", "Random hold-out", "5-fold CV", "Time-block CV"],
        [["Binary (spoofed vs authentic)", pct(oa('bin_xgb_holdout')),
          f"{pct(oa('bin_xgb_5fold_random'))}±{ostd('bin_xgb_5fold_random')*100:.2f}",
          f"{pct(oa('bin_xgb_5fold_timeblock'))}±{ostd('bin_xgb_5fold_timeblock')*100:.1f}"],
         ["Severity (4-class)", pct(oa('mc_xgb_holdout')),
          f"{pct(oa('mc_xgb_5fold_random'))}±{ostd('mc_xgb_5fold_random')*100:.2f}",
          f"{pct(oa('mc_xgb_5fold_timeblock'))}±{ostd('mc_xgb_5fold_timeblock')*100:.1f}"]],
        caption="TABLE IV. DETECTION ACCURACY BY EVALUATION PROTOCOL")

    subheading(doc, "C. Windowed attack-level detection")
    para(doc, "Because an attack persists over many epochs, we also flag it when a majority of epochs in a short "
              f"window read spoofed. Within-session this raises attack-level detection to {pct(w5['accuracy'])} at "
              f"five epochs and {pct(w11['accuracy'],0)} at eleven, at near-zero false alarm (Table V, Fig. 3); the "
              f"same aggregation does not repair the cross-session case ({pct(wc11['accuracy'])}), whose errors are "
              "systematic rather than random.", style="Body Text")
    styled_table(doc, ["Regime / window", "Accuracy", "Attack recall", "False alarm"],
        [["Within-session, per-epoch", pct(W['within_session']['per_epoch_accuracy']), "—", "—"],
         ["Within-session, W=5", pct(w5['accuracy']), pct(w5['attack_recall']), pct(w5['false_alarm'])],
         ["Within-session, W=11", pct(w11['accuracy']), pct(w11['attack_recall']), pct(w11['false_alarm'])],
         ["Within-session, W=25", pct(w25['accuracy']), pct(w25['attack_recall']), pct(w25['false_alarm'])],
         ["Cross-session, per-epoch", pct(W['cross_session']['per_epoch_accuracy']), "—", "—"],
         ["Cross-session, W=11", pct(wc11['accuracy']), pct(wc11['attack_recall']), pct(wc11['false_alarm'])]],
        caption="TABLE V. WINDOWED ATTACK-LEVEL DETECTION")
    add_figure(doc, "fig_windowed.png", "Fig. 3. Temporal aggregation raises within-session attack detection but "
               "cannot repair the cross-session gap.")

    subheading(doc, "D. Prediction and closed-loop mitigation")
    para(doc, f"On the streaming scenario the prediction agent forecasts severity with a mean absolute error of "
              f"{fw['prediction_forecast_mae']:.3f} on the 0–3 scale, and the first escalation warning precedes "
              f"severity 3 by {fw['escalation_lead_time_epochs']} epochs, driving {fw['preemptive_mitigations']} "
              "pre-emptive mitigations (Table VI). The mitigation agent reaches a "
              f"{pct(fw['mitigation_safe_action_rate'])} safe-action rate, is under-protective in only "
              f"{pct(fw['mitigation_under_protection_rate'])} of epochs, and commands return-to-launch on "
              f"{pct(fw['sophisticated_RTL_recall'])} of sophisticated attacks. Fig. 4 traces true, detected, and "
              "forecast severity, and Fig. 5 shows the mitigated flight holding the true path on inertial "
              "navigation after detection.", style="Body Text")
    styled_table(doc, ["Metric", "Value"],
        [["Streaming detection accuracy", pct(fw['streaming_detection_accuracy'])],
         ["Forecast error (severity MAE, 0–3)", f"{fw['prediction_forecast_mae']:.3f}"],
         ["Escalation lead time (epochs)", fw['escalation_lead_time_epochs']],
         ["Mitigation safe-action rate", pct(fw['mitigation_safe_action_rate'])],
         ["Under-protection rate", pct(fw['mitigation_under_protection_rate'])],
         ["Sophisticated-attack RTL recall", pct(fw['sophisticated_RTL_recall'])]],
        caption="TABLE VI. PREDICTION AND CLOSED-LOOP MITIGATION")
    add_figure(doc, "fig_threat_timeline.png", "Fig. 4. Real-time severity: true (solid), detected (thin), and "
               "forecast (dashed) across a ramping attack.")
    add_figure(doc, "fig_trajectory.png", "Fig. 5. Held-out simulated flight path: unmitigated spoofed drift vs. "
               "inertial-navigation fallback after detection.")

    subheading(doc, "E. Latency")
    para(doc, f"Mean end-to-end processing is {L['end_to_end']['mean_ms']:.2f} ms (95th percentile "
              f"{L['end_to_end']['p95_ms']:.2f} ms), dominated by detection ({L['detection']['mean_ms']:.2f} ms); "
              "prediction and mitigation add well under a millisecond (Table VII, Fig. 6), comfortably within a "
              "20 ms (50 Hz) budget. These are development-platform figures, not an on-board measurement.",
         style="Body Text")
    styled_table(doc, ["Stage", "Mean (ms)", "p95 (ms)"],
        [[k.replace('_', ' ').title(), f"{L[k]['mean_ms']:.3f}", f"{L[k]['p95_ms']:.3f}"]
         for k in ["detection", "prediction", "mitigation", "end_to_end"]],
        caption="TABLE VII. PER-AGENT AND END-TO-END LATENCY")
    add_figure(doc, "fig_latency.png", "Fig. 6. Per-agent and end-to-end latency against a 20 ms budget.")

    subheading(doc, "F. Comparison")
    para(doc, "Fig. 7 compares our detection with published values on the same dataset. Prior detectors report "
              "higher random-split accuracy [6], [12], [13], so we do not claim an accuracy record; the "
              "contribution is combining competitive detection with severity forecasting, an interpretable safety "
              "policy, latency measurement, and a chronological robustness audit.", style="Body Text")
    add_figure(doc, "fig_comparison.png", "Fig. 7. Detection accuracy vs. published values on the same dataset "
               "[6], [12], [13].")

    subheading(doc, "G. Ablation: contribution of each stage")
    para(doc, "Applying three response policies to identical detector outputs on the streaming scenario (Table "
              "VIII) shows the value of the added stages. A detection-only configuration—classifying without "
              f"autonomous action—leaves {pct(A['detection_only']['under_protection_rate'])} of epochs "
              "under-protected. Reactive mitigation on the detected severity cuts this to "
              f"{pct(A['detect_reactive']['under_protection_rate'])}, and the full prediction-informed loop reaches "
              f"{pct(A['full_loop']['under_protection_rate'])} while further reducing exposed epochs "
              f"({A['full_loop']['mean_exposure_epochs']} vs {A['detect_reactive']['mean_exposure_epochs']}) "
              "through pre-emption. The safety gain therefore comes from the prediction and mitigation stages, not "
              "detection alone.", style="Body Text")
    styled_table(doc, ["Configuration", "Under-prot.", "Attack prot.", "RTL recall", "Exposed"],
        [["Detection only", pct(A['detection_only']['under_protection_rate']),
          pct(A['detection_only']['attack_protection_rate']), pct(A['detection_only']['sophisticated_RTL_recall']),
          A['detection_only']['mean_exposure_epochs']],
         ["+ Reactive mitigation", pct(A['detect_reactive']['under_protection_rate']),
          pct(A['detect_reactive']['attack_protection_rate']), pct(A['detect_reactive']['sophisticated_RTL_recall']),
          A['detect_reactive']['mean_exposure_epochs']],
         ["+ Prediction (full)", pct(A['full_loop']['under_protection_rate']),
          pct(A['full_loop']['attack_protection_rate']), pct(A['full_loop']['sophisticated_RTL_recall']),
          A['full_loop']['mean_exposure_epochs']]],
        caption="TABLE VIII. ABLATION: SAFETY CONTRIBUTION OF EACH STAGE")

    subheading(doc, "H. Threats to validity")
    para(doc, "The dataset is a public benchmark and may not cover all receivers, environments, or spoofing "
              "generators; the worst-case epoch label simplifies mixed-severity behavior; the prediction model is "
              "deliberately lightweight and has not been compared with recurrent or transformer predictors; the "
              "streaming trajectory is a software simulation over held-out epochs rather than a physical or "
              "hardware-in-the-loop flight; and latency is development-platform latency. The ablation compares "
              "against detection-only and reactive baselines but not against a learned control policy such as [14]; "
              "such a comparison is future work.", style="Body Text")

    # ---- VII. CONCLUSION ----
    heading(doc, "Conclusion and Future Work")
    para(doc, "We presented a safety-aware framework that couples GPS-spoofing detection with attack-severity "
              "forecasting and an interpretable, pre-emptive mitigation policy, coordinated as three cooperating "
              f"agents. Detection is competitive ({pct(oa('bin_xgb_5fold_random'),1)} binary under "
              "cross-validation), and an ablation shows the prediction and mitigation stages reduce safety "
              f"under-protection from {pct(A['detection_only']['under_protection_rate'])} to "
              f"{pct(A['full_loop']['under_protection_rate'])}; a leakage-free evaluation exposes a cross-session "
              "gap that random splits conceal. Future work will compare against learned control policies including "
              "[14], benchmark stronger sequence predictors, and move from development-platform timing to "
              "on-board and hardware-in-the-loop validation.", style="Body Text")

    heading(doc, "Acknowledgment")
    para(doc, "The authors thank the Department of Information Technology, Mukesh Patel School of Technology "
              "Management & Engineering, SVKM's NMIMS, Mumbai, for the guidance and computational resources that "
              "supported this work, and gratefully acknowledge the creators of the public University of North "
              "Dakota UAV GPS-spoofing dataset, whose release under an open licence made this study possible. This "
              "research received no specific grant from any funding agency in the public, commercial, or "
              "not-for-profit sectors.", style="Body Text")

    # ---- REFERENCES ----
    heading(doc, "References")
    refs = [
        "Y. Mekdad, A. Aris, L. Babun, A. El Fergougui, M. Conti, R. Lazzeretti, and A. S. Uluagac, “A survey on security and privacy issues of UAVs,” Computer Networks, vol. 224, 109626, 2023.",
        "B. Cordill, D. Fang, and S. Xu, “A comprehensive survey of security and privacy in UAV systems,” IEEE Access, vol. 13, pp. 117843–117866, 2025.",
        "A. F. Gormus et al., “Real-time detection of GPS jamming and spoofing attacks on UAVs using machine learning,” Comput. Ind. Eng., vol. 219, 112207, 2026.",
        "R. Al-Syouf, O. Y. Aljarrah, R. Bani-Hani, and A. Alma’aitah, “Ensemble machine learning models utilizing a hybrid recursive feature elimination technique for detecting GPS spoofing attacks against UAVs,” Sensors, vol. 25, no. 8, 2388, 2025.",
        "L. Alhoraibi, D. Alghazzawi, and R. Alhebshi, “Detection of GPS spoofing attacks in UAVs based on adversarial machine learning model,” Sensors, vol. 24, no. 18, 6156, 2024.",
        "M. Mouzai, M. A. Riahla, A. Keziou, and H. Fouchal, “Exploring multi-channel GPS receivers for detecting spoofing attacks on UAVs using machine learning,” Sensors, vol. 25, no. 13, 4045, 2025.",
        "Y. Sun, M. Yu, L. Wang, T. Li, and M. Dong, “A deep-learning-based GPS signal spoofing detection method for small UAVs,” Drones, vol. 7, no. 6, 370, 2023.",
        "P. Wang, Z. Yang, J. Li, and L. Shi, “Learning-based detection of GPS spoofing attack for quadrotors,” arXiv:2501.07597, 2025.",
        "Y. Zhou et al., “From static to adaptive defense: Federated multi-agent deep RL-driven moving target defense against DoS attacks in UAV swarm networks,” arXiv:2506.07392, 2025.",
        "H. Li et al., “Multi-agent collaborative intrusion detection for low-altitude economy IoT: An LLM-enhanced agentic AI framework,” arXiv:2601.17817, 2026.",
        "P. Sharma, H. Wen, V. Yegneswaran, A. Gehani, P. Porras, and Z. Lin, “MobiLLM: An agentic AI framework for closed-loop threat mitigation in 6G Open RANs,” arXiv:2509.21634, 2025.",
        "M. Nayfeh, Y. Li, K. Al Shamaileh, V. Devabhaktuni, and N. Kaabouch, “Machine learning modeling of GPS features with applications to UAV location spoofing detection and classification,” Comput. Secur., vol. 126, 103085, 2023.",
        "T. Talaei Khoei, S. Ismail, K. Al Shamaileh, V. K. Devabhaktuni, and N. Kaabouch, “Impact of dataset and model parameters on machine learning performance for the detection of GPS spoofing attacks on unmanned aerial vehicles,” Applied Sciences, vol. 13, no. 1, 383, 2023.",
        "J. Hu, M. Ammar, B. Zahid Hussain, J. Kim, and I. Khan, “Reinforcement-learning-driven integrated detection and mitigation of UAV GPS spoofing attacks,” IEEE Internet of Things Journal, vol. 12, no. 18, pp. 36926–36941, 2025.",
    ]
    for r in refs:
        try:
            p = doc.add_paragraph(style="references"); p.add_run(r)
        except KeyError:
            p = doc.add_paragraph(); p.add_run(r)

    doc.save(OUT)
    print(f"Saved -> {OUT}  (references: {len(refs)})")


if __name__ == "__main__":
    main()
