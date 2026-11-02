"""
Multi-Agent Agentic AI Framework for UAV GPS Spoofing defense.

Three cooperating agents, coordinated by an Orchestrator, operate on a stream of
GPS receiver epochs (one epoch = one time step, up to 8 channels fused):

  DetectionAgent   -- multi-class classifier: authentic / simplistic /
                      intermediate / sophisticated spoofing (+ confidence).
  PredictionAgent  -- sliding-window autoregression over the detected severity
                      stream; forecasts the near-future threat level and trend,
                      giving early warning before an attack fully manifests.
  MitigationAgent  -- finite-state policy mapping the (current, forecast)
                      severity to a GPS-specific safety action:
                      accept GPS -> integrity cross-check -> switch to inertial
                      navigation (reject GPS) -> emergency Return-To-Launch.

The Orchestrator fuses detection + forecast into a consensus threat each epoch,
invokes mitigation, and records per-agent + end-to-end latency.

The DetectionAgent takes any fitted-or-unfitted sklearn-style estimator, so the
detection model/representation chosen from the benchmark can be swapped in.
"""
import time
from collections import deque

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Ordinal severity ladder for the four classes.
SEVERITY = {"authentic": 0, "simplistic": 1, "intermediate": 2, "sophisticated": 3}
SEVERITY_NAME = {v: k for k, v in SEVERITY.items()}


class DetectionAgent:
    """Multi-class spoofing detector. Wraps any sklearn estimator with predict_proba."""

    def __init__(self, model=None):
        self.name = "Detection Agent"
        self.model = model if model is not None else RandomForestClassifier(
            n_estimators=200, max_depth=None, n_jobs=-1, random_state=42)
        self.le = LabelEncoder()
        self.classes_ = None
        self.feature_cols = None
        self.is_trained = False

    def train(self, X, y_names, feature_cols=None):
        self.feature_cols = feature_cols
        y = self.le.fit_transform(np.asarray(y_names))
        self.model.fit(X, y)
        self.classes_ = list(self.le.classes_)
        self.is_trained = True
        return self

    def predict_batch(self, X):
        """Vectorised classification for offline metric computation."""
        idx = self.model.predict(X)
        return self.le.inverse_transform(idx)

    def detect(self, x_row):
        """Classify one epoch. Returns class name, ordinal severity, confidence, latency."""
        t = time.perf_counter()
        x = np.asarray(x_row, dtype=float).reshape(1, -1)
        probs = self.model.predict_proba(x)[0]
        idx = int(np.argmax(probs))
        cls = self.le.inverse_transform([idx])[0]
        return {
            "class": cls,
            "severity": SEVERITY.get(cls, 0),
            "confidence": float(probs[idx]),
            "latency_ms": (time.perf_counter() - t) * 1000.0,
        }


class PredictionAgent:
    """Sliding-window autoregressive forecaster over the detected-severity stream."""

    def __init__(self, window=25, min_history=5, slope_threshold=0.01):
        self.name = "Prediction Agent"
        self.window = window
        self.min_history = min_history
        self.slope_threshold = slope_threshold
        self.hist = deque(maxlen=window)

    def update(self, severity):
        self.hist.append(float(severity))

    def forecast(self):
        """Forecast next-epoch severity and trend from a least-squares line fit over the window."""
        t = time.perf_counter()
        n = len(self.hist)
        if n < self.min_history:
            last = self.hist[-1] if self.hist else 0.0
            return {"forecast_severity": float(last), "trend": "STABLE", "slope": 0.0,
                    "latency_ms": (time.perf_counter() - t) * 1000.0}
        y = np.fromiter(self.hist, dtype=float)
        x = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        forecast = float(np.clip(slope * n + intercept, 0.0, 3.0))
        if slope > self.slope_threshold:
            trend = "ESCALATING"
        elif slope < -self.slope_threshold:
            trend = "DE-ESCALATING"
        else:
            trend = "STABLE"
        return {"forecast_severity": forecast, "trend": trend, "slope": float(slope),
                "latency_ms": (time.perf_counter() - t) * 1000.0}


class MitigationAgent:
    """Finite-state GPS safety policy driven by (current, forecast) severity."""

    # severity -> (system_state, physical_action)
    POLICY = {
        0: ("NORMAL_GPS_NAV", "ACCEPT_GPS"),
        1: ("HEIGHTENED_MONITORING", "CROSS_CHECK_GPS_INTEGRITY"),
        2: ("GPS_REJECT", "SWITCH_TO_INERTIAL_NAVIGATION"),
        3: ("EMERGENCY", "RETURN_TO_LAUNCH"),
    }

    def __init__(self, preempt_margin=0.5):
        self.name = "Mitigation Agent"
        self.state = "NORMAL_GPS_NAV"
        self.preempt_margin = preempt_margin

    def mitigate(self, severity, forecast_severity, confidence):
        """Act on current severity, pre-empting one level early if the forecast is clearly worse."""
        t = time.perf_counter()
        effective = int(severity)
        # Early action: if the forecast trend clearly points higher, escalate pre-emptively.
        if forecast_severity >= severity + self.preempt_margin:
            effective = int(min(3, round(forecast_severity)))
        effective = int(np.clip(effective, 0, 3))
        state, action = self.POLICY[effective]
        self.state = state
        return {"system_state": state, "action": action, "effective_severity": effective,
                "preemptive": effective > int(severity),
                "latency_ms": (time.perf_counter() - t) * 1000.0}


class Orchestrator:
    """Coordinates the three agents per epoch and returns a consensus decision."""

    def __init__(self, detection_agent, prediction_agent, mitigation_agent):
        self.name = "Consensus Orchestrator"
        self.detection = detection_agent
        self.prediction = prediction_agent
        self.mitigation = mitigation_agent

    def step(self, x_row):
        t = time.perf_counter()
        det = self.detection.detect(x_row)
        self.prediction.update(det["severity"])
        pred = self.prediction.forecast()
        mit = self.mitigation.mitigate(det["severity"], pred["forecast_severity"], det["confidence"])

        # Consensus threat: the action the system actually commits to this epoch.
        consensus_severity = mit["effective_severity"]
        return {
            "detected_class": det["class"],
            "detected_severity": det["severity"],
            "confidence": det["confidence"],
            "forecast_severity": pred["forecast_severity"],
            "trend": pred["trend"],
            "consensus_severity": consensus_severity,
            "consensus_class": SEVERITY_NAME[consensus_severity],
            "system_state": mit["system_state"],
            "mitigation_action": mit["action"],
            "preemptive": mit["preemptive"],
            "latencies": {
                "detection_ms": det["latency_ms"],
                "prediction_ms": pred["latency_ms"],
                "mitigation_ms": mit["latency_ms"],
                "total_ms": (time.perf_counter() - t) * 1000.0,
            },
        }
