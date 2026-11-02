import time
import numpy as np
from sklearn.linear_model import Ridge

class PredictionAgent:
    def __init__(self, history_len=15):
        self.name = "Prediction Agent"
        self.history_len = history_len
        self.anomaly_history = []
        self.model = Ridge(alpha=1.0)
        self.is_fitted = False
        
    def add_observation(self, score):
        self.anomaly_history.append(score)
        if len(self.anomaly_history) > self.history_len:
            self.anomaly_history.pop(0)
            
    def predict_next_threat_level(self):
        t_start = time.perf_counter()
        
        if len(self.anomaly_history) < 5:
            # Not enough history to forecast
            t_latency = (time.perf_counter() - t_start) * 1000.0
            return {"predicted_threat_score": 0.0, "trend": "STABLE", "latency_ms": t_latency}
            
        # Fit a simple autoregressive linear model on the sliding window history to predict the next value
        # Y = [score_t], X = [t]
        y = np.array(self.anomaly_history)
        x = np.arange(len(y)).reshape(-1, 1)
        
        self.model.fit(x, y)
        next_step = np.array([[len(y)]])
        predicted_score = float(self.model.predict(next_step)[0])
        predicted_score = np.clip(predicted_score, 0.0, 1.0)
        
        # Calculate slope/trend
        slope = float(self.model.coef_[0])
        if slope > 0.05:
            trend = "INCREASING"
        elif slope < -0.05:
            trend = "DECREASING"
        else:
            trend = "STABLE"
            
        t_latency = (time.perf_counter() - t_start) * 1000.0
        return {
            "predicted_threat_score": predicted_score,
            "trend": trend,
            "latency_ms": t_latency
        }
