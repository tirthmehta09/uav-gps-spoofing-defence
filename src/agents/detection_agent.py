import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class DetectionAgent:
    def __init__(self):
        self.name = "Detection Agent"
        # Using a fast RF classifier
        self.model = RandomForestClassifier(n_estimators=30, max_depth=12, random_state=42, n_jobs=-1)
        self.label_encoder = LabelEncoder()
        self.feature_cols = None
        self.is_trained = False
        
    def train(self, df_train):
        print(f"[{self.name}] Training model on {len(df_train)} rows...")
        start_time = time.perf_counter()
        
        # Exclude metadata and target column
        self.feature_cols = [c for c in df_train.columns if c not in ['timestamp_c', 'class', 'frame.protocols']]
        
        X = df_train[self.feature_cols].values
        y = self.label_encoder.fit_transform(df_train['class'].values)
        
        self.model.fit(X, y)
        self.is_trained = True
        
        duration = time.perf_counter() - start_time
        print(f"[{self.name}] Training complete in {duration:.2f} seconds.")
        
    def evaluate_packet(self, packet_features_dict):
        t_start = time.perf_counter()
        
        if not self.is_trained:
            return {"predicted_class": "benign", "confidence": 1.0, "latency_ms": 0.0}
            
        # Convert raw dict features to numpy array
        feat_vector = np.array([packet_features_dict.get(col, 0.0) for col in self.feature_cols])
        feat_vector = np.nan_to_num(feat_vector).reshape(1, -1)
        
        pred_idx = self.model.predict(feat_vector)[0]
        probs = self.model.predict_proba(feat_vector)[0]
        
        pred_class = self.label_encoder.classes_[pred_idx]
        confidence = float(probs[pred_idx])
        
        t_latency = (time.perf_counter() - t_start) * 1000.0
        return {
            "predicted_class": pred_class,
            "confidence": confidence,
            "latency_ms": t_latency
        }
