import time

class ConsensusOrchestrator:
    def __init__(self, detection_agent, prediction_agent, mitigation_agent):
        self.name = "Consensus Orchestrator"
        self.detection_agent = detection_agent
        self.prediction_agent = prediction_agent
        self.mitigation_agent = mitigation_agent
        
    def process_packet(self, packet_features):
        t_start = time.perf_counter()
        
        # 1. Detection Agent Classification
        det_res = self.detection_agent.evaluate_packet(packet_features)
        
        # 2. Update Prediction Agent History
        # We define a feature-based anomaly score = 1.0 - benign_probability
        # if predicted class is benign, anomaly score is 1.0 - confidence
        # if predicted class is attack, anomaly score is confidence
        is_attack = det_res['predicted_class'] != 'benign'
        score = det_res['confidence'] if is_attack else (1.0 - det_res['confidence'])
        self.prediction_agent.add_observation(score)
        
        # 3. Prediction Agent Forecasting
        pred_res = self.prediction_agent.predict_next_threat_level()
        
        # 4. Consensus voting logic
        confirmed_threat = "benign"
        consensus_confidence = 0.0
        
        if is_attack:
            confirmed_threat = det_res['predicted_class']
            consensus_confidence = det_res['confidence']
            # If the predictor also confirms an increasing or high score, boost confidence
            if pred_res['predicted_threat_score'] > 0.5:
                consensus_confidence = min(0.99, consensus_confidence + 0.1)
        else:
            # If detector says benign but predictor forecasts a huge jump (early warning), flag it
            if pred_res['predicted_threat_score'] > 0.75 and pred_res['trend'] == "INCREASING":
                confirmed_threat = "UNKNOWN_ANOMALY"
                consensus_confidence = pred_res['predicted_threat_score']
                
        # If there are severe indicators from both, declare critical multi-attack
        if is_attack and pred_res['predicted_threat_score'] > 0.85:
            confirmed_threat = "CRITICAL_MULTI_ATTACK"
            
        # 5. Mitigation Agent action
        mit_res = self.mitigation_agent.determine_mitigation(confirmed_threat, consensus_confidence)
        
        t_latency = (time.perf_counter() - t_start) * 1000.0
        return {
            "packet_class": confirmed_threat,
            "confidence": consensus_confidence,
            "prediction_score": pred_res['predicted_threat_score'],
            "prediction_trend": pred_res['trend'],
            "system_state": mit_res['system_state'],
            "mitigation_action": mit_res['mitigation_action'],
            "latencies": {
                "detection_ms": det_res['latency_ms'],
                "prediction_ms": pred_res['latency_ms'],
                "mitigation_ms": mit_res['latency_ms'],
                "total_ms": t_latency
            }
        }
