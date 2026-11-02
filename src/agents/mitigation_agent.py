import time

class MitigationAgent:
    def __init__(self):
        self.name = "Mitigation Agent"
        self.current_state = "NORMAL_FLIGHT"
        
    def determine_mitigation(self, confirmed_threat, confidence):
        t_start = time.perf_counter()
        
        action = "MONITOR"
        
        if confirmed_threat == "DoS attack" and confidence > 0.70:
            self.current_state = "TRAFFIC_THROTTLE"
            action = "ENABLE_PACKET_RATE_LIMITER"
        elif confirmed_threat == "Replay" and confidence > 0.70:
            self.current_state = "CRYPTOGRAPHIC_LOCKOUT"
            action = "LOCK_UNAUTHENTICATED_COMMANDS_ENGAGE_CHALLENGE_RESPONSE"
        elif confirmed_threat == "CRITICAL_MULTI_ATTACK":
            self.current_state = "EMERGENCY_RTL"
            action = "TRIGGER_FAILSAFE_RETURN_TO_LAUNCH"
        else:
            self.current_state = "NORMAL_FLIGHT"
            action = "ALLOW_ALL_TRAFFIC"
            
        t_latency = (time.perf_counter() - t_start) * 1000.0
        return {
            "system_state": self.current_state,
            "mitigation_action": action,
            "latency_ms": t_latency
        }
