
def episode_reward_from_metrics(m, w_risk=1.0, w_count=0.0005):
    risk_term = w_risk * m["risk_mean"]
    count_term = w_count * m["n_events"]
    return -(risk_term + count_term)
