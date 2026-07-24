"""Deal-grain pipeline: score deals, forecast pipeline with intervals, bridge it.

The v0.7.0 opportunity layer — the analyses a RevOps team actually runs:
calibrated per-deal win probability, a *probabilistic* pipeline forecast,
a created/won/lost waterfall between two snapshots, and a driver-based what-if.

Run:  python examples/deal_pipeline.py
"""

import numpy as np
import pandas as pd

from forecast_os.gtm import DealScorer, Scenario, compare_scenarios, weighted_pipeline
from forecast_os.snapshots import waterfall_summary

rng = np.random.default_rng(24)

# --- 1. Train a win-probability model on historical closed deals -----------
n = 800
stage_age = rng.uniform(0, 90, n)
activities = rng.poisson(8, n).astype(float)
logit = 0.04 * activities - 0.02 * stage_age + 0.2
won = rng.random(n) < 1 / (1 + np.exp(-logit))
history = pd.DataFrame(
    {"opp_id": range(n), "amount": rng.lognormal(10, 0.6, n),
     "stage_age": stage_age, "activities": activities, "won": won}
)
scorer = DealScorer().fit(history, target="won", features=["stage_age", "activities"])
print("win-probability drivers (standardized coef):")
print(scorer.coef_.round(3).to_string(), "\n")

# --- 2. Probabilistic pipeline forecast on the open pipeline ---------------
m = 150
opens = pd.DataFrame(
    {"opp_id": range(9000, 9000 + m), "amount": rng.lognormal(10, 0.6, m),
     "stage_age": rng.uniform(0, 90, m), "activities": rng.poisson(8, m).astype(float),
     "region": rng.choice(["EMEA", "AMER"], m)}
)
pipe = weighted_pipeline(opens, scorer=scorer, by="region", level=80)
print("probabilistic pipeline (E[won $] with an 80% interval, by region):")
print(pipe.to_string(index=False, float_format="${:,.0f}".format), "\n")

# --- 3. Pipeline waterfall between last week and this week ------------------
stages = ["prospect", "qualified", "proposal", "negotiation"]
last_week = pd.DataFrame(
    {"opp_id": [1, 2, 3, 4, 5], "amount": [40_000, 25_000, 60_000, 80_000, 30_000],
     "stage": ["qualified", "proposal", "proposal", "negotiation", "qualified"]}
)
this_week = pd.DataFrame(
    {"opp_id": [2, 3, 4, 5, 6], "amount": [30_000, 60_000, 80_000, 30_000, 45_000],
     "stage": ["negotiation", "proposal", "closed_won", "closed_lost", "qualified"]}
)
bridge = waterfall_summary(last_week, this_week, stages=stages,
                           won_stage="closed_won", lost_stage="closed_lost")
print("pipeline waterfall (how open pipeline moved week over week):")
print(bridge.to_string(index=False, float_format="${:,.0f}".format), "\n")

# --- 4. Driver-based what-if -----------------------------------------------
base = Scenario({"top_of_funnel": 1200, "win_rate": 0.28, "acv": 42_000})
board = compare_scenarios(
    base, base.bump(win_rate=-0.05), base.with_(top_of_funnel=1500),
    labels=["commit", "win-rate -5pt", "+300 leads"],
)
board["projection"] = board["projection"].map("${:,.0f}".format)
board["delta"] = board["delta"].map("${:+,.0f}".format)
board["pct_delta"] = board["pct_delta"].map("{:+.0%}".format)
print("scenario planning (best / base / worst bookings):")
print(board.to_string(index=False))
