"""GTM quickstart: from a CRM-shaped export to a coherent, governed forecast.

The tour: raw opportunity records -> hierarchy-ready bookings panel ->
reconciled rep/team forecasts -> quota attainment probability ->
forecast-accuracy governance with signed bias.

Run:  python examples/gtm_quickstart.py
"""

import numpy as np
import pandas as pd

import forecast_os as fos
from forecast_os.gtm import attainment_probability, to_panel

rng = np.random.default_rng(7)

# --- 1. A CRM-shaped export: one row per closed-won opportunity ------------
reps = {"west/alice": 60_000, "west/bob": 45_000, "east/carol": 80_000}
rows = []
for rep, acv in reps.items():
    for month in pd.date_range("2023-01-01", periods=36, freq="MS"):
        # deals land mid-quarter-light, quarter-end-heavy
        n_deals = rng.poisson(2 + 3 * (month.month % 3 == 0))
        for _ in range(n_deals):
            rows.append(
                {
                    "rep": rep,
                    "close_date": month + pd.Timedelta(days=int(rng.integers(0, 28))),
                    "amount": acv * float(rng.lognormal(0, 0.35)),
                }
            )
records = pd.DataFrame(rows)
print(f"CRM export: {len(records)} opportunities, {records['rep'].nunique()} reps")

# --- 2. Event records -> monthly bookings panel (hierarchy-ready ids) ------
panel = to_panel(records, id_cols=["rep"], date_col="close_date",
                 value_col="amount", freq="MS", agg="sum")
print(f"panel: {panel['unique_id'].nunique()} series x {len(panel)} rows")

# --- 3. Coherent forecasts: reps, teams, and total that actually add up ----
model = fos.get_model("reconciled", model="auto_ets", method="mint_ols")
model.fit(panel)
fc = model.predict(h=6, level=[80])
total_next_q = fc[fc["unique_id"] == "total"].head(3)["yhat"].sum()
print(f"\nreconciled forecast covers: {sorted(fc['unique_id'].unique())}")
print(f"next-quarter total bookings forecast: ${total_next_q:,.0f}")

# --- 4. Quota attainment probability from the forecast intervals -----------
quotas = {"west": 2_000_000, "east": 1_400_000, "total": 3_400_000}
teams = fc[fc["unique_id"].isin(quotas)]
att = attainment_probability(teams, quota=quotas, level=80)
print("\nP(6-month bookings >= quota):")
print(att.round(3).to_string(index=False))

# --- 5. Governance: is anyone's forecast systematically sandbagged? --------
cv = fos.cross_validation(panel, models=["seasonal_naive", "auto_ets"], h=3, n_windows=4,
                          level=[80])
scores = fos.evaluate(cv, metrics=["mase", "pct_bias", "coverage"],
                      train_df=panel, seasonality=3)
print("\nforecast governance (MASE, signed bias, 80% coverage):")
print(scores.round(3).to_string(index=False))
