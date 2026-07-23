"""Model comparison: rank the whole model zoo on synthetic seasonal data.

Run:  python examples/model_comparison.py
"""

import forecast_os as fos

# 5 daily series with trend + weekly seasonality
df = fos.generate_series(n_series=5, length=300, seasonality=7, trend=0.3, seed=42)

engine = fos.ForecastEngine()
leaderboard = engine.compare(
    df,
    h=14,
    n_windows=3,
    metrics=["mase", "rmse", "smape"],
    seasonality=7,
    models=[
        "naive",
        "seasonal_naive",
        "drift",
        "ses",
        "holt",
        fos.get_model("holt_winters", season_length=7),
        fos.get_model("theta", season_length=7),
        "auto_arima",
        "kalman",
        fos.get_model("ridge_lag", lags=14, season_length=7),
        "ensemble",
    ],
)
print("Leaderboard (mean over 5 series, 3 CV windows, h=14) — lower is better:\n")
print(leaderboard.round(3).to_string())

# AutoSelect picks a (possibly different) winner per series
auto = fos.get_model("auto_select", val_h=14)
auto.fit(df)
print("\nAutoSelect per-series winners:", auto.best_models_)
