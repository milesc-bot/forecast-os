"""Quickstart: forecast AirPassengers with intervals, then cross-validate.

Run:  python examples/quickstart.py
"""

import forecast_os as fos

df = fos.load_air_passengers()
print(f"Loaded {df['unique_id'].nunique()} series, {len(df)} rows\n")

# Fit one model, forecast a year ahead with 80/95% intervals
model = fos.get_model("auto_ets", season_length=12)
model.fit(df)
forecast = model.predict(h=12, level=[80, 95])
print("AutoETS 12-month forecast:")
print(forecast.to_string(index=False, float_format="{:.1f}".format))

# Walk-forward cross-validation of several models
cv = fos.cross_validation(
    df,
    models=["seasonal_naive", "theta", "auto_ets"],
    h=12,
    n_windows=3,
)
scores = fos.evaluate(cv, metrics=["mase", "smape"], train_df=df, seasonality=12)
print("\nCross-validated accuracy (3 windows, h=12):")
print(scores.round(3).to_string(index=False))
