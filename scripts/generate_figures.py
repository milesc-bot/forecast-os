"""Generate the README gallery figures from seeded simulated data.

Every figure is produced in a light and a dark variant (the README serves the
matching one via ``<picture>``), rendered on the palette's chart surface so the
PNGs look right on either GitHub theme.

Run:  .venv/bin/python scripts/generate_figures.py     (writes docs/assets/*.png)

Requires matplotlib (``pip install -e ".[dev]"``). Deterministic: all data
comes from forecast-os's seeded simulators.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import forecast_os as fos
from forecast_os.finance import (
    GARCH11,
    MarkovRegimeSwitching,
    MonteCarloSimulator,
    StrategyBacktester,
)

OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "assets"

# Palette (validated: see docs in the repo history / dataviz reference palette)
MODES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "baseline": "#c3c2b7",
        "bar_muted": "#c3c2b7",
        "blue": "#2a78d6",
        "orange": "#eb6834",
        # washes: alphas tuned per surface so bands stay visible in both modes
        "wash1": 0.10,
        "wash2": 0.20,
        "span": 0.12,
        "fill": 0.15,
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "baseline": "#383835",
        "bar_muted": "#4a4945",
        "blue": "#3987e5",
        "orange": "#d95926",
        "wash1": 0.18,
        "wash2": 0.34,
        "span": 0.18,
        "fill": 0.25,
    },
}


def style_axes(ax, c, x_dates=False, baseline=True):
    """Recessive chrome: hairline y-grid, single bottom spine, muted ticks."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(c["baseline"])
    ax.spines["bottom"].set_linewidth(0.8)
    if not baseline:
        ax.spines["bottom"].set_visible(False)
    ax.grid(axis="y", color=c["grid"], linewidth=0.7, zorder=0)
    ax.tick_params(colors=c["muted"], labelsize=9, length=0, pad=6)
    ax.margins(x=0.01)
    if x_dates:
        loc = mdates.AutoDateLocator(minticks=4, maxticks=7)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.set_facecolor("none")


def new_fig(c, title, subtitle, nrows=1, ratios=None, figsize=(8.0, 4.4)):
    fig = plt.figure(figsize=figsize, dpi=200, facecolor=c["surface"])
    top = 0.88 if nrows == 1 else 0.87
    gs = fig.add_gridspec(
        nrows, 1, top=top, bottom=0.10, left=0.075, right=0.975,
        height_ratios=ratios or [1] * nrows, hspace=0.32,
    )
    axes = [fig.add_subplot(gs[i]) for i in range(nrows)]
    fig.text(0.075, 0.965, title, ha="left", va="top",
             fontsize=13, fontweight="bold", color=c["ink"])
    fig.text(0.075, 0.925, subtitle, ha="left", va="top",
             fontsize=9.5, color=c["ink2"])
    return fig, axes


def save(fig, name, mode):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}-{mode}.png"
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"wrote {path.relative_to(OUT.parents[1])}")


# --------------------------------------------------------------------------- #
# Data (computed once, plotted per mode)                                      #
# --------------------------------------------------------------------------- #


def data_forecast():
    df = fos.generate_series(n_series=1, length=140, freq="D", level=210.0,
                             trend=0.45, seasonality=7, season_amp=14.0,
                             noise=3.0, seed=20)
    model = fos.get_model("auto_ets", season_length=7).fit(df)
    fc = model.predict(h=28, level=[80, 95])
    return {"history": df.tail(98), "fc": fc}


def data_leaderboard():
    df = fos.generate_series(n_series=5, length=300, seasonality=7,
                             trend=0.3, seed=42)
    models = [
        "naive", "drift", "ses", "kalman", "auto_arima",
        fos.get_model("seasonal_naive", season_length=7),
        fos.get_model("theta", season_length=7),
        fos.get_model("holt_winters", season_length=7),
        fos.get_model("ridge_lag", lags=14, season_length=7),
        # ensemble of the seasonal-aware members (an ensemble of non-seasonal
        # models on seasonal data would just average their errors)
        fos.get_model(
            "ensemble",
            models=(
                fos.get_model("theta", season_length=7),
                fos.get_model("holt_winters", season_length=7),
                fos.get_model("ridge_lag", lags=14, season_length=7),
            ),
        ),
    ]
    lb = fos.ForecastEngine().compare(df, h=14, n_windows=3,
                                      metrics=["mase"], seasonality=7,
                                      models=models)
    return {"lb": lb.sort_values("mase", ascending=False)}


def data_garch():
    panel = fos.generate_returns(length=1000, mu=0.0002,
                                 garch=(2e-6, 0.10, 0.86), seed=11)
    r = panel["y"].to_numpy()
    ds = panel["ds"].to_numpy()
    g = GARCH11().fit(r)
    h = 30
    fc = g.forecast_volatility(h)
    fc_ds = pd.date_range(ds[-1], periods=h + 1, freq="B")[1:]
    lr = float(np.sqrt(g.omega_ / (1 - g.alpha_ - g.beta_)))
    return {"ds": ds, "r": r, "vol": g.cond_vol_, "fc": fc, "fc_ds": fc_ds, "lr": lr}


def data_montecarlo():
    panel = fos.generate_returns(length=750, mu=0.00035, sigma=0.011, seed=6)
    r = panel["y"].to_numpy()
    mc = MonteCarloSimulator.from_returns(r, seed=7)
    h = 252
    paths = mc.simulate(s0=100.0, h=h, n_paths=2000)
    qs = {q: np.quantile(paths, q / 100, axis=0) for q in (5, 25, 50, 75, 95)}
    hist = np.cumprod(1 + r[-60:])
    hist = hist / hist[-1] * 100.0
    # representative sample paths (terminal values near set quantiles) so the
    # spaghetti illustrates the fan instead of one decorative outlier
    term = paths[:, -1]
    idx = [int(np.argmin(np.abs(term - np.quantile(term, q))))
           for q in (0.10, 0.35, 0.65, 0.90)]
    return {"qs": qs, "h": h, "hist": hist, "spaghetti": paths[idx]}


def data_regimes():
    rng = np.random.default_rng(1)
    segs = [(260, 0.0009, 0.007), (110, -0.0022, 0.019), (240, 0.0008, 0.008),
            (120, -0.0018, 0.021), (270, 0.0009, 0.007)]
    r = np.concatenate([mu + sig * rng.standard_normal(n) for n, mu, sig in segs])
    ds = pd.date_range("2022-01-03", periods=len(r), freq="B")
    m = MarkovRegimeSwitching(seed=3).fit(r)
    p_bear = m.smoothed_probs_[:, 0]
    price = 100 * np.cumprod(1 + r)
    return {"ds": ds, "price": price, "p_bear": p_bear}


def data_backtest():
    # AR(1)-momentum returns with a planted bear window: structure the ridge
    # forecaster can genuinely exploit, so long/flat visibly diverges from
    # buy & hold instead of hugging it.
    segs = [(420, 0.0009, 0.007), (160, -0.0022, 0.016), (320, 0.0009, 0.007)]
    phi = 0.25
    rng = np.random.default_rng(3)
    eps = np.concatenate([sig * rng.standard_normal(n) for n, _, sig in segs])
    mu = np.concatenate([np.full(n, m) for n, m, _ in segs])
    r = np.empty(len(eps))
    prev = 0.0
    for t in range(len(eps)):
        r[t] = mu[t] + phi * prev + eps[t]
        prev = r[t] - mu[t]
    ds = pd.date_range("2023-01-02", periods=len(r), freq="B")
    panel = pd.DataFrame({"unique_id": "asset-0", "ds": ds, "y": r})

    bt = StrategyBacktester(fos.get_model("ridge_lag", lags=5), cost_bps=1.0)
    res = bt.run(panel, test_size=320)
    f = res.frame
    strat = np.cumprod(1 + f["strategy_return"].to_numpy())
    bh = np.cumprod(1 + f["y"].to_numpy())
    dd = strat / np.maximum.accumulate(strat) - 1
    return {"ds": pd.to_datetime(f["ds"].to_numpy()), "strat": strat, "bh": bh, "dd": dd}


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #


def fig_forecast(d, c, mode):
    fig, (ax,) = new_fig(
        c, "Probabilistic forecasting",
        "auto_ets on simulated daily demand — 28-day horizon with 80 / 95% intervals",
    )
    h, fc = d["history"], d["fc"]
    cut = h["ds"].iloc[-1]
    ax.fill_between(fc["ds"], fc["lo-95"], fc["hi-95"], color=c["blue"],
                    alpha=c["wash1"], linewidth=0, zorder=2)
    ax.fill_between(fc["ds"], fc["lo-80"], fc["hi-80"], color=c["blue"],
                    alpha=c["wash2"], linewidth=0, zorder=2)
    ax.plot(h["ds"], h["y"], color=c["ink2"], linewidth=1.6, zorder=3)
    fc_x = pd.concat([h["ds"].iloc[-1:], fc["ds"]])
    fc_y = pd.concat([h["y"].iloc[-1:], fc["yhat"]])
    ax.plot(fc_x, fc_y, color=c["blue"], linewidth=2.0, zorder=4)
    ax.axvline(cut, color=c["baseline"], linewidth=0.8, zorder=1)
    style_axes(ax, c, x_dates=True)

    ymax = ax.get_ylim()[1]
    ax.annotate("forecast", (cut, ymax), xytext=(6, -2),
                textcoords="offset points", ha="left", va="top",
                fontsize=9, color=c["muted"])
    ax.annotate("history", (h["ds"].iloc[10], h["y"].iloc[:30].max()),
                xytext=(0, 14), textcoords="offset points",
                fontsize=9, color=c["ink2"])
    save(fig, "forecast", mode)


def fig_leaderboard(d, c, mode):
    lb = d["lb"]
    fig, (ax,) = new_fig(
        c, "Model leaderboard",
        "walk-forward CV on 5 simulated seasonal series — MASE over 3 windows × h=14"
        " (lower is better)",
    )
    names = lb.index.tolist()
    vals = lb["mase"].to_numpy()
    best = int(np.argmax(vals == vals.min()))
    colors = [c["blue"] if i == best else c["bar_muted"] for i in range(len(vals))]
    ax.barh(names, vals, height=0.58, color=colors, zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + vals.max() * 0.012, i, f"{v:.2f}", va="center", ha="left",
                fontsize=9, color=c["ink"] if i == best else c["ink2"])
    ax.axvline(1.0, color=c["muted"], linewidth=0.8, linestyle=(0, (3, 3)), zorder=4)
    ax.set_ylim(-0.6, len(names) + 0.35)
    ax.text(1.03, len(names) - 0.05, "MASE = 1", fontsize=8.5, color=c["muted"],
            ha="left", va="center", clip_on=True)
    style_axes(ax, c, baseline=False)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=c["grid"], linewidth=0.7, zorder=0)
    ax.set_xlim(0, vals.max() * 1.12)
    ax.tick_params(axis="y", labelsize=9.5, colors=c["ink2"])
    save(fig, "leaderboard", mode)


def fig_garch(d, c, mode):
    fig, (ax1, ax2) = new_fig(
        c, "Volatility modeling",
        "GARCH(1,1) conditional volatility on simulated returns, with a 30-day forecast",
        nrows=2, ratios=[1, 1.35], figsize=(8.0, 5.2),
    )
    z = slice(-420, None)  # zoom so the 30-day forecast is legible
    ax1.plot(d["ds"][z], d["r"][z] * 100, color=c["muted"], linewidth=0.6)
    ax1.axhline(0, color=c["baseline"], linewidth=0.8)
    style_axes(ax1, c, x_dates=True)
    ax1.set_ylabel("daily return %", fontsize=9, color=c["muted"])

    vol = d["vol"] * 100
    lr = d["lr"] * 100
    ax2.plot(d["ds"][z], vol[z], color=c["blue"], linewidth=1.8)
    ax2.plot(d["fc_ds"], d["fc"] * 100, color=c["blue"], linewidth=1.8,
             linestyle=(0, (4, 3)))
    ax2.axhline(lr, color=c["muted"], linewidth=0.8, linestyle=(0, (3, 3)))
    style_axes(ax2, c, x_dates=True)
    ax2.set_ylabel("conditional vol %", fontsize=9, color=c["muted"])
    # label the reference line where the series sits well below it, so the
    # text never crosses the data stroke
    volz = vol[z]
    clear = [i for i in range(15, len(volz) - 40)
             if np.all(volz[i - 12:i + 12] < lr - 0.10)]
    lx = d["ds"][z][clear[len(clear) // 2]] if clear else d["ds"][z][10]
    ax2.annotate("long-run vol", (lx, lr), xytext=(0, 5), ha="center",
                 textcoords="offset points", fontsize=8.5, color=c["muted"])
    mid = len(d["fc_ds"]) // 2
    ax2.annotate("30-day forecast", (d["fc_ds"][mid], d["fc"][mid] * 100),
                 xytext=(-4, -16), textcoords="offset points", ha="center",
                 va="top", fontsize=9, color=c["ink"])
    ax1.set_xlim(*ax2.get_xlim())
    save(fig, "garch", mode)


def fig_montecarlo(d, c, mode):
    fig, (ax,) = new_fig(
        c, "Scenario simulation",
        "Monte Carlo GBM calibrated from simulated returns — 2,000 paths, 1-year fan",
    )
    x = np.arange(1, d["h"] + 1)
    xh = np.arange(-59, 1)
    qs = d["qs"]
    ax.fill_between(x, qs[5], qs[95], color=c["blue"], alpha=c["wash1"], linewidth=0)
    ax.fill_between(x, qs[25], qs[75], color=c["blue"], alpha=c["wash2"], linewidth=0)
    for p in d["spaghetti"]:
        ax.plot(x, p, color=c["muted"], linewidth=0.5, alpha=0.55, zorder=2)
    ax.plot(x, qs[50], color=c["blue"], linewidth=2.0, zorder=4)
    ax.plot(xh, d["hist"], color=c["ink2"], linewidth=1.6, zorder=3)
    ax.annotate("sample paths", (150, d["spaghetti"][3][149]), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=8.5,
                color=c["muted"],
                bbox=dict(facecolor=c["surface"], edgecolor="none", pad=1.0,
                          alpha=0.8))
    ax.axvline(0, color=c["baseline"], linewidth=0.8, zorder=1)
    style_axes(ax, c)
    ax.set_xlabel("trading days from today", fontsize=9, color=c["muted"])

    for q, label in ((95, "P95"), (75, "P75"), (50, "median"), (25, "P25"), (5, "P05")):
        ax.annotate(f"{label}  {qs[q][-1]:.0f}", (d["h"], qs[q][-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5,
                    color=c["ink"] if q == 50 else c["muted"])
    ax.annotate("history", (-52, d["hist"][:20].max()), xytext=(0, 12),
                textcoords="offset points", fontsize=9, color=c["ink2"])
    ax.set_xlim(-62, d["h"] + 26)
    save(fig, "montecarlo", mode)


def fig_regimes(d, c, mode):
    fig, (ax1, ax2) = new_fig(
        c, "Regime detection",
        "2-state Markov switching on simulated returns — smoothed bear-market probability",
        nrows=2, ratios=[2.0, 1.0], figsize=(8.0, 5.2),
    )
    bear = d["p_bear"] > 0.5
    spans, start = [], None
    for i, b in enumerate(bear):
        if b and start is None:
            start = i
        elif not b and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(bear)))

    ax1.plot(d["ds"], d["price"], color=c["blue"], linewidth=1.8, zorder=3)
    for s, e in spans:
        ax1.axvspan(d["ds"][s], d["ds"][e - 1], color=c["orange"],
                    alpha=c["span"], linewidth=0, zorder=1)
    style_axes(ax1, c, x_dates=True)
    ax1.set_ylabel("portfolio value", fontsize=9, color=c["muted"])
    if spans:
        s, e = spans[0]
        mid = d["ds"][(s + e) // 2]
        ax1.text(mid, 0.93, "bear regime", transform=ax1.get_xaxis_transform(),
                 ha="center", fontsize=9, color=c["ink2"],
                 bbox=dict(facecolor=c["surface"], edgecolor="none", pad=1.5,
                           alpha=0.85))

    ax2.fill_between(d["ds"], 0, d["p_bear"], color=c["orange"], alpha=c["fill"],
                     linewidth=0)
    ax2.plot(d["ds"], d["p_bear"], color=c["orange"], linewidth=1.6)
    ax2.axhline(0.5, color=c["baseline"], linewidth=0.8)
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_yticks([0, 0.5, 1])
    style_axes(ax2, c, x_dates=True)
    ax2.set_ylabel("P(bear)", fontsize=9, color=c["muted"])
    save(fig, "regimes", mode)


def fig_backtest(d, c, mode):
    fig, (ax1, ax2) = new_fig(
        c, "Strategy backtesting",
        "walk-forward ridge forecaster, long/flat with 1 bp costs, vs buy & hold —"
        " simulated momentum returns",
        nrows=2, ratios=[2.2, 1.0], figsize=(8.0, 5.2),
    )
    label_box = dict(facecolor=c["surface"], edgecolor="none", pad=1.5, alpha=0.85)
    ax1.plot(d["ds"], d["strat"], color=c["blue"], linewidth=1.8, label="strategy")
    ax1.plot(d["ds"], d["bh"], color=c["orange"], linewidth=1.8, label="buy & hold")
    strat_on_top = d["strat"][-1] >= d["bh"][-1]
    ends = [(d["strat"], c["blue"], -13 if strat_on_top else -3),
            (d["bh"], c["orange"], -3 if strat_on_top else -13)]
    for series, col, dy in ends:
        ax1.plot(d["ds"][-1], series[-1], "o", markersize=5, color=col,
                 markeredgecolor=c["surface"], markeredgewidth=1.4, zorder=5)
        ax1.annotate(f"{(series[-1] - 1) * 100:+.1f}%", (d["ds"][-1], series[-1]),
                     xytext=(8, dy), textcoords="offset points", fontsize=9,
                     color=c["ink"], bbox=label_box)
    leg = ax1.legend(loc="upper left", frameon=False, fontsize=9,
                     handlelength=1.4, borderaxespad=0.1)
    for t in leg.get_texts():
        t.set_color(c["ink2"])
    style_axes(ax1, c, x_dates=True)
    ax1.set_ylabel("growth of $1", fontsize=9, color=c["muted"])
    ax1.margins(x=0.07, y=0.12)
    ax1.tick_params(labelbottom=False)

    ax2.fill_between(d["ds"], 0, d["dd"] * 100, color=c["blue"], alpha=c["fill"],
                     linewidth=0)
    ax2.plot(d["ds"], d["dd"] * 100, color=c["blue"], linewidth=1.2)
    imin = int(np.argmin(d["dd"]))
    ax2.set_ylim(d["dd"].min() * 100 * 1.35, d["dd"].max() * 100 + 0.4)
    ax2.annotate(f"max drawdown {d['dd'][imin] * 100:.1f}%",
                 (d["ds"][imin], d["dd"][imin] * 100), xytext=(10, 6),
                 textcoords="offset points", fontsize=8.5, color=c["ink2"],
                 bbox=label_box)
    style_axes(ax2, c, x_dates=True)
    ax2.set_ylabel("strategy drawdown %", fontsize=9, color=c["muted"])
    ax2.set_xlim(*ax1.get_xlim())
    save(fig, "backtest", mode)


FIGS = [
    (data_forecast, fig_forecast),
    (data_leaderboard, fig_leaderboard),
    (data_garch, fig_garch),
    (data_montecarlo, fig_montecarlo),
    (data_regimes, fig_regimes),
    (data_backtest, fig_backtest),
]


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
    })
    for data_fn, fig_fn in FIGS:
        d = data_fn()
        for mode, c in MODES.items():
            fig_fn(d, c, mode)


if __name__ == "__main__":
    main()
