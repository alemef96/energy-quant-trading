"""
forward_curve.py

Build a monthly power forward curve from quoted market products using a
no-arbitrage cascade, then split it into peak / off-peak.

Why this exists
---------------
A power desk does not trade the spot — it trades forwards (months, quarters,
calendar years). Those products are quoted at different granularities and must
be mutually consistent by no-arbitrage: the (time-weighted) average of the
finer contracts inside a period must equal the price of the coarser contract
covering that period.

    Cal-2026  =  average(Q1, Q2, Q3, Q4)      (weighted by days)
    Q1-2026   =  average(Jan, Feb, Mar)        (weighted by days)

This module takes quoted Months (front), Quarters, and the Calendar year and
"cascades" them down to a full 12-month curve:
  * directly quoted months are used as-is;
  * an unquoted month inside a quoted quarter is backed out by no-arbitrage;
  * a quarter with no quoted months is shaped into months by a seasonal profile,
    constrained so the months average back to the quarter quote;
  * the Cal quote is used as an independent no-arbitrage consistency check.

Finally each monthly baseload price is split into peak and off-peak legs, kept
consistent so the hour-weighted average returns the baseload.

The resulting curve is the term structure the spot model should be anchored to:
the flat long-term mean theta becomes a time-varying theta(t) = forward(t).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Market quotes (illustrative but realistic Cal-2026 baseload, EUR/MWh)
# =============================================================================
DELIVERY_YEAR = 2026

QUOTES = {
    "cal":      {"Cal-26": 82.0},
    "quarters": {"Q1": 104.0, "Q2": 67.0, "Q3": 73.0, "Q4": 89.0},
    "months":   {1: 112.0, 2: 103.0},          # front months quoted directly (Jan, Feb)
}

# Seasonal shape prior (winter-heavy, mild summer bump). Only used to distribute
# quarters that have no directly quoted months; the cascade still forces the
# quarter average to match the quote exactly, so the shape sets form, not level.
SEASONAL_SHAPE = np.array([1.22, 1.16, 1.05, 0.92, 0.86, 0.88,
                           0.95, 0.93, 0.94, 1.02, 1.10, 1.18])

# Peak premium over baseload by month (higher in winter evenings)
PEAK_PREMIUM = np.array([0.26, 0.24, 0.20, 0.15, 0.12, 0.12,
                         0.14, 0.14, 0.16, 0.19, 0.23, 0.25])

QUARTER_MONTHS = {"Q1": [1, 2, 3], "Q2": [4, 5, 6], "Q3": [7, 8, 9], "Q4": [10, 11, 12]}


# =============================================================================
# Calendar helpers
# =============================================================================
def month_days(year: int) -> np.ndarray:
    """Number of calendar days in each month of `year` (length 12)."""
    return np.array([(pd.Timestamp(year, m, 1) + pd.offsets.MonthEnd(0)).day
                     for m in range(1, 13)])

def month_business_days(year: int) -> np.ndarray:
    """Business days per month (proxy for peak-day count)."""
    out = []
    for m in range(1, 13):
        start = pd.Timestamp(year, m, 1)
        end = start + pd.offsets.MonthEnd(0)
        out.append(int(np.busday_count(start.date(), (end + pd.Timedelta(days=1)).date())))
    return np.array(out)


# =============================================================================
# 1. No-arbitrage cascade  ->  monthly baseload curve
# =============================================================================
def build_baseload_curve(year: int = DELIVERY_YEAR) -> pd.DataFrame:
    days = month_days(year)                       # weights for time-averaging
    shape = SEASONAL_SHAPE / SEASONAL_SHAPE.mean()
    fwd = np.full(12, np.nan)                      # monthly baseload forward

    quoted_months = QUOTES["months"]
    for m, price in quoted_months.items():
        fwd[m - 1] = price                        # (a) use quoted months directly

    # (b)+(c) cascade each quarter into its months, enforcing the quarter average
    for q, price in QUOTES["quarters"].items():
        months = QUARTER_MONTHS[q]
        idx = [m - 1 for m in months]
        w = days[idx]
        W = w.sum()
        fixed = [i for i in idx if not np.isnan(fwd[i])]
        free = [i for i in idx if np.isnan(fwd[i])]

        s_fixed = sum(days[i] * fwd[i] for i in fixed)
        if not free:
            # fully determined by quoted months: verify consistency, don't overwrite
            implied_q = s_fixed / W
            if abs(implied_q - price) > 1e-6:
                print(f"  [warn] {q}: quoted months imply {implied_q:.2f} vs quote {price:.2f}")
            continue
        # solve common scalar k so that shape-distributed free months + fixed months
        # average (day-weighted) to the quarter quote
        s_free_shape = sum(days[i] * shape[i] for i in free)
        k = (price * W - s_fixed) / s_free_shape
        for i in free:
            fwd[i] = k * shape[i]

    # (d) no-arbitrage consistency check against the quoted Cal
    cal_quote = list(QUOTES["cal"].values())[0]
    cal_implied = np.average(fwd, weights=days)

    curve = pd.DataFrame({
        "Month": [pd.Timestamp(year, m, 1).strftime("%b-%y") for m in range(1, 13)],
        "Days": days,
        "Baseload": fwd,
    })
    curve.attrs["cal_quote"] = cal_quote
    curve.attrs["cal_implied"] = cal_implied
    return curve


# =============================================================================
# 2. Peak / off-peak split (hour-weighted, no-arbitrage consistent)
# =============================================================================
def add_peak_offpeak(curve: pd.DataFrame, year: int = DELIVERY_YEAR) -> pd.DataFrame:
    days = curve["Days"].values
    total_hours = 24.0 * days
    peak_hours = 12.0 * month_business_days(year)         # 12h peak window on business days
    offpeak_hours = total_hours - peak_hours

    baseload = curve["Baseload"].values
    peak = baseload * (1.0 + PEAK_PREMIUM)
    # off-peak backed out so hour-weighted average returns baseload exactly
    offpeak = (baseload * total_hours - peak * peak_hours) / offpeak_hours

    curve = curve.copy()
    curve["Peak"] = peak
    curve["OffPeak"] = offpeak
    return curve


# =============================================================================
# 3. Daily expansion  ->  theta(t) for the spot model
# =============================================================================
def daily_baseload_curve(curve: pd.DataFrame, year: int = DELIVERY_YEAR) -> pd.Series:
    """Expand the monthly baseload curve to a daily step curve (flat within month)."""
    idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    daily = pd.Series(index=idx, dtype=float)
    for m in range(1, 13):
        daily[daily.index.month == m] = curve["Baseload"].iloc[m - 1]
    daily.name = "theta_t"
    return daily


def theta_for_month(curve: pd.DataFrame, month: int) -> float:
    """Forward level to anchor the spot model for a given delivery month."""
    return float(curve["Baseload"].iloc[month - 1])


# =============================================================================
# 4. Plot
# =============================================================================
def plot_curve(curve: pd.DataFrame, fname: str = "data/forward_curve.png"):
    import os
    os.makedirs("data", exist_ok=True)
    NAVY, TEAL, SEA, AMBER, RED = "#12293D", "#118AB2", "#2A9D8F", "#E9A23B", "#D1495B"
    x = np.arange(12)
    months = curve["Month"].values

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)

    # monthly baseload
    ax.plot(x, curve["Baseload"], color=NAVY, lw=2.4, marker="o", ms=6,
            zorder=5, label="Monthly baseload (cascaded)")
    # peak / off-peak
    ax.plot(x, curve["Peak"], color=RED, lw=1.6, ls="--", marker="^", ms=4, label="Peak")
    ax.plot(x, curve["OffPeak"], color=SEA, lw=1.6, ls="--", marker="v", ms=4, label="Off-peak")

    # quarter quotes as horizontal bands
    for q, price in QUOTES["quarters"].items():
        idx = [m - 1 for m in QUARTER_MONTHS[q]]
        ax.hlines(price, idx[0] - 0.35, idx[-1] + 0.35, color=AMBER, lw=3, zorder=4)
        ax.text(np.mean(idx), price + 2.0, f"{q}={price:.0f}", color="#B87A16",
                fontsize=9.5, fontweight="bold", ha="center")
    ax.plot([], [], color=AMBER, lw=3, label="Quarter quotes")

    # cal check
    cal_q = curve.attrs["cal_quote"]; cal_i = curve.attrs["cal_implied"]
    ax.axhline(cal_q, color=TEAL, lw=1.3, ls=":", zorder=3)
    ax.text(11.4, cal_q + 1.2, f"Cal quote={cal_q:.0f}\nimplied={cal_i:.1f}",
            color=TEAL, fontsize=9, ha="right", fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_ylabel("Forward price (\u20ac/MWh)"); ax.set_xlabel("Delivery month")
    ax.set_title("Power forward curve \u2014 no-arbitrage cascade with seasonality & peak/off-peak",
                 fontsize=12, fontweight="bold", color=NAVY, loc="left")
    ax.grid(True, alpha=0.25)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5, ncol=2, loc="upper center")
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fname


# =============================================================================
# Main
# =============================================================================
def main():
    print("================ POWER FORWARD CURVE BUILDER ================")
    print("Cascading quoted products (Cal -> Quarters -> Months), no-arbitrage...\n")

    curve = build_baseload_curve()
    curve = add_peak_offpeak(curve)

    with pd.option_context("display.float_format", lambda v: f"{v:7.2f}"):
        print(curve.to_string(index=False))

    cal_q = curve.attrs["cal_quote"]; cal_i = curve.attrs["cal_implied"]
    print(f"\nNo-arbitrage check  |  Cal quote: {cal_q:.2f}   curve-implied Cal: {cal_i:.2f}"
          f"   basis: {cal_i - cal_q:+.2f} EUR/MWh")

    fname = plot_curve(curve)
    print(f"Saved -> {fname}")

    # ---- connection to the spot model ----
    print("\n--- CONNECTING TO THE SPOT MODEL ---")
    for m, label in [(1, "Jan (winter)"), (7, "Jul (summer)")]:
        print(f"  theta(t) for {label:12s}: {theta_for_month(curve, m):6.2f} EUR/MWh")
    print("  -> the flat long-term mean in the MRJD model becomes theta(t) = forward(t),")
    print("     so an option or a plant is valued against the correct delivery-period level,")
    print("     not a single global average.")


if __name__ == "__main__":
    main()