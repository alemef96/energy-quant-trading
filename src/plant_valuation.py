"""
lsm_gas_plant.py

Weather-driven structural power price + switching real-option valuation of a
gas-fired (CCGT) power plant.

Two ideas combined:

1. STRUCTURAL PRICE FORMATION (weather -> merit order)
   Power spot prices are not modelled as an exogenous jump-diffusion. Instead they
   emerge from the merit order: weather drives electricity demand (heating / cooling)
   and renewable supply (wind), which together set RESIDUAL LOAD = demand - renewables.
   The clearing price is the marginal cost of the last unit needed to meet residual
   load. As residual load rises (cold snap + wind lull) we move right in the merit
   order onto gas, then onto scarcity-priced peakers -> price spikes emerge endogenously.

2. SWITCHING REAL OPTION (start-up cost -> dynamic programming)
   A CCGT is not a single-exercise American option; it is dispatched (ON/OFF) every
   day. Turning the plant on incurs a start-up cost, so the optimal policy is NOT
   myopic "run whenever the spark spread is positive" - it must weigh the cost of
   cycling. This makes the problem a two-state switching control, solved by
   Least-Squares Monte Carlo (Longstaff-Schwartz) with backward induction.

The value the plant captures is, in effect, the value of the weather-driven price
spikes: this is the financial expression of a climate hazard.
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# High-degree polynomial regressions on standardised data are numerically fine here
# but can trip numpy's conditioning heuristic on days with little spread dispersion.
try:
    warnings.simplefilter("ignore", np.exceptions.RankWarning)   # numpy >= 2.0
except AttributeError:                                            # numpy < 2.0
    warnings.simplefilter("ignore", np.RankWarning)


# =============================================================================
# Parameters
# =============================================================================
@dataclass
class MarketConfig:
    # --- horizon / simulation ---
    days: int = 365
    paths: int = 20_000
    r_annual: float = 0.05           # discount rate

    # --- temperature anomaly (OU around 0 degrees C from comfort) ---
    temp_kappa: float = 0.15         # mean-reversion speed
    temp_sigma: float = 1.6          # daily vol (deg C)

    # --- wind availability index in [0, 1] (OU on the logit) ---
    wind_mean: float = 0.45          # long-run capacity factor
    wind_kappa: float = 0.20
    wind_sigma: float = 0.55

    # --- demand (GW) ---
    demand_base: float = 50.0
    heat_slope: float = 1.6          # +GW per deg C below comfort (cold -> heating)
    cool_slope: float = 1.0          # +GW per deg C above comfort (hot -> cooling)
    demand_noise: float = 1.2        # GW

    # --- renewable fleet (GW) ---
    renew_capacity: float = 30.0

    # --- merit-order stack ---
    baseload_cap: float = 36.0       # Q1: nuclear/coal/must-run capacity (GW)
    gas_cap: float = 18.0            # gas tranche width (GW); Q2 = Q1 + gas_cap
    baseload_cost: float = 35.0      # EUR/MWh, cheap infra-marginal tech
    scarcity_slope: float = 45.0     # EUR/MWh per GW above Q2 (peaker/scarcity)
    price_cap: float = 500.0         # EUR/MWh scarcity cap

    # --- fuel / carbon (drives production cost) ---
    gas_base: float = 17.0           # EUR/MWh_thermal
    gas_cold_sens: float = 1.0       # gas price rises in cold snaps (EUR/MWh_th per deg C)
    carbon_price: float = 50.0       # EUR/ton CO2
    emission_factor: float = 0.30    # ton CO2 per MWh_thermal

    # our (efficient) plant vs the marginal fleet unit
    plant_efficiency: float = 0.55   # our CCGT is modern/efficient
    fleet_efficiency: float = 0.45   # marginal gas unit setting the price is older

    # --- plant operation ---
    start_up_cost: float = 12.0      # EUR/MWh penalty to switch OFF -> ON


# =============================================================================
# 1. Weather + structural price simulation
# =============================================================================
def simulate_weather_and_price(cfg: MarketConfig, seed: int = 42) -> dict:
    """Simulate correlated weather drivers and derive the structural power price.

    Returns arrays of shape (days, paths).
    """
    rng = np.random.default_rng(seed)
    T, N = cfg.days, cfg.paths

    # --- temperature anomaly: Ornstein-Uhlenbeck around 0 ---
    temp = np.zeros((T, N))
    for t in range(1, T):
        dW = rng.standard_normal(N)
        temp[t] = temp[t - 1] - cfg.temp_kappa * temp[t - 1] + cfg.temp_sigma * dW

    # --- wind availability: OU on the logit, squashed into (0, 1) ---
    logit = np.full((T, N), np.log(cfg.wind_mean / (1 - cfg.wind_mean)))
    for t in range(1, T):
        dW = rng.standard_normal(N)
        logit[t] = logit[t - 1] - cfg.wind_kappa * logit[t - 1] + cfg.wind_sigma * dW
    logit += np.log(cfg.wind_mean / (1 - cfg.wind_mean))  # recentre
    wind = 1.0 / (1.0 + np.exp(-logit))

    # --- demand: base + heating (cold) + cooling (hot) + noise ---
    heating = cfg.heat_slope * np.maximum(-temp, 0.0)
    cooling = cfg.cool_slope * np.maximum(temp, 0.0)
    demand = cfg.demand_base + heating + cooling + cfg.demand_noise * rng.standard_normal((T, N))

    # --- renewable supply and residual load ---
    renewables = cfg.renew_capacity * wind
    residual = demand - renewables

    # --- production cost: our plant (VPC_own) and the marginal fleet unit (VPC_mkt) ---
    gas_price = cfg.gas_base + cfg.gas_cold_sens * np.maximum(-temp, 0.0)  # cold -> costlier gas
    carbon_term = cfg.carbon_price * cfg.emission_factor
    vpc_own = gas_price / cfg.plant_efficiency + carbon_term / cfg.plant_efficiency
    vpc_mkt = gas_price / cfg.fleet_efficiency + carbon_term / cfg.fleet_efficiency

    # --- merit order: price = marginal cost of the tranche serving residual load ---
    Q1 = cfg.baseload_cap
    Q2 = cfg.baseload_cap + cfg.gas_cap
    price = np.empty((T, N))
    below = residual <= Q1
    gas_zone = (residual > Q1) & (residual <= Q2)
    scarcity = residual > Q2
    price[below] = cfg.baseload_cost
    price[gas_zone] = vpc_mkt[gas_zone]
    price[scarcity] = vpc_mkt[scarcity] + cfg.scarcity_slope * (residual[scarcity] - Q2)
    price = np.minimum(price, cfg.price_cap)

    # --- clean spark spread for OUR plant ---
    spread = price - vpc_own

    return {
        "temp": temp, "wind": wind, "demand": demand, "renewables": renewables,
        "residual": residual, "price": price, "vpc_own": vpc_own, "spread": spread,
    }


# =============================================================================
# 2. Switching real-option valuation (Longstaff-Schwartz)
# =============================================================================
def value_plant_switching_lsm(spread: np.ndarray, cfg: MarketConfig,
                              basis_degree: int = 4):
    """Value the plant as a two-state (ON/OFF) switching option via LSM.

    State machine, decided at the start of each day t:
        OFF -> run:  pay start_up_cost, earn spread_t, end day ON
        OFF -> stay: earn 0, end day OFF
        ON  -> stay: earn spread_t, end day ON  (no start-up cost)
        ON  -> stop: earn 0, end day OFF

    Backward induction with regression-estimated continuation values.
    Returns (plant_value, dispatch_matrix, naive_strip_value).
    """
    T, N = spread.shape
    dt = 1.0 / 365.0
    df = np.exp(-cfg.r_annual * dt)

    # Value-to-go of arriving in each state at t+1. Terminal: nothing left to run.
    V_off = np.zeros(N)
    V_on = np.zeros(N)

    # store the per-day continuation-value regressions so we can forward-simulate
    # the optimal policy afterwards (true dispatch, not a proxy)
    policy = [None] * T

    def fit(future_value, s):
        """Regress df * future_value on a standardised polynomial of the spread."""
        target = df * future_value
        mu, sd = s.mean(), s.std() + 1e-9
        # if the target is (near-)constant the polynomial fit is rank-deficient;
        # fall back to its mean (a degree-0 continuation estimate)
        if target.std() < 1e-8:
            return (mu, sd, np.array([target.mean()]))
        z = (s - mu) / sd
        coeffs = np.polyfit(z, target, deg=basis_degree)
        return (mu, sd, coeffs)

    def predict(model, s):
        mu, sd, coeffs = model
        return np.polyval(coeffs, (s - mu) / sd)

    # ---- Backward induction ----
    for t in range(T - 1, -1, -1):
        s = spread[t]
        if t == T - 1:
            model_off = model_on = (0.0, 1.0, np.zeros(basis_degree + 1))
            cont_off = cont_on = np.zeros(N)
        else:
            model_off = fit(V_off, s)
            model_on = fit(V_on, s)
            cont_off = predict(model_off, s)
            cont_on = predict(model_on, s)
        policy[t] = (model_off, model_on)

        # value-to-go from the START of day t, given entry state
        run = -cfg.start_up_cost + s + cont_on   # OFF -> switch on
        idle = cont_off                          # OFF -> stay off
        keep = s + cont_on                       # ON  -> stay on (no start-up cost)
        stop = cont_off                          # ON  -> switch off
        V_off = np.maximum(run, idle)
        V_on = np.maximum(keep, stop)

    # Plant is commissioned OFF at t=0
    plant_value = float(np.mean(V_off))

    # ---- Forward pass: simulate the optimal dispatch path per scenario ----
    dispatch = np.zeros((T, N))
    is_on = np.zeros(N, dtype=bool)              # start OFF
    for t in range(T):
        s = spread[t]
        model_off, model_on = policy[t]
        cont_off = predict(model_off, s)
        cont_on = predict(model_on, s)
        # decision depends on the current state
        want_on_from_off = (-cfg.start_up_cost + s + cont_on) > cont_off
        want_on_from_on = (s + cont_on) > cont_off
        is_on = np.where(is_on, want_on_from_on, want_on_from_off)
        dispatch[t] = is_on.astype(float)

    # Naive benchmark: myopic strip of daily options, ignoring start-up cost.
    # This is an UPPER BOUND (frictionless): run whenever spread > 0.
    discount = df ** np.arange(T)[:, None]
    naive_strip = float(np.mean(np.sum(np.maximum(spread, 0.0) * discount, axis=0)))

    return plant_value, dispatch, naive_strip


# =============================================================================
# 3. Diagnostics plot (one representative path)
# =============================================================================
def plot_representative_path(sim: dict, dispatch: np.ndarray, cfg: MarketConfig,
                             path_idx: int = 0, fname: str = "data/weather_dispatch.png"):
    T = cfg.days
    days = np.arange(T)
    temp = sim["temp"][:, path_idx]
    wind = sim["wind"][:, path_idx]
    price = sim["price"][:, path_idx]
    disp = dispatch[:, path_idx]

    NAVY, TEAL, AMBER, RED, GREEN = "#12293D", "#118AB2", "#E9A23B", "#D1495B", "#2A9D8F"
    fig, ax = plt.subplots(3, 1, figsize=(10, 7.2), dpi=150, sharex=True)

    # weather drivers
    ax[0].plot(days, temp, color=TEAL, lw=1.4, label="Temperature anomaly (\u00b0C)")
    ax[0].axhline(0, color="#B9C6D0", lw=0.8)
    ax[0].fill_between(days, temp, 0, where=temp < 0, color=TEAL, alpha=0.15)
    ax2 = ax[0].twinx()
    ax2.plot(days, wind, color=GREEN, lw=1.4, label="Wind availability")
    ax2.set_ylim(0, 1)
    ax[0].set_ylabel("Temp anomaly (\u00b0C)", color=NAVY, fontsize=10)
    ax2.set_ylabel("Wind avail.", color=GREEN, fontsize=10)
    ax[0].set_title("Weather drivers", fontsize=12, fontweight="bold", color=NAVY, loc="left")

    # power price
    ax[1].plot(days, price, color=NAVY, lw=1.3)
    ax[1].fill_between(days, price, price.min(), color=NAVY, alpha=0.06)
    ax[1].set_ylabel("Power price\n(\u20ac/MWh)", fontsize=10, color=NAVY)
    ax[1].set_title("Structural power price (merit-order clearing)", fontsize=12,
                    fontweight="bold", color=NAVY, loc="left")

    # dispatch
    ax[2].fill_between(days, 0, 1, where=disp > 0.5, color=AMBER, alpha=0.55, step="mid")
    ax[2].set_ylim(0, 1); ax[2].set_yticks([0, 1]); ax[2].set_yticklabels(["OFF", "ON"])
    ax[2].set_ylabel("Dispatch", fontsize=10, color=NAVY)
    ax[2].set_xlabel("Day", fontsize=10)
    ax[2].set_title("Optimal plant dispatch (switching policy)", fontsize=12,
                    fontweight="bold", color=NAVY, loc="left")

    for a in ax:
        for spine in ["top", "right"]:
            a.spines[spine].set_visible(False)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fname


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 74)
    print(" WEATHER-DRIVEN MERIT ORDER  ->  SWITCHING REAL-OPTION PLANT VALUATION")
    print("=" * 74)

    cfg = MarketConfig()

    print("\n[1] Simulating weather drivers and structural power prices...")
    sim = simulate_weather_and_price(cfg)
    price, spread, residual = sim["price"], sim["spread"], sim["residual"]

    print(f"    Paths simulated:            {cfg.paths:,} x {cfg.days} days")
    print(f"    Mean power price:           {price.mean():6.2f} EUR/MWh")
    print(f"    95th pct power price:       {np.percentile(price, 95):6.2f} EUR/MWh")
    print(f"    99th pct power price:       {np.percentile(price, 99):6.2f} EUR/MWh")
    print(f"    Scarcity days (RL > Q2):    {100 * np.mean(residual > (cfg.baseload_cap + cfg.gas_cap)):5.2f} %")
    print(f"    Mean clean spark spread:    {spread.mean():6.2f} EUR/MWh")

    print("\n[2] Valuing the plant as a switching option (Longstaff-Schwartz)...")
    plant_value, dispatch, naive_strip = value_plant_switching_lsm(spread, cfg)

    print(f"    Naive strip (no start-up):  {naive_strip:8.2f} EUR/MWh   <- frictionless upper bound")
    print(f"    Switching LSM value:        {plant_value:8.2f} EUR/MWh   <- with start-up cost")
    print(f"    Value lost to cycling:      {naive_strip - plant_value:8.2f} EUR/MWh "
          f"({100 * (naive_strip - plant_value) / naive_strip:4.1f}%)")
    print(f"    Mean capacity factor:       {100 * dispatch.mean():5.1f} %")

    os.makedirs("data", exist_ok=True)
    print("\n[3] Saving diagnostics plot...")
    # choose a path with a clear, representative spike (peak price near the 97th pct of peaks)
    peak_per_path = sim["price"].max(axis=0)
    target = np.percentile(peak_per_path, 97)
    path_idx = int(np.argmin(np.abs(peak_per_path - target)))
    fname = plot_representative_path(sim, dispatch, cfg, path_idx=path_idx)
    print(f"    -> {fname}  (path #{path_idx}, peak {peak_per_path[path_idx]:.0f} EUR/MWh)")

    print("\n" + "=" * 74)
    print(" COMMENTARY")
    print("=" * 74)
    print(
        "* Spikes are ENDOGENOUS. Prices are not driven by an exogenous jump process;\n"
        "  they emerge when weather (cold snaps, wind lulls) pushes residual load into\n"
        "  the scarcity tranche of the merit order. This is a climate hazard expressed\n"
        "  as a price.\n\n"
        "* LSM is JUSTIFIED here. Because switching the plant ON costs money, the optimal\n"
        "  policy is not myopic 'run when spread > 0'. The gap between the naive strip and\n"
        "  the switching value is exactly the economic cost of the start-up friction.\n\n"
        "* The plant's value is concentrated in weather-driven tail events - which is why\n"
        "  a hazard model on top of this (conditioning jump/scarcity frequency on climate\n"
        "  projections) directly changes the asset's fair value."
    )


if __name__ == "__main__":
    main()