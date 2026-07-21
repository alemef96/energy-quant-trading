import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_complete_energy_pipeline():
    print("================ MASTER QUANT PIPELINE & VISUALIZATION ================")
    
    # 1. GENERATION & CALIBRATION (MRJD)
    np.random.seed(42)
    days = 365
    dt = 1.0
    
    kappa_p, theta_p, sigma_p, lam_p = 0.3, 60.0, 0.20, 0.03
    jump_mu, jump_sigma = 80.0, 20.0
    
    prices = np.zeros(days)
    prices[0] = theta_p
    
    for t in range(1, days):
        dW = np.random.normal(0, np.sqrt(dt))
        dq = np.random.poisson(lam_p * dt)
        jump = np.random.normal(jump_mu, jump_sigma) if dq > 0 else 0.0
        dS = kappa_p * (theta_p - prices[t-1]) * dt + sigma_p * prices[t-1] * dW + jump
        prices[t] = max(0.0, prices[t-1] + dS)
        
    dates = pd.date_range(start="2026-01-01", periods=days, freq="D")
    df_market = pd.DataFrame({"Price": prices}, index=dates)
    
    # Spike Cleansing (Vectorized)
    df_market['Return'] = df_market['Price'].pct_change()
    r_mean = df_market['Return'].rolling(20).mean()
    r_std = df_market['Return'].rolling(20).std()
    df_market['Is_Jump'] = np.abs(df_market['Return'] - r_mean) > (3.0 * r_std)
    df_market['Is_Jump'] = df_market['Is_Jump'].fillna(False)
    
    df_market['Clean_Price'] = np.where(df_market['Is_Jump'], np.nan, df_market['Price'])
    df_market['Clean_Price'] = df_market['Clean_Price'].ffill().fillna(df_market['Price'])

    # OLS Calibration
    S = df_market['Clean_Price'].values
    dS_arr = S[1:] - S[:-1]
    S_lag = S[:-1]
    slope, intercept = np.polyfit(S_lag, dS_arr, deg=1)
    kappa_cal = -slope
    theta_cal = intercept / kappa_cal
    
    print(f"[Calibration] Kappa: {kappa_cal:.4f} | Theta: {theta_cal:.2f} | Lambda: {lam_p}")

    # 2. MONTE CARLO SIMULATION FOR RISK & TAIL ANALYSIS
    sim_paths = 20000
    delivery_days = 30
    S_matrix = np.zeros((delivery_days + 1, sim_paths))
    S_matrix[0, :] = S[-1]
    
    for t in range(1, delivery_days + 1):
        dW_m = np.random.normal(0, 1, sim_paths)
        dq_m = np.random.poisson(lam_p * dt, sim_paths)
        jumps_m = np.random.normal(jump_mu, jump_sigma, sim_paths) * (dq_m > 0)
        dS_m = kappa_cal * (theta_cal - S_matrix[t-1, :]) * dt + sigma_p * S_matrix[t-1, :] * dW_m + jumps_m
        S_matrix[t, :] = np.maximum(0.0, S_matrix[t-1, :] + dS_m)
        
    path_averages = np.mean(S_matrix[1:, :], axis=0)
    var_95 = np.percentile(path_averages, 95)
    cvar_95 = np.mean(path_averages[path_averages > var_95])

    # 3. CCGT PLANT OPTIMIZATION (LSM & CLEAN SPARK SPREAD)
    gas_spot = theta_cal * 0.55
    eua_carbon = 70.0
    efficiency = 0.52
    emission_factor = 0.20
    vpc = (gas_spot / efficiency) + (eua_carbon * emission_factor)
    start_up_cost = 2.0
    
    css_matrix = S_matrix[1:, :] - vpc
    intrinsic_val = np.maximum(css_matrix, 0.0)
    
    # LSM Backward Induction
    active_cf = intrinsic_val[-1, :]
    df_factor = np.exp(-0.05 * (1.0 / 365.0))
    dispatch_decisions = np.zeros((delivery_days, sim_paths))
    
    for t in range(delivery_days - 1, 0, -1):
        spot_css = css_matrix[t, :]
        itm = np.where(spot_css > 0)[0]
        if len(itm) > 0:
            X = spot_css[itm]
            Y = active_cf[itm] * df_factor
            reg = np.polyfit(X, Y, deg=2)
            cont_val = np.polyval(reg, X)
            exercise_now = spot_css[itm] - start_up_cost
            ex_sub = np.where(exercise_now > cont_val)[0]
            ex_idx = itm[ex_sub]
            active_cf[ex_idx] = exercise_now[ex_sub]
            dispatch_decisions[t, ex_idx] = 1.0
            active_cf[~np.isin(np.arange(sim_paths), ex_idx)] *= df_factor
        else:
            active_cf *= df_factor

    # 4. GENERATING PROFESSIONAL VISUALIZATIONS
    os.makedirs('data', exist_ok=True)
    
    # FIGURE 1: MRJD Calibration & Price Series
    plt.figure(figsize=(10, 5))
    plt.plot(df_market.index, df_market['Price'], color='dodgerblue', alpha=0.5, label='Synthetic Spot Price (with Spikes)')
    plt.plot(df_market.index, df_market['Clean_Price'], color='black', linewidth=1.5, label='Cleansed Price (MRJD Base)')
    plt.axhline(y=theta_cal, color='crimson', linestyle='--', label=f'Long-term Mean (Theta = {theta_cal:.1f})')
    plt.title("1. MRJD Calibration & Spike Cleansing Engine")
    plt.xlabel("Date")
    plt.ylabel("Price (€/MWh)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('data/calibration_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    # FIGURE 2: Risk Profile (VaR vs CVaR)
    
    plt.figure(figsize=(10, 5))
    counts, bins, patches = plt.hist(path_averages, bins=80, density=True, alpha=0.6, color='royalblue', edgecolor='black')
    plt.axvline(x=var_95, color='orange', linestyle='--', linewidth=2, label=f'95% VaR: {var_95:.1f}')
    plt.axvline(x=cvar_95, color='red', linestyle='-.', linewidth=2, label=f'95% Expected Shortfall (CVaR): {cvar_95:.1f}')
    for patch in patches:
        if patch.get_x() > var_95:
            patch.set_facecolor('crimson')
            patch.set_alpha(0.8)
    plt.title("2. Portfolio Risk Analysis: Fat Tails & Expected Shortfall")
    plt.xlabel("Monthly Average Settlement Price (€/MWh)")
    plt.ylabel("Probability Density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('data/risk_profile.png', dpi=300, bbox_inches='tight')
    plt.close()

    # FIGURE 3: Plant Operations & Dispatch Rate (LSM)
    
    plt.figure(figsize=(10, 5))
    dispatch_rate = np.mean(dispatch_decisions, axis=1) * 100
    plt.plot(dispatch_rate, color='forestgreen', linewidth=2.5, marker='o', markersize=4)
    plt.title("3. CCGT Plant Optimal Dispatch Rate (LSM Continuation Value)")
    plt.xlabel("Delivery Timeline (Days)")
    plt.ylabel("Active Plant Capacity (%)")
    plt.grid(True, alpha=0.3)
    plt.savefig('data/plant_operations.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n[SUCCESS] All pipelines executed and high-res charts saved inside the 'data/' folder:")
    print(" - data/calibration_results.png")
    print(" - data/risk_profile.png")
    print(" - data/plant_operations.png")

if __name__ == "__main__":
    run_complete_energy_pipeline()