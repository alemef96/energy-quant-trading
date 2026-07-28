import os
import numpy as np
import matplotlib.pyplot as plt
from pricing import price_asian_call_mrjd
from forward_curve import build_baseload_curve, add_peak_offpeak, theta_for_month

def run_portfolio_risk_analysis(delivery_month=1):
    print("================ PORTFOLIO RISK & TAIL ANALYSIS ================")

    # 1. Anchor the model to the traded forward curve rather than a flat guess.
    #    theta is the forward level for the delivery month being analysed.
    curve = add_peak_offpeak(build_baseload_curve())
    theta = theta_for_month(curve, delivery_month)
    S0 = theta                     # start at the forward level for that delivery period
    kappa = 0.3
    sigma = 0.15
    lam = 0.0274       # Jump frequency
    jump_mu = 80.0     # Asymmetric spike magnitude mean
    jump_sigma = 20.0  # Spike volatility

    delivery_days = 30
    strike = round(theta * 1.08)   # ~8% out-of-the-money cap, relative to the forward
    risk_free_rate = 0.05
    sim_paths = 50000

    month_name = curve["Month"].iloc[delivery_month - 1]
    print(f"Delivery period: {month_name}  |  forward (theta) = {theta:.2f} \u20ac/MWh  |  strike = {strike} \u20ac/MWh")

    # 2. Run the Monte Carlo simulation engine, compensating the jump drift so the
    #    simulated mean reproduces the forward level (no systematic mispricing).
    print(f"Generating {sim_paths} simulated price paths via MRJD...")
    option_price, S_matrix = price_asian_call_mrjd(
        S0, kappa, theta, sigma, lam, jump_mu, jump_sigma,
        strike, delivery_days, risk_free_rate, paths=sim_paths,
        compensate_jumps=True
    )

    # 3. Extract path averages (the basis for electricity contract settlement)
    path_averages = np.mean(S_matrix[1:, :], axis=0)

    # 4. Quantify Tail Risk Metrics (Value at Risk & Expected Shortfall)
    # Since we model a power buyer or a short trader, risk lives in the high-price tail
    confidence_level = 0.95
    var_95 = np.percentile(path_averages, confidence_level * 100)
    cvar_95 = np.mean(path_averages[path_averages > var_95])

    print("\n--- RISK METRICS RESULTS ---")
    print(f"Baseline Expected Price (Mean): {np.mean(path_averages):.2f} €/MWh")
    print(f"95% Value at Risk (VaR):        {var_95:.2f} €/MWh")
    print(f"95% Expected Shortfall (CVaR):  {cvar_95:.2f} €/MWh")
    print(f"Asian Call Option Premium:      {option_price:.4f} €/MWh")

    # 5. Generate Advanced Quantitative Visualizations
    plt.figure(figsize=(14, 6))

    # Plot A: Sample Simulated Trajectories
    plt.subplot(1, 2, 1)
    sample_paths = 30
    plt.plot(S_matrix[:, :sample_paths], alpha=0.6, linewidth=1.5)
    plt.axhline(y=theta, color="black", linestyle="--", label=f"Forward level \u03b8 = {theta:.1f}")
    plt.title("MRJD Monte Carlo Price Paths (Sample)")
    plt.xlabel("Delivery Timeline (Days)")
    plt.ylabel("Electricity Spot Price (€/MWh)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot B: Probability Density and Tail Risk
    plt.subplot(1, 2, 2)
    counts, bins, patches = plt.hist(path_averages, bins=100, density=True, alpha=0.6, color='royalblue', edgecolor='black')

    # Draw vertical risk lines
    plt.axvline(x=np.mean(path_averages), color='green', linestyle='-', linewidth=2, label=f'Mean: {np.mean(path_averages):.1f}')
    plt.axvline(x=var_95, color='orange', linestyle='--', linewidth=2, label=f'95% VaR: {var_95:.1f}')
    plt.axvline(x=cvar_95, color='red', linestyle='-.', linewidth=2, label=f'95% CVaR: {cvar_95:.1f}')

    # Color the tail area beyond VaR in red to emphasize the Expected Shortfall region
    for patch in patches:
        if patch.get_x() > var_95:
            patch.set_facecolor('crimson')
            patch.set_alpha(0.7)

    plt.title(f"Distribution of Monthly Price Averages ({month_name} delivery)")
    plt.xlabel("Arithmetic Average Price (€/MWh)")
    plt.ylabel("Probability Density")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save chart to a static file for GitHub presentation
    os.makedirs('data', exist_ok=True)
    plot_path = 'data/risk_profile.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[Success] Diagnostic chart saved successfully to '{plot_path}'")

    return {"mean": float(np.mean(path_averages)), "var_95": float(var_95),
            "cvar_95": float(cvar_95), "option_price": float(option_price)}


if __name__ == "__main__":
    run_portfolio_risk_analysis()