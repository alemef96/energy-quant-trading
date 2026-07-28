import numpy as np
import pandas as pd


def price_asian_call_mrjd(S0, kappa, theta, sigma, lam, jump_mu, jump_sigma, K, T, r,
                          paths=20000, compensate_jumps=False):
    """
    Prices an Asian Call Option using a vectorized Monte Carlo simulation
    under the Mean-Reverting Jump-Diffusion (MRJD) framework.

    Parameters:
    - S0: Initial spot price
    - kappa: Speed of mean reversion
    - theta: Long-term mean level. EITHER a scalar (flat long-term mean) OR an
             array of length T (a time-varying target theta(t) — e.g. the forward
             curve, so the model is anchored to quoted market levels).
    - sigma: Diffusion volatility
    - lam: Jump intensity (Poisson process lambda)
    - jump_mu: Mean of the jump magnitude
    - jump_sigma: Volatility of the jump magnitude
    - K: Option strike price
    - T: Days to maturity/delivery period
    - r: Annual risk-free interest rate
    - paths: Number of simulated Monte Carlo trajectories
    - compensate_jumps: if True, subtract the expected jump contribution
             (lambda * jump_mu) from the drift, so that the simulated mean
             reverts to theta itself rather than overshooting it. This is what
             lets the simulation reprice the forward curve instead of drifting
             systematically above it.

    Returns (option_price, S) where S has shape (T+1, paths).
    """
    dt = 1.0  # Daily time step

    # Accept a scalar or a full theta(t) path; broadcast to length T
    theta_path = np.asarray(theta, dtype=float)
    if theta_path.ndim == 0:
        theta_path = np.full(T, float(theta))
    elif len(theta_path) != T:
        raise ValueError(f"theta must be scalar or length T={T}, got {len(theta_path)}")

    # Expected jump contribution per day; removing it makes E[S] revert to theta
    jump_drift = lam * jump_mu if compensate_jumps else 0.0

    # Grid initialization: rows = days (time steps), columns = independent simulation paths
    S = np.zeros((T + 1, paths))
    S[0, :] = S0

    # Vectorized Monte Carlo Path Generation across all paths simultaneously
    for t in range(1, T + 1):
        # Generate standard normal random variables for the continuous diffusion
        dW = np.random.normal(0, 1, paths)

        # Generate Poisson random variables for jump arrivals
        dq = np.random.poisson(lam * dt, paths)

        # Calculate jump sizes (only applied if a jump occurs, i.e., dq > 0)
        jumps = np.random.normal(jump_mu, jump_sigma, paths) * (dq > 0)

        # Euler-Maruyama discretization step, targeting theta(t) on this day
        theta_t = theta_path[t - 1]
        dS = (kappa * (theta_t - S[t-1, :]) * dt
              + sigma * S[t-1, :] * dW
              + jumps - jump_drift * dt)

        # Bound prices to prevent non-physical negative drops
        S[t, :] = np.maximum(5.0, S[t-1, :] + dS)

    # Calculate the arithmetic average for each path over the delivery period (excluding t=0)
    path_averages = np.mean(S[1:, :], axis=0)

    # Calculate option payoffs at maturity
    payoffs = np.maximum(path_averages - K, 0)

    # Discount the average expected payoff back to present value
    time_to_maturity_years = T / 365.0
    option_price = np.exp(-r * time_to_maturity_years) * np.mean(payoffs)

    return option_price, S


if __name__ == "__main__":
    print("================ OPTION PRICING ENGINE ================")

    # Inputs: parameters closely aligned with our calibration script
    S0_market = 60.0
    kappa_calib = 0.3
    theta_calib = 60.0
    sigma_calib = 0.15
    lambda_calib = 0.0274  # Calibrated jump frequency
    jump_mu_est = 80.0
    jump_sigma_est = 20.0

    # Option Contract Details
    strike_price = 65.0    # Out-of-the-money Call option
    delivery_days = 30     # 1-month contract length
    risk_free_rate = 0.05  # 5% annual risk-free rate
    sim_paths = 50000      # High path count for statistical precision

    print(f"Pricing a {delivery_days}-Day Asian Call Option...")
    print(f"Initial Price: {S0_market} €/MWh | Strike Price: {strike_price} €/MWh\n")

    price, paths_matrix = price_asian_call_mrjd(
        S0=S0_market, kappa=kappa_calib, theta=theta_calib, sigma=sigma_calib,
        lam=lambda_calib, jump_mu=jump_mu_est, jump_sigma=jump_sigma_est,
        K=strike_price, T=delivery_days, r=risk_free_rate, paths=sim_paths
    )

    print(f">>> Estimated Asian Call Option Price: {price:.4f} €/MWh")
    print(f"Simulated paths matrix shape: {paths_matrix.shape} (Days x Paths)")