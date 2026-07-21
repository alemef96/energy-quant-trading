import numpy as np
import pandas as pd

def generate_synthetic_power_data(days=365, seed=42):
    """
    Simulates a historical Spot Power price series exhibiting Mean Reversion,
    Standard Diffusion, and Poisson Jumps (MRJD).
    """
    np.random.seed(seed)
    dt = 1.0  # Daily time step

    # Target true parameters
    kappa_true = 0.3    # Speed of mean reversion
    theta_true = 60.0   # Long-term marginal cost equilibrium (€/MWh)
    sigma_true = 0.15   # Volatility of the diffusion component
    lambda_true = 0.05  # Jump probability (5% chance per day)
    jump_mu = 80.0      # Average jump magnitude
    jump_sigma = 20.0   # Jump volatility

    prices = np.zeros(days)
    prices[0] = theta_true

    for t in range(1, days):
        S_old = prices[t-1]

        # 1. Brownian Diffusion Component
        dW = np.random.normal(0, np.sqrt(dt))
        diffusion = sigma_true * S_old * dW

        # 2. Poisson Jump Component
        dq = np.random.poisson(lambda_true * dt)
        jump = 0.0
        if dq > 0:
            jump = np.random.normal(jump_mu, jump_sigma)

        # Euler-Maruyama SDE Discretization
        dS = kappa_true * (theta_true - S_old) * dt + diffusion + jump
        prices[t] = max(5.0, S_old + dS)  # Prevent unphysical negative spot drops

    dates = pd.date_range(start="2026-01-01", periods=days, freq="D")
    return pd.DataFrame({"Price": prices}, index=dates)


def filter_spikes(df, threshold_sigma=3.0):
    """
    Applies a rolling threshold statistical filter to isolate price spikes
    from normal continuous diffusive regimes.
    """
    df = df.copy()
    df['Return'] = df['Price'].pct_change()

    # 20-day lookback window for rolling statistics
    rolling_window = 20
    df['Rolling_Mean'] = df['Return'].rolling(window=rolling_window).mean()
    df['Rolling_Std'] = df['Return'].rolling(window=rolling_window).std()

    # Identify Jumps via 3-Sigma Outlier Detection rule
    df['Is_Jump'] = np.abs(df['Return'] - df['Rolling_Mean']) > (threshold_sigma * df['Rolling_Std'])
    df['Is_Jump'] = df['Is_Jump'].fillna(False)

    # Clean the price series by substituting spikes with the last valid non-spike price (Vectorized)
    df['Clean_Price'] = np.where(df['Is_Jump'], np.nan, df['Price'])
    df['Clean_Price'] = df['Clean_Price'].ffill()

    # Edge case: if the very first day is a jump, ffill() leaves a NaN -> refill with original price.
    df['Clean_Price'] = df['Clean_Price'].fillna(df['Price'])

    return df


def calibrate_mrjd(df_processed):
    """
    Fits Kappa, Theta, and Sigma parameters onto the cleansed price series
    using an Ordinary Least Squares (OLS) regression on the OU process.
    """
    S = df_processed['Clean_Price'].values

    # Compute dS_t and lagged price S_t
    dS = S[1:] - S[:-1]
    S_lag = S[:-1]

    # OLS Linear Regression: dS = intercept + slope * S_lag + error
    poly = np.polyfit(S_lag, dS, deg=1)
    slope = poly[0]
    intercept = poly[1]

    # Map discrete OLS coefficients back to continuous SDE parameters
    kappa_calibrated = -slope
    theta_calibrated = intercept / kappa_calibrated

    # Extract diffusion volatility from scaled residuals
    residuals = dS - (intercept + slope * S_lag)
    scaled_residuals = residuals / S_lag
    sigma_calibrated = np.std(scaled_residuals)

    # Calibrate Jump Frequency (Lambda) based on filtered events
    jump_events = df_processed[df_processed['Is_Jump'] == True]
    lambda_calibrated = len(jump_events) / len(df_processed)

    return {
        "Kappa (Mean Reversion Speed)": kappa_calibrated,
        "Theta (Long-term Mean Price)": theta_calibrated,
        "Sigma (Diffusion Volatility)": sigma_calibrated,
        "Lambda (Jump Intensity)": lambda_calibrated
    }


if __name__ == "__main__":
    print("1. Executing ingestion pipeline (Simulating market data)...")
    raw_market_data = generate_synthetic_power_data()

    print("2. Running statistical Spike Cleansing filter...")
    processed_market_data = filter_spikes(raw_market_data)

    print("3. Fitting MRJD parameters via linear regression...")
    calibrated_params = calibrate_mrjd(processed_market_data)

    print("\n================ CALIBRATION RESULTS ================")
    for parameter, value in calibrated_params.items():
        print(f"{parameter}: {value:.4f}")

    print(f"\nTotal isolated structural market jumps: {processed_market_data['Is_Jump'].sum()}")