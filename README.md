# Energy Quantitative Trading: MRJD & Derivatives Pricing Framework

A production-ready quantitative framework designed for energy market risk management, spot price calibration, and derivatives pricing. 

Unlike equity or FX markets, electricity cannot be economically stored at a large scale. Production must match consumption instantaneously, rendering traditional Geometric Brownian Motion models obsolete. This repository implements a specialized **Mean-Reverting Jump-Diffusion (MRJD)** model to capture power market dynamics.

## Core Features
* **Spike Cleansing**: A rolling $3\sigma$ threshold filter to isolate structural market spikes (caused by grid shocks or extreme weather) from daily diffusive noise.
* **OLS Calibration**: Linearization of the underlying Ornstein-Uhlenbeck process via Euler-Maruyama discretization to estimate speed of mean reversion ($\kappa$), long-term mean ($\theta$), and volatility ($\sigma$).
* **Asian Options Pricing**: A Monte Carlo simulation engine designed for path-dependent power derivatives settled on arithmetic price averages.

## Quantitative Risk Analysis & Visual Insights

By executing the Monte Carlo engine over 50,000 independent simulated paths under the calibrated MRJD framework, we map the asymmetric risk profile of the power portfolio:

![Power Portfolio Risk Profile](data/risk_profile.png)

### Key Metrics & Strategic Interpretation
* **Expected Price Baseline**: The equilibrium pricing remains anchored near the marginal fuel cost baseline ($\theta = 60.00$ €/MWh) due to the aggressive speed of mean reversion ($\kappa = 0.3$).
* **95% Value at Risk (VaR)**: Indicates a 95% probability that the delivery price will not exceed the quantified threshold. However, VaR fails to capture the structural severity of power price spikes.
* **95% Expected Shortfall (CVaR)**: Highlights the conditional tail expectation. In the 5% worst-case scenarios (the red tail region in the histogram), the average price exposure escalates dramatically. This proves that for highly skewed energy distributions, CVaR is the only reliable capital allocation metric for risk desks.

## Repository Structure
```text
energy-quant-trading/
├── data/                  # Local storage for historical price data
├── src/                   # Main source code
│   ├── __init__.py
│   ├── calibration.py     # MRJD calibration engine
│   └── pricing.py         # Monte Carlo pricing simulator
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation


