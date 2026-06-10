# Energy Quantitative Trading: MRJD & Derivatives Pricing Framework

A production-ready quantitative framework designed for energy market risk management, spot price calibration, and derivatives pricing. 

Unlike equity or FX markets, electricity cannot be economically stored at a large scale. Production must match consumption instantaneously, rendering traditional Geometric Brownian Motion models obsolete. This repository implements a specialized **Mean-Reverting Jump-Diffusion (MRJD)** model to capture power market dynamics.

## Core Features
* **Spike Cleansing**: A rolling $3\sigma$ threshold filter to isolate structural market spikes (caused by grid shocks or extreme weather) from daily diffusive noise.
* **OLS Calibration**: Linearization of the underlying Ornstein-Uhlenbeck process via Euler-Maruyama discretization to estimate speed of mean reversion ($\kappa$), long-term mean ($\theta$), and volatility ($\sigma$).
* **Asian Options Pricing**: A Monte Carlo simulation engine designed for path-dependent power derivatives settled on arithmetic price averages.

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