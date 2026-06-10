# Energy Quantitative Trading: MRJD & Derivatives Pricing Framework

[cite_start]A production-ready quantitative framework designed for energy market risk management, spot price calibration, and derivatives pricing[cite: 94]. 

[cite_start]Unlike equity or FX markets, electricity cannot be economically stored at a large scale[cite: 1, 20]. [cite_start]Production must match consumption instantaneously [cite: 2, 21][cite_start], rendering traditional Geometric Brownian Motion models obsolete[cite: 2, 23]. [cite_start]This repository implements a specialized **Mean-Reverting Jump-Diffusion (MRJD)** model to capture power market dynamics[cite: 4, 62].

## Core Features
* [cite_start]**Spike Cleansing**: A rolling $3\sigma$ threshold filter to isolate structural market spikes (caused by grid shocks or extreme weather) from daily diffusive noise[cite: 3, 133].
* [cite_start]**OLS Calibration**: Linearization of the underlying Ornstein-Uhlenbeck process via Euler-Maruyama discretization to estimate speed of mean reversion ($\kappa$), long-term mean ($\theta$), and volatility ($\sigma$)[cite: 135, 136, 137].
* [cite_start]**Asian Options Pricing**: A Monte Carlo simulation engine designed for path-dependent power derivatives settled on arithmetic price averages[cite: 141, 145].

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