"""
main.py — single entry point for the energy-quant-trading pipeline.

Runs the three independent building blocks in sequence and regenerates every
chart in data/:

    1. MRJD calibration        -> data/calibration_results.png
    2. Portfolio tail risk     -> data/risk_profile.png        (VaR / CVaR)
    3. Plant real-option value -> data/weather_dispatch.png    (weather -> merit order -> dispatch)

Each module is also runnable on its own (python calibration.py, python risk_analysis.py,
python plant_valuation.py); this script just wires them together.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibration import generate_synthetic_power_data, filter_spikes, calibrate_mrjd
from risk_analysis import run_portfolio_risk_analysis
import forward_curve
import plant_valuation


def make_calibration_chart():
    """Calibrate the MRJD model and plot the price / cleansed-price / theta diagnostic."""
    df = generate_synthetic_power_data()
    df = filter_spikes(df)
    params = calibrate_mrjd(df)
    theta = params["Theta (Long-term Mean Price)"]

    print("--- CALIBRATION RESULTS ---")
    for k, v in params.items():
        print(f"    {k}: {v:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["Price"], color="dodgerblue", alpha=0.5,
             label="Synthetic spot price (with spikes)")
    plt.plot(df.index, df["Clean_Price"], color="black", linewidth=1.5,
             label="Cleansed price (MRJD base)")
    plt.axhline(y=theta, color="crimson", linestyle="--",
                label=f"Long-term mean (theta = {theta:.1f})")
    plt.title("MRJD calibration & spike-cleansing engine")
    plt.xlabel("Date"); plt.ylabel("Price (\u20ac/MWh)")
    plt.grid(True, alpha=0.3); plt.legend()
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/calibration_results.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("    -> data/calibration_results.png")
    return params


def main():
    os.makedirs("data", exist_ok=True)

    print("\n[1/4] MRJD calibration ...")
    make_calibration_chart()

    print("\n[2/4] Forward curve construction ...")
    forward_curve.main()

    print("\n[3/4] Portfolio tail risk (VaR / CVaR) ...")
    run_portfolio_risk_analysis()

    print("\n[4/4] Weather-driven plant valuation ...")
    plant_valuation.main()

    print("\n[DONE] All charts regenerated in data/.")


if __name__ == "__main__":
    main()