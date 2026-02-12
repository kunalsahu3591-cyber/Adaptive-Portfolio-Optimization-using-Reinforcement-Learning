"""
Synthetic Sample Data Generator for Adaptive Portfolio Optimization.
Generates realistic 5-minute interval OHLCV data for DASH, LTC, and STR.
"""

import os
import argparse
import numpy as np
import pandas as pd
import config


def generate_market_data(output_file=config.RAW_FILE, n_steps=3000, seed=42):
    """
    Generates synthetic geometric Brownian motion OHLCV market data.
    """
    np.random.seed(seed)
    
    start_time = pd.Timestamp("2023-01-01 00:00:00")
    timestamps = [start_time + pd.Timedelta(minutes=5 * i) for i in range(n_steps)]

    base_prices = {
        "DASH": 150.0,
        "LTC": 85.0,
        "STR": 0.25
    }

    volatilities = {
        "DASH": 0.002,
        "LTC": 0.0018,
        "STR": 0.003
    }

    records = []

    for coin in config.COINS:
        p0 = base_prices.get(coin, 100.0)
        vol = volatilities.get(coin, 0.002)

        # Log returns
        returns = np.random.normal(loc=0.00001, scale=vol, size=n_steps)
        price_series = p0 * np.exp(np.cumsum(returns))

        for t, close_p in zip(timestamps, price_series):
            # Generate realistic intraday spread
            open_p = close_p * (1.0 + np.random.normal(0, vol * 0.3))
            high_p = max(open_p, close_p) * (1.0 + abs(np.random.normal(0, vol * 0.5)))
            low_p = min(open_p, close_p) * (1.0 - abs(np.random.normal(0, vol * 0.5)))
            volume = max(100.0, np.random.lognormal(mean=7.0, sigma=0.8))
            weighted_avg = (high_p + low_p + 2 * close_p) / 4.0
            quote_volume = volume * weighted_avg

            records.append({
                "date": int(t.timestamp()),
                "coin": coin,
                "high": round(high_p, 4),
                "low": round(low_p, 4),
                "open": round(open_p, 4),
                "close": round(close_p, 4),
                "volume": round(volume, 2),
                "quoteVolume": round(quote_volume, 2),
                "weightedAverage": round(weighted_avg, 4)
            })

    df = pd.DataFrame(records)
    # Sort chronologically by date, then coin
    df = df.sort_values(["date", "coin"]).reset_index(drop=True)

    print(f"Generated {len(df)} synthetic market records across {len(config.COINS)} assets.")
    df.to_csv(output_file, index=False)
    print(f"Sample data written to '{output_file}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic cryptocurrency market data.")
    parser.add_argument("--output", default=config.RAW_FILE, help="Path for output raw CSV.")
    parser.add_argument("--steps", type=int, default=3000, help="Number of 5-minute timesteps per coin.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    generate_market_data(args.output, args.steps, args.seed)
