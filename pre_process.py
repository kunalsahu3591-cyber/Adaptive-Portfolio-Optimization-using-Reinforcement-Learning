"""
Data Preprocessing and Feature Engineering Pipeline.
Transforms raw OHLCV price series into normalized indicators and multi-horizon rolling statistics.
"""

import os
import argparse
import pandas as pd
import config


def preprocess_data(input_file=config.RAW_FILE, output_file=config.FILE):
    """
    Computes normalized relative price ratios and rolling feature averages for coins.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Input dataset '{input_file}' not found. Please provide '{input_file}' or run generate_sample_data.py."
        )

    print(f"Loading raw dataset from '{input_file}'...")
    df = pd.read_csv(input_file)

    # Date parsing
    if pd.api.types.is_numeric_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], unit='s')
    else:
        df["date"] = pd.to_datetime(df["date"])

    # Filter targets and sort
    df = df[df["coin"].isin(config.COINS)].sort_values(["coin", "date"]).reset_index(drop=True)

    print("Calculating relative price metrics (vh, vl, vc)...")
    df["vh"] = df["high"] / df["open"]
    df["vl"] = df["low"] / df["open"]
    df["vc"] = df["close"] / df["open"]

    print("Calculating first differences...")
    df["open_s"] = df.groupby("coin")["open"].diff().fillna(0)
    df["volume_s"] = df.groupby("coin")["volume"].diff().fillna(0)
    df["quoteVolume_s"] = df.groupby("coin")["quoteVolume"].diff().fillna(0)
    df["weightedAverage_s"] = df.groupby("coin")["weightedAverage"].diff().fillna(0)

    base_scols = ["vh", "vl", "vc", "open_s", "volume_s", "quoteVolume_s", "weightedAverage_s"]

    print("Computing rolling window features (7, 14, 30 periods)...")
    for col in base_scols:
        for window in [7, 14, 30]:
            col_name = f"{col}_roll_{window}"
            df[col_name] = (
                df.groupby("coin")[col]
                .transform(lambda x: x.rolling(window, min_periods=1).mean().bfill())
            )

    # Sort back by date for temporal alignment in the environment
    df = df.sort_values("date").reset_index(drop=True)

    print(f"Saving preprocessed dataset to '{output_file}'...")
    df.to_csv(output_file, index=False)
    print(f"Preprocessing completed successfully. Shape: {df.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess OHLCV crypto market data.")
    parser.add_argument("--input", default=config.RAW_FILE, help="Path to input raw CSV.")
    parser.add_argument("--output", default=config.FILE, help="Path to output preprocessed CSV.")
    args = parser.parse_args()

    preprocess_data(args.input, args.output)
