"""
Main training script for Tabular Alpha Decay engine.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from tabular_alpha_model import TabularAlphaDecayModel
import push_results

def run_tabular_alpha():
    print(f"=== P2-ETF-TABULAR-ALPHA-DECAY Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    df_master = df_master[df_master['Date'] >= config.TRAIN_START]

    macro = data_manager.prepare_macro_features(df_master)

    all_results = {}
    top_picks = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        returns = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue

        # Build features and targets for full history
        features_df = data_manager.build_features(returns, macro)
        if len(features_df) < config.MIN_OBSERVATIONS:
            continue

        # Train model on full history
        model = TabularAlphaDecayModel(
            lgb_params=config.LGB_PARAMS,
            decay_max_lag=config.DECAY_MAX_LAG
        )
        print("  Training LightGBM ranker and estimating decay...")
        model.fit(features_df.drop(columns=['target']), features_df['target'])

        # Predict on the most recent day per ticker
        latest_features = features_df[features_df['ticker'].isin(tickers)]
        latest_features = latest_features.groupby('ticker').last().reset_index()

        predictions = model.predict_with_decay(latest_features)

        # Build universe results
        universe_results = {}
        for _, row in predictions.iterrows():
            ticker = row['ticker']
            universe_results[ticker] = {
                "ticker": ticker,
                "raw_pred": row['raw_pred'],
                "decay_adjusted": row['decay_adjusted'],
                "half_life": row['half_life']
            }

        all_results[universe_name] = universe_results
        sorted_tickers = sorted(universe_results.items(),
                                key=lambda x: x[1]["decay_adjusted"], reverse=True)
        top_picks[universe_name] = [
            {k: v for k, v in d.items() if k != 'ticker'} | {"ticker": t}
            for t, d in sorted_tickers[:3]
        ]

    output_payload = {
        "run_date": config.TODAY,
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith("_") and k.isupper() and k != "HF_TOKEN"},
        "daily_trading": {
            "universes": all_results,
            "top_picks": top_picks
        }
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_tabular_alpha()
