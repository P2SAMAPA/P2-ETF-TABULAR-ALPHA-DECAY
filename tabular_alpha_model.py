"""
Tabular Alpha + Alpha Decay model using LightGBM and auto‑correlation decay.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import curve_fit

class TabularAlphaDecayModel:
    def __init__(self, lgb_params=None, decay_max_lag=21):
        self.lgb_params = lgb_params or {}
        self.decay_max_lag = decay_max_lag
        self.model = None
        self.half_life = None          # global half-life (days)
        self.decay_factor = None       # exp(-ln(2)/half_life) for 1 day
        self.feature_names = None

    def fit(self, features: pd.DataFrame, target: pd.Series):
        """
        Train LightGBM ranker and estimate decay half-life.
        features: DataFrame with columns (ret_1d, ret_5d, ret_21d, macro..., ticker)
        target: Series of next-day returns
        """
        # Drop ticker column for training
        X = features.drop(columns=['ticker'])
        self.feature_names = X.columns.tolist()
        y = target

        # Train LightGBM
        dataset = lgb.Dataset(X, label=y, group=None)
        self.model = lgb.train(self.lgb_params, dataset, num_boost_round=200)

        # Predict on training data to compute decay
        preds = self.model.predict(X)

        # Add predictions and ticker back to compute decay per ETF
        df = features[['ticker']].copy()
        df['pred'] = preds
        df['actual'] = y.values
        df['date'] = features.index   # assuming index is date

        self._estimate_decay(df)
        return True

    def _estimate_decay(self, df: pd.DataFrame):
        """
        Estimate global half-life of predictive power using auto‑correlation
        between predictions and forward returns across all ETFs.
        """
        all_corrs = []
        for lag in range(1, self.decay_max_lag + 1):
            corrs = []
            for ticker in df['ticker'].unique():
                sub = df[df['ticker'] == ticker].sort_values('date')
                if len(sub) < lag + config.DECAY_MIN_SAMPLES:
                    continue
                corr = sub['pred'].iloc[:-lag].corr(sub['actual'].iloc[lag:])
                if not np.isnan(corr):
                    corrs.append(corr)
            if corrs:
                all_corrs.append((lag, np.mean(corrs)))
            else:
                break

        if len(all_corrs) < 2:
            self.half_life = 5.0  # default
            self.decay_factor = np.exp(-np.log(2) / self.half_life)
            return

        lags, corrs = zip(*all_corrs)
        lags = np.array(lags)
        corrs = np.array(corrs)

        # Exponential decay fit: corr(lag) = A * exp(-B * lag)
        def exp_decay(x, A, B):
            return A * np.exp(-B * x)

        try:
            popt, _ = curve_fit(exp_decay, lags, corrs, p0=[0.1, 0.1], maxfev=5000)
            A, B = popt
            if B > 0:
                self.half_life = np.log(2) / B
            else:
                self.half_life = 5.0
        except Exception:
            self.half_life = 5.0

        self.decay_factor = np.exp(-np.log(2) / self.half_life)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict raw next-day return for each ETF."""
        X = features.drop(columns=['ticker'])
        preds = self.model.predict(X)
        return pd.Series(preds, index=features.index)

    def predict_with_decay(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with raw_pred and decay_adjusted_pred per ETF."""
        raw = self.predict(features)
        tickers = features['ticker'].values
        df = pd.DataFrame({'ticker': tickers, 'raw_pred': raw})
        # Apply decay adjustment only for the most recent row per ticker? We'll average per ticker.
        # For daily ranking we just need the latest prediction, decay-adjusted.
        # We'll group by ticker and take the last prediction.
        latest = df.groupby('ticker').last().reset_index()
        latest['decay_adjusted'] = latest['raw_pred'] * self.decay_factor
        latest['half_life'] = self.half_life
        return latest
