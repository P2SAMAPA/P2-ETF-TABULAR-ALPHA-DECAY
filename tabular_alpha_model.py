"""
Tabular Alpha + Alpha Decay model using LightGBM regression.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import curve_fit
import config

class TabularAlphaDecayModel:
    def __init__(self, lgb_params=None, decay_max_lag=21):
        self.lgb_params = lgb_params or {}
        self.decay_max_lag = decay_max_lag
        self.model = None
        self.half_life = None
        self.decay_factor = None
        self.feature_names = None

    def fit(self, features: pd.DataFrame, target: pd.Series):
        # features index is date, but we don't need grouping for regression
        X = features.drop(columns=['ticker'])
        y = target
        self.feature_names = X.columns.tolist()

        # Train LightGBM regression
        dataset = lgb.Dataset(X, label=y)
        self.model = lgb.train(self.lgb_params, dataset, num_boost_round=200)

        # Predict for decay estimation
        preds = self.model.predict(X)
        df = features[['ticker']].copy()
        df['pred'] = preds
        df['target'] = y.values
        df['date'] = features.index

        self._estimate_decay(df)
        return True

    def _estimate_decay(self, df: pd.DataFrame):
        all_corrs = []
        for lag in range(1, self.decay_max_lag + 1):
            corrs = []
            for ticker in df['ticker'].unique():
                sub = df[df['ticker'] == ticker].sort_values('date')
                if len(sub) < lag + config.DECAY_MIN_SAMPLES:
                    continue
                corr = sub['pred'].iloc[:-lag].corr(sub['target'].iloc[lag:])
                if not np.isnan(corr):
                    corrs.append(corr)
            if corrs:
                all_corrs.append((lag, np.mean(corrs)))
            else:
                break

        if len(all_corrs) < 2:
            self.half_life = 5.0
            self.decay_factor = np.exp(-np.log(2) / self.half_life)
            return

        lags, corrs = zip(*all_corrs)
        lags = np.array(lags)
        corrs = np.array(corrs)

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
        X = features.drop(columns=['ticker'])
        preds = self.model.predict(X)
        return pd.Series(preds, index=features.index)

    def predict_with_decay(self, features: pd.DataFrame) -> pd.DataFrame:
        raw = self.predict(features)
        tickers = features['ticker'].values
        df = pd.DataFrame({'ticker': tickers, 'raw_pred': raw})
        latest = df.groupby('ticker').last().reset_index()
        latest['decay_adjusted'] = latest['raw_pred'] * self.decay_factor
        latest['half_life'] = self.half_life
        return latest
