# P2-ETF-TABULAR-ALPHA-DECAY

**LightGBM Cross‑Sectional Ranking with Signal Half‑Life Decay**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-TABULAR-ALPHA-DECAY/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-TABULAR-ALPHA-DECAY/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--tabular--alpha--decay--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-tabular-alpha-decay-results)

## Overview

`P2-ETF-TABULAR-ALPHA-DECAY` combines a **LightGBM cross‑sectional ranking** engine with **alpha half‑life estimation**. The model is trained on the full 2008‑2026 dataset using lagged returns and macro features. The decay of predictive power over time is measured via auto‑correlation and modeled as exponential decay, giving a half‑life in days. The final ranking uses decay‑adjusted expected returns.

## Methodology

- **Features**: Lagged returns (1, 5, 21 days) and current macro values (VIX, DXY, T10Y2Y, TBILL_3M).
- **Ranker**: LightGBM LambdaRank trained on next‑day returns.
- **Decay**: Exponential fit to the correlation between predictions and forward returns at lags 1‑21 days.
- **Adjusted Return** = Raw Prediction × exp(-ln(2) / half‑life)

## Usage

```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
