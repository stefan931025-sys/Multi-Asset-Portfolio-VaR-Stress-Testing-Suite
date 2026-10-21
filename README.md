# Multi-Asset Portfolio VaR & Risk Attribution Suite

An institutional risk management tool for computing Value-at-Risk (VaR) and conducting tail-risk stress testing across multi-asset portfolios (Equities, FX, and Fixed Income).

## Key Features
* Dual VaR Calculation Methodologies: Implements both Parametric (Variance-Covariance) and Monte Carlo simulation engines (10,000+ trials).
* Cross-Asset Correlation Modeling: Accounts for cross-asset correlations across equity, currency, and fixed income instruments.
* Confidence Level Scaling: Computes customized 95% and 99% VaR metrics across 1-day to multi-day horizons.

## Tech Stack
* Language: Python
* Libraries: NumPy, Pandas, SciPy

## Quick Start
git clone https://github.com/yourusername/portfolio-var-risk-suite.git
cd portfolio-var-risk-suite
python portfolio_var.py
