# Multi-Asset Portfolio VaR & Risk Attribution Suite

[![CI Pipeline](https://github.com/stefan931025-sys/Multi-Asset-Portfolio-VaR-Stress-Testing-Suite/actions/workflows/ci.yml/badge.svg)](https://github.com/stefan931025-sys/Multi-Asset-Portfolio-VaR-Stress-Testing-Suite/actions/workflows/ci.yml)

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

An institutional risk management tool for computing Value-at-Risk (VaR) and conducting tail-risk stress testing across multi-asset portfolios (Equities, FX, and Fixed Income).

## Key Features

* **Dual VaR Calculation Methodologies:** Implements both Parametric (Variance-Covariance) and Monte Carlo simulation engines (10,000+ trials).
* **Cross-Asset Correlation Modeling:** Accounts for cross-asset correlations across equity, currency, and fixed income instruments.
* **Confidence Level Scaling:** Computes customized 95% and 99% VaR metrics across 1-day to multi-day horizons.

## Tech Stack

* **Language:** Python
* **Libraries:** NumPy, Pandas, SciPy

## Quick Start

```bash
git clone [https://github.com/stefan931025-sys/Multi-Asset-Portfolio-VaR-Stress-Testing-Suite.git](https://github.com/stefan931025-sys/Multi-Asset-Portfolio-VaR-Stress-Testing-Suite.git)
cd Multi-Asset-Portfolio-VaR-Stress-Testing-Suite
python portfolio_var.py
