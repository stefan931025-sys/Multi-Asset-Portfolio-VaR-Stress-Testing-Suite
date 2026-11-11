# Multi-Asset Portfolio VaR & Risk Attribution Suite

[![Multi-Asset Portfolio VaR CI](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/stefan931025-sys/Multi-Asset-Portfolio-VaR-Stress-Testing-Suite)
[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An institutional risk management tool for computing Value-at-Risk (VaR) and conducting tail-risk stress testing across multi-asset portfolios (Equities, FX, and Fixed Income).

---

## Key Features

* **Dual VaR Calculation Methodologies:** Implements both Parametric (Variance-Covariance) and Monte Carlo simulation engines (10,000+ trials).
* **Cross-Asset Correlation Modeling:** Accounts for cross-asset correlations across equity, currency, and fixed income instruments.
* **Confidence Level Scaling:** Computes customized 95% and 99% VaR metrics across 1-day to multi-day horizons.
* **Tail-Risk Stress Testing:** Simulates historical and hypothetical macro stress scenarios to measure portfolio vulnerability.

---

## Mathematical Specifications

### 1. Parametric Portfolio VaR
$$\text{VaR}_{\alpha} = z_{\alpha} \cdot \sigma_p \cdot \sqrt{t}$$
Where $\sigma_p = \sqrt{w^T \Sigma w}$ is the portfolio volatility derived from asset weights vector $w$ and covariance matrix $\Sigma$.

### 2. Monte Carlo Simulation Engine
Simulating asset price paths via Geometric Brownian Motion (GBM) with Cholesky decomposition of the correlation matrix:
$$S_{i, t} = S_{i, t-1} \exp\left( \left(\mu_i - \frac{\sigma_i^2}{2}\right)\Delta t + \sigma_i \sqrt{\Delta t} Z_i \right)$$

---

## System Architecture & Data Flow

