# Multi-Asset Portfolio VaR & Risk Attribution Suite

![CI Workflow](https://github.com/stefan931025-sys/Multi-Asset-Portfolio-VaR-Stress-Testing-Suite/actions/workflows/ci.yml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An institutional risk management tool for computing Value-at-Risk (VaR) and conducting tail-risk stress testing across multi-asset portfolios (Equities, FX, and Fixed Income).

## 🚀 Key Features

* **Dual VaR Calculation Methodologies:** Implements both Parametric (Variance-Covariance) and Monte Carlo simulation engines (10,000+ trials).
* **Cross-Asset Correlation Modeling:** Accounts for cross-asset correlations across equity, currency, and fixed income instruments.
* **Confidence Level Scaling:** Computes customized 95% and 99% VaR metrics across 1-day to multi-day horizons.
* **Tail-Risk Stress Testing:** Simulates historical and hypothetical macro stress scenarios to measure portfolio vulnerability.

---

## 📊 Mathematical Specifications

### 1. Parametric Portfolio VaR

$$\text{VaR}_{\alpha} = z_{\alpha} \cdot \sigma_p \cdot \sqrt{t}$$

Where $\sigma_p = \sqrt{w^T \Sigma w}$ is the portfolio volatility derived from asset weights vector $w$ and covariance matrix $\Sigma$.

### 2. Monte Carlo Simulation Engine

Simulating asset price paths via Geometric Brownian Motion (GBM) with Cholesky decomposition of the correlation matrix:

$$S_t = S_{t-1} \exp\left((\mu - \frac{\sigma^2}{2})\Delta t + \sigma \sqrt{\Delta t} Z\right)$$

---

## 📁 Repository Structure

```text
Multi-Asset-Portfolio-VaR-Stress-Testing-Suite/
├── .github/
│   └── workflows/
│       └── ci.yml         # GitHub Actions automated test workflow
├── core/
│   └── risk_engine.py     # Parametric & Monte Carlo VaR engine
├── tests/
│   └── test_risk_engine.py# Automated pytest suite
├── main.py                # Main execution script
├── repo_auditor.py        # Static AST analysis governance tool
├── requirements.txt       # Python dependencies
└── README.md              # System documentation
