from __future__ import annotations

"""
Module: core/risk_engine.py
Description: Institutional Risk Management Engine for Parametric VaR, Monte Carlo VaR, and Stress Testing.
"""

import numpy as np
import pandas as pd

class PortfolioRiskEngine:
    """
    Computes portfolio risk metrics including Parametric VaR, Monte Carlo VaR, and Stress Testing.
    """
    def __init__(self, weights: pd.Series, random_seed: int | None = 42) -> None:
        self.weights = weights
        self.random_seed = random_seed

    def compute_parametric_var(self, returns: pd.DataFrame, confidence_level: float = 0.95) -> float:
        """Computes parametric variance-covariance VaR."""
        cov_matrix = returns.cov()
        portfolio_variance = np.dot(self.weights.values, np.dot(cov_matrix.values, self.weights.values))
        portfolio_volatility = np.sqrt(portfolio_variance * 252)
        
        # Z-score lookup for confidence level
        z_score = 1.645 if confidence_level >= 0.95 else 1.282
        var = portfolio_volatility * z_score
        return float(var)

    def compute_monte_carlo_var(self, returns: pd.DataFrame, confidence_level: float = 0.99, simulations: int = 10000) -> float:
        """Computes Monte Carlo simulation VaR."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        
        # Simulate correlated paths using Cholesky decomposition
        L = np.linalg.cholesky(cov_matrix)
        z = np.random.normal(size=(simulations, len(self.weights)))
        simulated_returns = mean_returns + np.dot(z, L.T)
        
        portfolio_sim_returns = np.dot(simulated_returns, self.weights.values)
        var = np.percentile(portfolio_sim_returns, (1 - confidence_level) * 100)
        return float(abs(var))

    def run_stress_test(self, portfolio_returns: pd.Series, shock_multiplier: float = 2.5) -> float:
        """Applies stress test shock multiplier to evaluate tail risk."""
        stressed_loss = float(portfolio_returns.std() * shock_multiplier * np.sqrt(252))
        return stressed_loss
