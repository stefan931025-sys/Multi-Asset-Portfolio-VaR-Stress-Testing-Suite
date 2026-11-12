from __future__ import annotations

"""
Module: tests/test_risk_engine.py
Description: Unit tests for PortfolioRiskEngine matching clean method signatures.
"""

import pytest
import numpy as np
import pandas as pd
from core.risk_engine import PortfolioRiskEngine

@pytest.fixture
def sample_portfolio_data() -> tuple[pd.Series, pd.DataFrame]:
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="B")
    returns = pd.DataFrame({
        "Equity": np.random.normal(0.0005, 0.012, 100),
        "FX": np.random.normal(0.0001, 0.008, 100),
        "FixedIncome": np.random.normal(0.0002, 0.005, 100)
    }, index=dates)
    weights = pd.Series({"Equity": 0.5, "FX": 0.3, "FixedIncome": 0.2})
    return weights, returns

def test_parametric_var(sample_portfolio_data: tuple[pd.Series, pd.DataFrame]) -> None:
    weights, returns = sample_portfolio_data
    engine = PortfolioRiskEngine(weights=weights)
    var_95 = engine.compute_parametric_var(returns=returns, confidence_level=0.95)
    assert isinstance(var_95, float)
    assert var_95 > 0.0

def test_monte_carlo_var(sample_portfolio_data: tuple[pd.Series, pd.DataFrame]) -> None:
    weights, returns = sample_portfolio_data
    engine = PortfolioRiskEngine(weights=weights)
    mc_var = engine.compute_monte_carlo_var(returns=returns, confidence_level=0.99, simulations=1000)
    assert isinstance(mc_var, float)
    assert mc_var > 0.0

def test_stress_testing(sample_portfolio_data: tuple[pd.Series, pd.DataFrame]) -> None:
    weights, returns = sample_portfolio_data
    engine = PortfolioRiskEngine(weights=weights)
    portfolio_rets = returns.dot(weights)
    stressed_loss = engine.run_stress_test(portfolio_returns=portfolio_rets, shock_multiplier=2.5)
    assert isinstance(stressed_loss, float)
    assert stressed_loss > 0.0
