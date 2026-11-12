from __future__ import annotations

"""
Module: tests/test_risk_engine.py
Description: Unit tests for the PortfolioRiskEngine covering Parametric VaR, Monte Carlo VaR, and Stress Testing.
"""

import pytest
import numpy as np
import pandas as pd
from core.risk_engine import PortfolioRiskEngine

@pytest.fixture
def sample_portfolio_data() -> tuple[pd.Series, pd.DataFrame]:
    """Fixture providing synthetic asset returns and portfolio weights for testing."""
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
    """Validates parametric variance-covariance VaR calculation."""
    weights, returns = sample_portfolio_data
    engine = PortfolioRiskEngine(weights=weights, returns=returns)
    var_95 = engine.compute_parametric_var(confidence_level=0.95)
    
    assert isinstance(var_95, float)
    assert var_95 > 0.0

def test_monte_carlo_var(sample_portfolio_data: tuple[pd.Series, pd.DataFrame]) -> None:
    """Validates Monte Carlo simulation VaR calculation."""
    weights, returns = sample_portfolio_data
    engine = PortfolioRiskEngine(weights=weights, returns=returns)
    mc_var = engine.compute_monte_carlo_var(confidence_level=0.99, simulations=1000)
    
    assert isinstance(mc_var, float)
    assert mc_var > 0.0

def test_stress_testing(sample_portfolio_data: tuple[pd.Series, pd.DataFrame]) -> None:
    """Validates tail-risk stress testing scenario application."""
    weights, returns = sample_portfolio_data
    engine = PortfolioRiskEngine(weights=weights, returns=returns)
    stressed_loss = engine.run_stress_test(shock_multiplier=2.5)
    
    assert isinstance(stressed_loss, float)
    assert stressed_loss > 0.0
