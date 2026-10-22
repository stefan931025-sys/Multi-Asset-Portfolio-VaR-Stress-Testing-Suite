"""
Automated Unit Testing Suite (pytest)
Ensures mathematical correctness and exception handling for risk engines.
"""

import numpy as np
import pandas as pd
import pytest
from src.risk_engine import PortfolioRiskEngine


@pytest.fixture
def mock_returns_data():
    """Generates 250 days of synthetic return data for 3 assets."""
    np.random.seed(42)
    data = np.random.normal(0.0005, 0.015, size=(250, 3))
    return pd.DataFrame(data, columns=["SPY", "TLT", "UUP"])


def test_weight_validation_error(mock_returns_data):
    """Verifies that weights not summing to 1.0 raise ValueError."""
    invalid_weights = [0.5, 0.2, 0.1]  # Sums to 0.8
    with pytest.raises(ValueError):
        PortfolioRiskEngine(
            portfolio_value=1_000_000,
            weights=invalid_weights,
            returns_df=mock_returns_data,
        )


def test_parametric_var_calculation(mock_returns_data):
    """Validates Parametric VaR output structure and positivity."""
    engine = PortfolioRiskEngine(
        portfolio_value=10_000_000,
        weights=[0.5, 0.3, 0.2],
        returns_df=mock_returns_data,
        transaction_cost_bps=2.0,
    )
    result = engine.calculate_parametric_var(confidence_level=0.99, horizon_days=1)

    assert "Parametric_VaR_Dollar" in result
    assert result["Parametric_VaR_Dollar"] > 0
    assert result["Annualized_Portfolio_Vol"] > 0


def test_monte_carlo_reproducibility(mock_returns_data):
    """Ensures fixed seed yields identical Monte Carlo VaR output."""
    engine = PortfolioRiskEngine(
        portfolio_value=5_000_000,
        weights=[0.4, 0.4, 0.2],
        returns_df=mock_returns_data,
    )
    res1 = engine.calculate_monte_carlo_var(
        confidence_level=0.95, random_seed=42
    )
    res2 = engine.calculate_monte_carlo_var(
        confidence_level=0.95, random_seed=42
    )

    assert res1["MonteCarlo_VaR_Dollar"] == res2["MonteCarlo_VaR_Dollar"]
