import numpy as np
import pandas as pd

def simulate_portfolio_var(
    portfolio_value: float,
    weights: list,
    annual_returns: list,
    annual_volatilities: list,
    confidence_level: float = 0.95,
    days: int = 1
):
    """
    Computes parametric and historical Value-at-Risk (VaR) for a multi-asset portfolio.
    """
    weights = np.array(weights)
    returns = np.array(annual_returns)
    volatilities = np.array(annual_volatilities)
    
    # Assume a simplified correlation matrix (e.g., moderate cross-asset correlation)
    n = len(weights)
    corr_matrix = np.eye(n) + 0.3 * (np.ones((n, n)) - np.eye(n))
    
    # Covariance matrix calculation (Annualized)
    cov_matrix = np.outer(volatilities, volatilities) * corr_matrix
    
    # Portfolio expected return and variance
    port_return = np.dot(weights, returns)
    port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    port_volatility = np.sqrt(port_variance)
    
    # Scale volatility to the desired time horizon (e.g., 1-day VaR)
    time_scaling = np.sqrt(days / 252)
    period_volatility = port_volatility * time_scaling
    
    # Z-score for the given confidence level (e.g., 1.645 for 95%, 2.33 for 99%)
    from scipy.stats import norm
    z_score = norm.ppf(confidence_level)
    
    # Value-at-Risk calculations
    parametric_var_pct = (z_score * period_volatility) - (port_return * (days / 252))
    parametric_var_dollar = portfolio_value * parametric_var_pct
    
    # Monte Carlo Portfolio Simulation (10,000 trials)
    np.random.seed(42)
    simulated_returns = np.random.normal(port_return * (days / 252), period_volatility, 10000)
    historical_var_pct = np.percentile(simulated_returns, (1 - confidence_level) * 100)
    historical_var_dollar = portfolio_value * abs(historical_var_pct)
    
    results = {
        "Portfolio Value": f"${portfolio_value:,.2f}",
        "Confidence Level": f"{confidence_level * 100}%",
        "Time Horizon": f"{days} Day(s)",
        "Parametric VaR ($)": f"${parametric_var_dollar:,.2f}",
        "Monte Carlo VaR ($)": f"${historical_var_dollar:,.2f}",
        "Portfolio Annual Volatility": f"{port_volatility * 100:.2f}%"
    }
    
    return results

# --- EXECUTION ---
if __name__ == "__main__":
    print("--- MULTI-ASSET PORTFOLIO VaR ANALYSIS ---")
    
    # Example: Portfolio split across Equities (Asset 1), FX/Commodities (Asset 2), Fixed Income (Asset 3)
    portfolio_metrics = simulate_portfolio_var(
        portfolio_value=10_000_000, # $10 Million Portfolio
        weights=[0.50, 0.30, 0.20],   # 50% Equities, 30% FX, 20% Bonds
        annual_returns=[0.10, 0.04, 0.06],
        annual_volatilities=[0.18, 0.12, 0.07],
        confidence_level=0.99,      # 99% Confidence Level
        days=1                      # 1-day horizon
    )
    
    for k, v in portfolio_metrics.items():
        print(f"{k}: {v}")
