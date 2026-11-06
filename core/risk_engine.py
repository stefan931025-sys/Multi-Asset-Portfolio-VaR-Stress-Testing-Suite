import numpy as np
import pandas as pd
from typing import List, Tuple

class MarketDataLoader:
    """Fetches and processes historical price data for portfolio assets."""
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def fetch_price_data(self) -> pd.DataFrame:
        import yfinance as yf
        data = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=True
        )
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]["Close" in data.columns] if "Close" in data else data
        else:
            prices = data["Close"] if "Close" in data else data
            
        prices = prices.ffill().dropna()
        if prices.empty:
            raise ValueError("No historical price data fetched. Verify tickers and date range.")
        return prices

    def compute_log_returns(self) -> pd.DataFrame:
        prices = self.fetch_price_data()
        log_returns = np.log(prices / prices.shift(1)).dropna()
        return log_returns

    def get_portfolio_covariance(self, annualized: bool = True) -> Tuple[pd.Series, pd.DataFrame]:
        returns = self.compute_log_returns()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        if annualized:
            mean_returns = mean_returns * 252
            cov_matrix = cov_matrix * 252
        return mean_returns, cov_matrix


class PortfolioRiskEngine:
    """Computes Parametric VaR, Monte Carlo VaR, and validates portfolio weights."""
    def __init__(self, portfolio_value: float, weights: List[float], returns_df: pd.DataFrame, transaction_cost_bps: float = 0.0):
        self.portfolio_value = portfolio_value
        self.weights = np.array(weights)
        self.returns_df = returns_df
        self.transaction_cost_bps = transaction_cost_bps
        
        # Validate weights sum to 1.0 (within floating point tolerance)
        if not np.isclose(np.sum(self.weights), 1.0, atol=1e-4):
            raise ValueError("Portfolio weights must sum to 1.0")

    def calculate_parametric_var(self, confidence_level: float = 0.99, horizon_days: int = 1) -> dict:
        from scipy.stats import norm
        mean_returns = self.returns_df.mean()
        cov_matrix = self.returns_df.cov()
        
        portfolio_mean = np.dot(self.weights, mean_returns)
        portfolio_variance = np.dot(self.weights.T, np.dot(cov_matrix, self.weights))
        portfolio_vol = np.sqrt(portfolio_variance)
        
        z_score = norm.ppf(confidence_level)
        var_pct = z_score * portfolio_vol * np.sqrt(horizon_days) - portfolio_mean * horizon_days
        var_dollar = self.portfolio_value * var_pct
        
        return {
            "Parametric_VaR_Dollar": float(var_dollar),
            "Parametric_VaR_Pct": float(var_pct),
            "Annualized_Portfolio_Vol": float(portfolio_vol * np.sqrt(252))
        }

    def calculate_monte_carlo_var(self, confidence_level: float = 0.95, horizon_days: int = 1, simulations: int = 10000, random_seed: int = None) -> dict:
        if random_seed is not None:
            np.random.seed(random_seed)
            
        mean_returns = self.returns_df.mean().values
        cov_matrix = self.returns_df.cov().values
        
        simulated_returns = np.random.multivariate_normal(mean_returns, cov_matrix, simulations)
        portfolio_sim_returns = np.dot(simulated_returns, self.weights) * np.sqrt(horizon_days)
        
        var_pct = np.percentile(portfolio_sim_returns, (1 - confidence_level) * 100)
        var_dollar = self.portfolio_value * abs(var_pct)
        
        return {
            "MonteCarlo_VaR_Dollar": float(var_dollar),
            "MonteCarlo_VaR_Pct": float(abs(var_pct))
        }
