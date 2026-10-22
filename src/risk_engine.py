"""
Data Ingestion Module
Handles real historical market data ingestion, return matrix calculation, 
and missing data cleaning for multi-asset portfolios using Yahoo Finance.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
import yfinance as yf


class MarketDataLoader:
    """Fetches and processes historical price data for portfolio assets."""

    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def fetch_price_data(self) -> pd.DataFrame:
        """Downloads daily adjusted close prices for given tickers."""
        data = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=True,
        )

        # Handle single ticker vs multi-ticker DataFrame structure
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"] if "Close" in data else data
        else:
            prices = data[["Close"]] if "Close" in data else data

        # Forward fill missing business days, drop unresolvable NaNs
        prices = prices.ffill().dropna()

        if prices.empty:
            raise ValueError(
                "No historical price data fetched. Verify tickers and date range."
            )

        return prices

    def compute_log_returns(self) -> pd.DataFrame:
        """Computes continuous log returns from historical close prices."""
        prices = self.fetch_price_data()
        log_returns = np.log(prices / prices.shift(1)).dropna()
        return log_returns

    def get_portfolio_covariance(
        self, annualized: bool = True
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Calculates mean returns and covariance matrix (annualized by default)."""
        returns = self.compute_log_returns()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        if annualized:
            mean_returns = mean_returns * 252
            cov_matrix = cov_matrix * 252

        return mean_returns, cov_matrix
