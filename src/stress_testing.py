"""
Historical Scenario & Stress-Testing Engine
Applies macro shock scenarios (e.g., 2008 Financial Crisis, 2020 COVID Shock)
to assess portfolio tail-risk under extreme illiquidity.
"""

from typing import Dict
import numpy as np
import pandas as pd


class MacroStressTester:
    """Simulates market factor stress scenarios on multi-asset portfolios."""

    PREDEFINED_SCENARIOS = {
        "2008_GFC_Shock": {"SPY": -0.088, "TLT": 0.024, "UUP": 0.012},
        "2020_COVID_Liquidity_Crunch": {"SPY": -0.119, "TLT": -0.018, "UUP": 0.015},
        "2022_Rates_Inversion_Shock": {"SPY": -0.035, "TLT": -0.042, "UUP": 0.008},
    }

    def __init__(self, portfolio_value: float, weights: Dict[str, float]):
        self.portfolio_value = portfolio_value
        self.weights = weights

    def run_scenario(self, scenario_name: str) -> Dict[str, float]:
        """Evaluates immediate dollar and percentage drawdowns under stress scenario."""
        if scenario_name not in self.PREDEFINED_SCENARIOS:
            raise KeyError(f"Scenario '{scenario_name}' not recognized.")

        shocks = self.PREDEFINED_SCENARIOS[scenario_name]
        total_pnl_pct = 0.0

        for ticker, weight in self.weights.items():
            asset_shock = shocks.get(ticker, -0.05)  # Fallback 5% shock if unlisted
            total_pnl_pct += weight * asset_shock

        stress_dollar_loss = abs(self.portfolio_value * total_pnl_pct)

        return {
            "Scenario": scenario_name,
            "Simulated_Portfolio_Return_Pct": round(total_pnl_pct * 100, 2),
            "Estimated_Dollar_Loss": round(stress_dollar_loss, 2),
        }
