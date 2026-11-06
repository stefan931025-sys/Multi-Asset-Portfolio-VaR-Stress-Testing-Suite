from core.data_loader import MarketDataLoader
from core.risk_engine import PortfolioRiskEngine
from core.stress_testing import MacroStressTester

if __name__ == "__main__":
    print("--- INITIATING INSTITUTIONAL RISK ENGINE ---")

    # Assets: Equities (SPY), Fixed Income (TLT), USD Index (UUP)
    tickers = ["SPY", "TLT", "UUP"]
    weights = [0.50, 0.30, 0.20]
    portfolio_value = 10_000_000  # $10M Portfolio

    # 1. Fetch real historical market data
    loader = MarketDataLoader()
    loader.tickers = tickers
    loader.start_date = "2022-01-01"
    loader.end_date = "2026-01-01"
    
    returns_df = loader.compute_log_returns()

    # 2. Compute Multi-Asset Risk Metrics
    risk_engine = PortfolioRiskEngine(
        portfolio_value=portfolio_value,
        weights=weights,
        returns_df=returns_df,
        transaction_cost_bps=2.5  # 2.5 bps slippage penalty
    )

    p_var = risk_engine.calculate_parametric_var(confidence_level=0.99)
    h_var = risk_engine.calculate_historical_var(confidence_level=0.99)
    mc_var = risk_engine.calculate_monte_carlo_var(
        confidence_level=0.99, simulations=50_000
    )

    print("\nRisk Metrics - 99% 1-Day Horizon")
    print(f"Parametric VaR ($): ${p_var['Parametric_VaR_Dollar']:,.2f}")
    print(f"Historical VaR ($): ${h_var['Historical_VaR_Dollar']:,.2f}")
    print(f"Monte Carlo VaR ($): ${mc_var['MonteCarlo_VaR_Dollar']:,.2f}")

    # 3. Macro Stress Scenario Evaluation
    weight_dict = dict(zip(tickers, weights))
    stress_tester = MacroStressTester(
        portfolio_value=portfolio_value, weights=weight_dict
    )
    
    stress_result = stress_tester.run_scenario("2020_COVID_Liquidity_Crunch")

    print("\nMacro Stress Scenario: 2020 COVID Liquidity Crunch")
    print("-" * 50)
    print(f"Simulated Portfolio Return: {stress_result['Simulated_Portfolio_Return_Pct']}%")
    print(f"Estimated Dollar Loss: ${stress_result['Estimated_Dollar_Loss']:,.2f}")
