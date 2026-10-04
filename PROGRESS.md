# Project Progress

Last updated: 2026-10-04

## Goal
Turn the standalone Nifty 50 moving-average backtester into a Streamlit web app that is easy to use, then extend it to more tickers and strategies. Longer-term idea: accept plain-language strategy requests and use live market data. Do not claim yfinance is guaranteed real-time or add live trade execution in this phase.

## Completed
- Added `backtester.py`, which downloads daily prices with yfinance, calculates SMA crossovers, simulates full-capital buys/sells, reports returns and win rate, and plots price plus moving averages.
- Added `yfinance>=0.2.40` and `matplotlib>=3.8` to `requirements.txt`.
- Installed dependencies in workspace environment `C:\Users\susha\OneDrive\sushant\.venv` (Python 3.12.10).
- Made backtester functions configurable: `download_close_prices(ticker, start_date, end_date)`, `add_strategy_signals(prices, short_window, long_window)`, and `run_backtest(data, initial_capital)`.
- Validated the configurable strategy and portfolio calculations with synthetic prices; check passed.
- Previously ran standalone backtest successfully with live data through 2026-09-25.
- Replaced the welcome page in `app.py` with a Streamlit Nifty 50 backtest screen.
- Added automatic recalculation when controls change, with price downloads cached for 15 minutes.
- Added 5-minute, 15-minute, 30-minute, 1-hour, and daily intervals, plus SMA and EMA crossovers.
- Added timeframe history-limit validation and a dark-only project theme in `.streamlit/config.toml`.
- Added Open/High/Low/Close downloads and a risk-sized trade simulator with account-risk %, stop distance, reward-to-risk target, and next-bar-open entries.
- Added trade analysis for profitable/losing/breakeven counts, gross and average P&L, largest win/loss, profit factor, expectancy, and max drawdown.
- Reorganized controls on the main page, added symbol and numeric-minute choices, and capped charts at 900 display points while retaining all bars for calculations.
- Added an example position sizing panel with estimated units, stop, target, planned risk, and potential reward.
- Added 1-minute and 2-minute data intervals supported by the app's controls.
- Included entry/stop slippage and transaction costs in planned stop-risk sizing, and reserved entry fees when enforcing the cash limit.
- Added regression tests for minute intervals, fee-aware cash sizing, and stop-risk limits.
- Added CSV downloads for completed trades and the equity curve.
- Verified fast-average changes update the results view and switching between 5-minute and daily bars works in Streamlit AppTest.
- Verified numeric interval and risk control changes update the view, interval changes select a valid history window, and a stop-gap loss is shown larger than planned risk.
- Checked `app.py` and `backtester.py`; no diagnostics reported.
- Started the dark-themed Streamlit server at `http://localhost:8501` (current run).

## Current State
- `app.py` has a practical risk-managed prototype for a small set of Indian indices/stocks, configurable bars, SMA/EMA crossover, and risk analytics.
- `backtester.py` includes separate basic and risk-managed simulators; its standalone `main()` remains a daily SMA example.
- The app does not place live orders. Gap risk may exceed planned stop risk; taxes and market impact are not modeled.

## Next Steps
1. Add strategies and symbols in small independently testable steps.
2. Consider a constrained plain-language prompt that maps onto supported settings; never execute arbitrary generated code.
3. Evaluate a reliable, licensed market-data provider if genuine real-time data is required.

## Run Commands (PowerShell, workspace root)
```powershell
.\.venv\Scripts\python.exe -m pip install -r .\new-real-project-app\requirements.txt
Set-Location .\new-real-project-app
..\.venv\Scripts\python.exe -m streamlit run .\app.py
```
