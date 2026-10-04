# Install dependencies with: python -m pip install pandas yfinance matplotlib

from datetime import date, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


TICKER = "^NSEI"
START_DATE = "2021-01-01"
INITIAL_CAPITAL = 100_000.0
SHORT_WINDOW = 20
LONG_WINDOW = 50


def download_price_data(
    ticker: str = TICKER,
    start_date: date | str = START_DATE,
    end_date: date | str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLC price bars for a ticker and inclusive date range."""
    supported_intervals = {"1m", "2m", "5m", "15m", "30m", "1h", "1d"}
    if interval not in supported_intervals:
        raise ValueError(f"Unsupported interval: {interval}")

    start_text = pd.Timestamp(start_date).date().isoformat()
    last_day = date.today() if end_date is None else pd.Timestamp(end_date).date()
    # yfinance treats the end date as exclusive, so add one day to include it.
    end_text = (last_day + timedelta(days=1)).isoformat()
    history = yf.download(
        ticker,
        start=start_text,
        end=end_text,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )

    fields = ("Open", "High", "Low", "Close")
    if history.empty or not set(fields).issubset(history.columns.get_level_values(0)):
        raise RuntimeError(f"No complete OHLC data was returned for {ticker} at interval {interval}.")

    price_fields = {}
    for field in fields:
        field_data = history[field]
        # Recent yfinance versions may return one column per ticker.
        if isinstance(field_data, pd.DataFrame):
            field_data = field_data.iloc[:, 0]
        price_fields[field] = field_data

    prices = pd.DataFrame(price_fields, index=history.index).dropna().astype(float)
    if prices.empty:
        raise RuntimeError(f"No valid OHLC bars remain for {ticker} after removing missing data.")

    return prices


def download_close_prices(
    ticker: str = TICKER,
    start_date: date | str = START_DATE,
    end_date: date | str | None = None,
    interval: str = "1d",
) -> pd.Series:
    """Download close prices for callers that only need a single series."""
    return download_price_data(ticker, start_date, end_date, interval)["Close"]


def add_strategy_signals(
    prices: pd.Series,
    short_window: int = SHORT_WINDOW,
    long_window: int = LONG_WINDOW,
    average_type: str = "SMA",
) -> pd.DataFrame:
    """Calculate SMA or EMA values and mark their crossovers."""
    if short_window < 1 or long_window < 1 or short_window >= long_window:
        raise ValueError("Moving average periods must be positive, with the short period smaller than the long period.")
    if average_type not in {"SMA", "EMA"}:
        raise ValueError("Average type must be SMA or EMA.")

    data = prices.to_frame() if isinstance(prices, pd.Series) else prices.copy()
    if "Close" not in data.columns:
        raise ValueError("Price data must contain a Close column.")
    short_column = f"{average_type}_{short_window}"
    long_column = f"{average_type}_{long_window}"
    if average_type == "SMA":
        data[short_column] = data["Close"].rolling(window=short_window).mean()
        data[long_column] = data["Close"].rolling(window=long_window).mean()
    else:
        data[short_column] = data["Close"].ewm(span=short_window, adjust=False, min_periods=short_window).mean()
        data[long_column] = data["Close"].ewm(span=long_window, adjust=False, min_periods=long_window).mean()

    # A crossover occurs only when the averages switch sides from one day to
    # the next; simply remaining above or below does not create a new signal.
    data["Buy"] = (data[short_column] > data[long_column]) & (
        data[short_column].shift(1) <= data[long_column].shift(1)
    )
    data["Sell"] = (data[short_column] < data[long_column]) & (
        data[short_column].shift(1) >= data[long_column].shift(1)
    )

    return data


def run_backtest(
    data: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
) -> tuple[float, list[float], pd.Series]:
    """Invest all available cash on buys and sell the full holding on sells."""
    if initial_capital <= 0:
        raise ValueError("Starting capital must be greater than zero.")
    if data.empty:
        raise ValueError("Cannot run a backtest without price data.")

    cash = initial_capital
    shares = 0.0
    entry_price = None
    completed_trade_returns = []
    portfolio_values = []

    for _, row in data.iterrows():
        close = row["Close"]

        # Buy fractional shares with all cash when a buy crossover occurs.
        if row["Buy"] and cash > 0:
            shares = cash / close
            cash = 0.0
            entry_price = close

        # Close the entire position on a sell crossover and record its return.
        elif row["Sell"] and shares > 0:
            cash = shares * close
            completed_trade_returns.append((close / entry_price) - 1.0)
            shares = 0.0
            entry_price = None

        # An open position is valued at today's close; no forced final-day sale.
        portfolio_values.append(cash + shares * close)

    balance_series = pd.Series(portfolio_values, index=data.index, name="Portfolio")
    final_balance = float(balance_series.iloc[-1])
    return final_balance, completed_trade_returns, balance_series


def run_risk_backtest(
    data: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    risk_per_trade_pct: float = 1.0,
    stop_loss_pct: float = 1.0,
    reward_risk_ratio: float = 2.0,
    transaction_cost_pct: float = 0.0,
    slippage_pct: float = 0.0,
) -> tuple[float, pd.DataFrame, pd.Series]:
    """Backtest long trades sized by account risk with stop and target exits and a simple cost model."""
    required_columns = {"Open", "High", "Low", "Close", "Buy", "Sell"}
    if data.empty or not required_columns.issubset(data.columns):
        raise ValueError("Risk backtest requires non-empty OHLC data with Buy and Sell signals.")
    if initial_capital <= 0:
        raise ValueError("Starting capital must be greater than zero.")
    if not 0 < risk_per_trade_pct <= 100:
        raise ValueError("Risk per trade must be greater than 0% and at most 100%.")
    if not 0 < stop_loss_pct < 100:
        raise ValueError("Stop loss must be greater than 0% and less than 100%.")
    if reward_risk_ratio <= 0:
        raise ValueError("Reward-to-risk ratio must be greater than zero.")
    if not 0 <= transaction_cost_pct < 100:
        raise ValueError("Transaction cost percentage must be between 0% and 100%.")
    if not 0 <= slippage_pct < 100:
        raise ValueError("Slippage percentage must be between 0% and 100%.")

    cash = float(initial_capital)
    position = None
    completed_trades = []
    equity_values = []

    def close_position(timestamp, exit_price, exit_reason):
        nonlocal cash, position
        if position is None:
            return

        quantity = position["quantity"]
        entry_fill_price = position["entry_fill_price"]
        exit_fill_price = float(exit_price) * (1 - slippage_pct / 100)
        entry_cost = position["entry_cost"]
        exit_cost = abs(quantity * exit_fill_price) * (transaction_cost_pct / 100)
        profit_loss = (exit_fill_price - entry_fill_price) * quantity - entry_cost - exit_cost
        cash += quantity * exit_fill_price - exit_cost
        completed_trades.append(
            {
                "Entry time": position["entry_time"],
                "Exit time": timestamp,
                "Entry price": position["entry_price"],
                "Entry fill price": entry_fill_price,
                "Exit price": exit_price,
                "Exit fill price": exit_fill_price,
                "Quantity": quantity,
                "Risk amount": position["risk_amount"],
                "Entry cost": entry_cost,
                "Exit cost": exit_cost,
                "P&L": profit_loss,
                "Return %": (exit_fill_price / entry_fill_price - 1) * 100,
                "R multiple": profit_loss / position["risk_amount"],
                "Exit reason": exit_reason,
                "Outcome": "Win" if profit_loss > 0 else "Loss" if profit_loss < 0 else "Breakeven",
            }
        )
        position = None

    def find_bar_exit(bar, signal_exit):
        if bar["Open"] <= position["stop_price"]:
            return bar["Open"], "Stop loss (gap)"
        if bar["Open"] >= position["target_price"]:
            return bar["Open"], "Target (gap)"
        if signal_exit:
            return bar["Open"], "Opposite crossover"
        if bar["Low"] <= position["stop_price"]:
            return position["stop_price"], "Stop loss"
        if bar["High"] >= position["target_price"]:
            return position["target_price"], "Target"
        return None

    for row_number, (timestamp, bar) in enumerate(data.iterrows()):
        previous_buy = row_number > 0 and bool(data["Buy"].iloc[row_number - 1])
        previous_sell = row_number > 0 and bool(data["Sell"].iloc[row_number - 1])
        entered_this_bar = False

        if position is not None:
            exit_fill = find_bar_exit(bar, previous_sell)
            if exit_fill:
                close_position(timestamp, float(exit_fill[0]), exit_fill[1])

        if position is None and previous_buy and not previous_sell:
            entry_price = float(bar["Open"])
            if entry_price > 0 and cash > 0:
                stop_price = entry_price * (1 - stop_loss_pct / 100)
                risk_budget = cash * risk_per_trade_pct / 100
                slippage_rate = slippage_pct / 100
                entry_fill_price = entry_price * (1 + slippage_rate)
                stop_fill_price = stop_price * (1 - slippage_rate)
                cost_rate = transaction_cost_pct / 100
                risk_per_unit = (
                    entry_fill_price
                    - stop_fill_price
                    + (entry_fill_price + stop_fill_price) * cost_rate
                )
                cash_per_unit = entry_fill_price * (1 + cost_rate)
                quantity = min(risk_budget / risk_per_unit, cash / cash_per_unit)
                risk_amount = quantity * risk_per_unit
                target_price = (
                    risk_per_unit * reward_risk_ratio + entry_fill_price * (1 + cost_rate)
                ) / ((1 - slippage_rate) * (1 - cost_rate))
                entry_cost = quantity * entry_fill_price * cost_rate
                cash -= quantity * entry_fill_price + entry_cost
                position = {
                    "entry_time": timestamp,
                    "entry_price": entry_price,
                    "entry_fill_price": entry_fill_price,
                    "quantity": quantity,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "risk_amount": risk_amount,
                    "entry_cost": entry_cost,
                }
                entered_this_bar = True

        if position is not None and entered_this_bar:
            exit_fill = find_bar_exit(bar, False)
            if exit_fill:
                close_position(timestamp, float(exit_fill[0]), exit_fill[1])

        if position is not None and row_number == len(data) - 1:
            close_position(timestamp, float(bar["Close"]), "End of test")

        equity_values.append(cash + (position["quantity"] * float(bar["Close"]) if position else 0))

    trades = pd.DataFrame(completed_trades)
    equity_curve = pd.Series(equity_values, index=data.index, name="Portfolio")
    return float(equity_curve.iloc[-1]), trades, equity_curve


def summarize_trades(trades: pd.DataFrame) -> dict[str, float | int | None]:
    """Summarize completed trade outcomes in account currency."""
    if trades.empty:
        return {
            "total": 0,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "average_win": None,
            "average_loss": None,
            "largest_win": None,
            "largest_loss": None,
            "profit_factor": None,
        }

    profits = trades.loc[trades["P&L"] > 0, "P&L"]
    losses = trades.loc[trades["P&L"] < 0, "P&L"]
    gross_profit = float(profits.sum())
    gross_loss = float(-losses.sum())

    return {
        "total": len(trades),
        "wins": len(profits),
        "losses": len(losses),
        "breakeven": int((trades["P&L"] == 0).sum()),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "average_win": float(profits.mean()) if not profits.empty else None,
        "average_loss": float(losses.mean()) if not losses.empty else None,
        "largest_win": float(profits.max()) if not profits.empty else None,
        "largest_loss": float(losses.min()) if not losses.empty else None,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
    }


def main() -> None:
    prices = download_close_prices()
    data = add_strategy_signals(prices)
    final_balance, trade_returns, _ = run_backtest(data)

    total_return = (final_balance / INITIAL_CAPITAL - 1.0) * 100.0
    if trade_returns:
        win_rate = sum(trade_return > 0 for trade_return in trade_returns) / len(trade_returns) * 100.0
        win_rate_text = f"{win_rate:.2f}% ({len(trade_returns)} completed trades)"
    else:
        win_rate_text = f"N/A (0 completed trades)"

    print("Nifty 50 20/50 SMA Crossover Backtest")
    print(f"Period: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Total Returns: {total_return:.2f}%")
    print(f"Win Rate: {win_rate_text}")
    print(f"Final Portfolio Balance: ₹{final_balance:,.2f}")

    # Show the index close alongside both averages for a visual strategy check.
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Close"], label="Nifty 50 Close", linewidth=1.2)
    plt.plot(data.index, data["SMA_20"], label="20-day SMA", linewidth=1.0)
    plt.plot(data.index, data["SMA_50"], label="50-day SMA", linewidth=1.0)
    plt.title("Nifty 50: 20/50-Day SMA Crossover")
    plt.xlabel("Date")
    plt.ylabel("Index Level")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
