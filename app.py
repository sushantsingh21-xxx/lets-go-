from datetime import date, timedelta
from math import ceil

import streamlit as st
import pandas as pd

from backtester import (
    INITIAL_CAPITAL,
    TICKER,
    add_strategy_signals,
    download_price_data,
    run_risk_backtest,
    summarize_trades,
)

st.set_page_config(page_title="Strategy Lab", page_icon="₹", layout="wide")


@st.cache_data(ttl=900, show_spinner=False)
def get_cached_prices(ticker: str, start: date, end: date, interval: str):
    return download_price_data(ticker, start, end, interval)


def downsample_chart_data(data, max_points: int = 900):
    if len(data) <= max_points:
        return data
    step = ceil(len(data) / max_points)
    sampled = data.iloc[::step]
    if sampled.index[-1] != data.index[-1]:
        sampled = pd.concat([sampled, data.iloc[[-1]]])
    return sampled


st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(145deg, #0b100e 0%, #111a16 58%, #0b100e 100%); }
    [data-testid="stHeader"] { background: rgba(11, 16, 14, 0.92); }
    .block-container { padding-top: 1.8rem; max-width: 1500px; }
    h1, h2, h3 { color: #edf3ee; }
    [data-testid="stMetric"] {
        background: #17211c;
        border: 1px solid #2a3a31;
        border-radius: 8px;
        padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] { color: #aebcb2; }
    [data-testid="stMetricValue"] { color: #d5efb4; }
    </style>
    """,
    unsafe_allow_html=True,
)

symbols = {
    "Nifty 50": TICKER,
    "Nifty Bank": "^NSEBANK",
    "Reliance Industries": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "TCS": "TCS.NS",
}
interval_codes = {1: "1m", 2: "2m", 5: "5m", 15: "15m", 30: "30m", 60: "1h", 1440: "1d"}
history_limits = {1: 7, 2: 60, 5: 60, 15: 60, 30: 60, 60: 730, 1440: None}

st.title("Strategy Lab")
st.caption("INDIAN MARKET BACKTESTING · HISTORICAL DATA · SIMULATED TRADES ONLY")

with st.container(border=True):
    st.subheader("Market and strategy")
    market_columns = st.columns(4)
    symbol_name = market_columns[0].selectbox("Market / symbol", list(symbols))
    interval_minutes = market_columns[1].number_input(
        "Bar interval (minutes)", min_value=1, max_value=1440, value=15, step=1
    )
    average_type = market_columns[2].selectbox("Crossover average", ["EMA", "SMA"])
    default_fast, default_slow = (9, 21) if average_type == "EMA" else (20, 50)
    short_window = market_columns[3].number_input(
        "Fast average (bars)", min_value=1, value=default_fast, step=1, key=f"fast_{average_type}"
    )

    strategy_columns = st.columns(4)
    long_window = strategy_columns[0].number_input(
        "Slow average (bars)", min_value=2, value=default_slow, step=1, key=f"slow_{average_type}"
    )
    default_days = 6 if interval_minutes == 1 else 55 if interval_minutes <= 30 else 700 if interval_minutes == 60 else 365
    if st.session_state.get("previous_interval_minutes") != interval_minutes:
        st.session_state["start_date"] = date.today() - timedelta(days=default_days)
        st.session_state["previous_interval_minutes"] = interval_minutes
    start_date = strategy_columns[1].date_input("From", key="start_date")
    end_date = strategy_columns[2].date_input("To", value=date.today())
    initial_capital = strategy_columns[3].number_input(
        "Starting capital (INR)", min_value=1_000, value=int(INITIAL_CAPITAL), step=10_000
    )

with st.container(border=True):
    st.subheader("Risk management")
    risk_columns = st.columns(5)
    risk_per_trade_pct = risk_columns[0].number_input(
        "Account risk per trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.25
    )
    stop_loss_pct = risk_columns[1].number_input(
        "Stop distance from entry (%)", min_value=0.1, max_value=20.0, value=1.0, step=0.25
    )
    reward_risk_ratio = risk_columns[2].number_input(
        "Reward-to-risk ratio", min_value=0.25, max_value=10.0, value=2.0, step=0.25
    )
    transaction_cost_pct = risk_columns[3].number_input(
        "Trading cost per side (%)", min_value=0.0, max_value=5.0, value=0.08, step=0.01
    )
    slippage_pct = risk_columns[4].number_input(
        "Slippage per side (%)", min_value=0.0, max_value=5.0, value=0.05, step=0.01
    )
    st.caption(
        "Position size is limited by both your risk budget and available cash. "
        "Signals execute at the next bar open; stop/target checks use each bar's high and low; "
        "costs and slippage are included in the simulation."
    )

interval = interval_codes.get(int(interval_minutes))
if interval is None:
    st.error("Choose a supported bar interval: 1, 2, 5, 15, 30, 60, or 1440 minutes (daily).")
elif start_date >= end_date:
    st.error("Choose an end date later than the start date.")
elif short_window >= long_window:
    st.error("Fast average length must be smaller than slow average length.")
elif history_limits[int(interval_minutes)] and (end_date - start_date).days + 1 > history_limits[int(interval_minutes)]:
    st.error(
        f"Yahoo Finance limits this interval to about {history_limits[int(interval_minutes)]} days of history. "
        "Shorten the date range and try again."
    )
else:
    ticker = symbols[symbol_name]
    try:
        with st.spinner(f"Loading {symbol_name} bars and calculating the backtest..."):
            prices = get_cached_prices(ticker, start_date, end_date, interval)
            if len(prices) <= long_window:
                raise ValueError("Not enough price bars for the selected slow average. Use a longer date range or shorter average.")
            strategy = add_strategy_signals(
                prices,
                short_window=int(short_window),
                long_window=int(long_window),
                average_type=average_type,
            )
            final_balance, trades, equity_curve = run_risk_backtest(
                strategy,
                initial_capital=float(initial_capital),
                risk_per_trade_pct=float(risk_per_trade_pct),
                stop_loss_pct=float(stop_loss_pct),
                reward_risk_ratio=float(reward_risk_ratio),
                transaction_cost_pct=float(transaction_cost_pct),
                slippage_pct=float(slippage_pct),
            )

        summary = summarize_trades(trades)
        total_return = (final_balance / initial_capital - 1) * 100
        drawdown = equity_curve / equity_curve.cummax() - 1
        max_drawdown = float(drawdown.min() * 100)

        interval_label = "1 day" if interval_minutes == 1440 else f"{interval_minutes} minute"
        st.subheader(f"{symbol_name} · {interval_label} bars · {average_type} {short_window}/{long_window}")
        st.caption(
            f"Account risk {risk_per_trade_pct:.2f}% · stop {stop_loss_pct:.2f}% · "
            f"target {reward_risk_ratio:.2f}R"
        )
        top_metrics = st.columns(4)
        top_metrics[0].metric("Net return", f"{total_return:+.2f}%")
        top_metrics[1].metric("Ending balance", f"₹{final_balance:,.2f}")
        top_metrics[2].metric("Max drawdown", f"{max_drawdown:.2f}%")
        top_metrics[3].metric("Completed trades", summary["total"])

        export_columns = st.columns(2)
        trade_csv = trades.to_csv(index=False).encode("utf-8")
        export_columns[0].download_button(
            "Download trade log CSV",
            data=trade_csv,
            file_name=f"{ticker.lower()}_trade_log.csv",
            mime="text/csv",
            disabled=trades.empty,
        )
        equity_csv = equity_curve.to_frame(name="Portfolio Value").reset_index().rename(columns={"index": "Date"}).to_csv(index=False).encode("utf-8")
        export_columns[1].download_button(
            "Download equity curve CSV",
            data=equity_csv,
            file_name=f"{ticker.lower()}_equity_curve.csv",
            mime="text/csv",
        )

        overview_tab, analysis_tab, trades_tab = st.tabs(["Overview", "Trade analysis", "Trade log"])
        with overview_tab:
            st.subheader("Price and strategy")
            fast_column = f"{average_type}_{short_window}"
            slow_column = f"{average_type}_{long_window}"
            chart = strategy[["Close", fast_column, slow_column]]
            st.line_chart(downsample_chart_data(chart), height=390)
            st.caption(
                f"Showing up to 900 chart points; all {len(strategy):,} bars are used in calculations. "
                "Trading costs and slippage are included in this model."
            )
            st.subheader("Portfolio value")
            st.line_chart(downsample_chart_data(equity_curve.to_frame()), height=240)

        with analysis_tab:
            st.subheader("Trade outcomes")
            outcome_columns = st.columns(5)
            outcome_columns[0].metric("Profitable trades", f"{summary['wins']} / {summary['total']}")
            outcome_columns[1].metric("Gross profit", f"₹{summary['gross_profit']:,.2f}")
            outcome_columns[2].metric("Losing trades", f"{summary['losses']} / {summary['total']}")
            outcome_columns[3].metric("Gross losses", f"-₹{summary['gross_loss']:,.2f}")
            outcome_columns[4].metric("Breakeven trades", summary["breakeven"])

            detail_columns = st.columns(4)
            detail_columns[0].metric(
                "Average winning trade",
                f"₹{summary['average_win']:,.2f}" if summary["average_win"] is not None else "N/A",
            )
            detail_columns[1].metric(
                "Largest winner",
                f"₹{summary['largest_win']:,.2f}" if summary["largest_win"] is not None else "N/A",
            )
            detail_columns[2].metric(
                "Average losing trade",
                f"-₹{abs(summary['average_loss']):,.2f}" if summary["average_loss"] is not None else "N/A",
            )
            detail_columns[3].metric(
                "Largest loss",
                f"-₹{abs(summary['largest_loss']):,.2f}" if summary["largest_loss"] is not None else "N/A",
            )

            extra_columns = st.columns(3)
            extra_columns[0].metric("Win rate", f"{summary['wins'] / summary['total'] * 100:.1f}%" if summary["total"] else "N/A")
            extra_columns[1].metric("Profit factor", f"{summary['profit_factor']:.2f}" if summary["profit_factor"] is not None else "N/A")
            expectancy = (summary["gross_profit"] - summary["gross_loss"]) / summary["total"] if summary["total"] else None
            extra_columns[2].metric("Average P&L / trade", f"₹{expectancy:,.2f}" if expectancy is not None else "N/A")

            latest_price = float(prices["Close"].iloc[-1])
            risk_budget = float(initial_capital) * float(risk_per_trade_pct) / 100
            stop_distance = latest_price * float(stop_loss_pct) / 100
            example_quantity = min(risk_budget / stop_distance, float(initial_capital) / latest_price)
            example_stop = latest_price - stop_distance
            example_target = latest_price + stop_distance * float(reward_risk_ratio)
            planned_risk = example_quantity * stop_distance
            st.subheader("Example position at latest close")
            st.caption("Illustrative cash-only sizing; this is not a live recommendation or order.")
            sizing_columns = st.columns(5)
            sizing_columns[0].metric("Estimated units", f"{example_quantity:.2f}")
            sizing_columns[1].metric("Approx. stop", f"₹{example_stop:,.2f}")
            sizing_columns[2].metric("Approx. target", f"₹{example_target:,.2f}")
            sizing_columns[3].metric("Planned risk", f"₹{planned_risk:,.2f}")
            sizing_columns[4].metric(
                "Potential reward",
                f"₹{planned_risk * float(reward_risk_ratio):,.2f}",
            )

        with trades_tab:
            st.subheader("Completed trade details")
            if trades.empty:
                st.info("No completed trades occurred with these settings and dates.")
            else:
                st.dataframe(trades, width="stretch", hide_index=True)

        st.caption(
            "Long-only simulation. Gaps through a stop can lose more than planned; if a bar touches both "
            "stop and target, the simulator assumes the stop was hit first."
        )
    except Exception as error:
        st.error(f"Backtest could not be completed: {error}")
