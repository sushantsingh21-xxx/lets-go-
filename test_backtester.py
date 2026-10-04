import unittest
from unittest.mock import patch

import pandas as pd

from backtester import download_price_data, run_risk_backtest


class BacktesterCostAndExportTests(unittest.TestCase):
    @staticmethod
    def _data_with_buy_then_stop(stop_low):
        return pd.DataFrame(
            [
                {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Buy": False, "Sell": False},
                {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Buy": True, "Sell": False},
                {"Open": 100.0, "High": 101.0, "Low": 99.5, "Close": 100.0, "Buy": False, "Sell": False},
                {"Open": 100.0, "High": 101.0, "Low": stop_low, "Close": 99.0, "Buy": False, "Sell": False},
            ],
            index=pd.date_range("2024-01-01", periods=4, freq="D"),
        )

    @patch("backtester.yf.download")
    def test_one_and_two_minute_intervals_are_supported(self, download):
        download.return_value = pd.DataFrame(
            {"Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.0]},
            index=pd.to_datetime(["2024-01-01 09:15"]),
        )

        for interval in ("1m", "2m"):
            with self.subTest(interval=interval):
                prices = download_price_data(interval=interval)
                self.assertEqual(len(prices), 1)
                self.assertEqual(download.call_args.kwargs["interval"], interval)

    def test_entry_fees_do_not_exceed_available_cash(self):
        initial_capital = 10000
        _, trades, _ = run_risk_backtest(
            self._data_with_buy_then_stop(stop_low=49.0),
            initial_capital=initial_capital,
            risk_per_trade_pct=100,
            stop_loss_pct=50,
            transaction_cost_pct=0.1,
            slippage_pct=0.1,
        )

        trade = trades.iloc[0]
        entry_spend = trade["Quantity"] * trade["Entry fill price"] + trade["Entry cost"]
        self.assertLessEqual(entry_spend, initial_capital + 1e-8)

    def test_stop_risk_budget_includes_costs_and_slippage(self):
        risk_budget = 100
        _, trades, _ = run_risk_backtest(
            self._data_with_buy_then_stop(stop_low=98.9),
            initial_capital=10000,
            risk_per_trade_pct=1,
            stop_loss_pct=1,
            transaction_cost_pct=0.1,
            slippage_pct=0.1,
        )

        trade = trades.iloc[0]
        self.assertAlmostEqual(trade["Risk amount"], risk_budget)
        self.assertGreaterEqual(trade["P&L"], -risk_budget - 1e-8)

    def test_target_preserves_reward_risk_ratio_after_costs(self):
        data = self._data_with_buy_then_stop(stop_low=99.1)
        data.loc[data.index[-1], "High"] = 110.0
        reward_risk_ratio = 2.5

        _, trades, _ = run_risk_backtest(
            data,
            initial_capital=10000,
            risk_per_trade_pct=1,
            stop_loss_pct=1,
            reward_risk_ratio=reward_risk_ratio,
            transaction_cost_pct=0.1,
            slippage_pct=0.1,
        )

        trade = trades.iloc[0]
        self.assertEqual(trade["Exit reason"], "Target")
        self.assertAlmostEqual(trade["R multiple"], reward_risk_ratio)

    def test_transaction_costs_reduce_trade_pnl(self):
        data = pd.DataFrame(
            [
                {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Buy": False, "Sell": False},
                {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Buy": True, "Sell": False},
                {"Open": 110.0, "High": 111.0, "Low": 109.0, "Close": 110.5, "Buy": False, "Sell": False},
                {"Open": 110.0, "High": 111.0, "Low": 109.0, "Close": 110.5, "Buy": False, "Sell": True},
            ],
            index=pd.to_datetime([
                "2024-01-01 09:15",
                "2024-01-02 09:15",
                "2024-01-03 09:15",
                "2024-01-04 09:15",
            ]),
        )

        _, trades, _ = run_risk_backtest(
            data,
            initial_capital=10000,
            risk_per_trade_pct=10,
            stop_loss_pct=5,
            reward_risk_ratio=2,
            transaction_cost_pct=0.1,
            slippage_pct=0.1,
        )

        self.assertTrue(not trades.empty)
        self.assertLess(float(trades.iloc[0]["P&L"]), 1000)

    def test_costs_and_slippage_parameters_are_optional(self):
        data = pd.DataFrame(
            [
                {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Buy": False, "Sell": False},
                {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Buy": True, "Sell": False},
                {"Open": 110.0, "High": 111.0, "Low": 109.0, "Close": 110.5, "Buy": False, "Sell": False},
                {"Open": 110.0, "High": 111.0, "Low": 109.0, "Close": 110.5, "Buy": False, "Sell": True},
            ],
            index=pd.to_datetime([
                "2024-01-01 09:15",
                "2024-01-02 09:15",
                "2024-01-03 09:15",
                "2024-01-04 09:15",
            ]),
        )

        final_balance, trades, _ = run_risk_backtest(
            data,
            initial_capital=10000,
            risk_per_trade_pct=10,
            stop_loss_pct=5,
            reward_risk_ratio=2,
        )

        self.assertGreater(final_balance, 0)
        self.assertTrue(isinstance(trades, pd.DataFrame))


if __name__ == "__main__":
    unittest.main()
