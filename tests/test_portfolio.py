from rufus import portfolio


def test_buy_cost_and_sell_proceeds():
    assert portfolio.buy_cost(100.0, 5) == 500.0
    assert portfolio.sell_proceeds(110.0, 5) == 550.0


def test_realized_and_unrealized_pnl():
    assert portfolio.realized_pnl(100.0, 120.0, 5) == 100.0
    assert portfolio.realized_pnl(100.0, 90.0, 5) == -50.0
    assert portfolio.unrealized_pnl(100.0, 105.0, 5) == 25.0


def test_total_value_sums_cash_and_positions():
    assert portfolio.total_value(400.0, 600.0) == 1000.0


def test_snapshot_report_rolls_up():
    data = {
        "cash": 8000.0,
        "open": [
            {"entry_price": 100.0, "current_price": 120.0, "quantity": 10},
            {"entry_price": 200.0, "current_price": 150.0, "quantity": 5},
        ],
        "closed": [
            {"realized_pnl": 50.0},
        ],
    }
    report = portfolio.snapshot_report(data)
    assert report["positions_value"] == 120 * 10 + 150 * 5
    assert report["total_value"] == 8000.0 + 120 * 10 + 150 * 5
    assert report["realized_pnl"] == 50.0
    assert report["unrealized_pnl"] == (120 - 100) * 10 + (150 - 200) * 5
    assert report["open_positions"] == 2
    assert report["closed_positions"] == 1