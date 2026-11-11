import numpy as np
import pandas as pd

from app.core.forecast_engine import build_features, calculate_metrics, analyse_funds


def make_fund(name, phase=0.0, n=1600):
    dates = pd.bdate_range("2018-01-01", periods=n)
    t = np.arange(n)
    nav = 10 * np.exp(0.00025 * t + 0.025 * np.sin(t / 80 + phase))
    return {
        "fund_name": name,
        "source_file": f"{name}.csv",
        "data": pd.DataFrame({
            "date": dates,
            "fund_name": name,
            "nav": nav,
            "source_file": f"{name}.csv",
        }),
        "warnings": [],
        "cleaning": {"rows_received": n, "rows_retained": n},
    }


def test_feature_engineering_is_fund_scoped():
    funds = [make_fund("A", 0.0, 700), make_fund("B", 0.5, 700)]
    combined = pd.concat([x["data"] for x in funds], ignore_index=True)
    features = build_features(combined)
    assert set(features["fund_name"]) == {"A", "B"}
    assert features.groupby("fund_name")["future_return_1y"].count().min() > 0


def test_metrics_are_returned():
    fund = make_fund("A", n=600)
    metrics = calculate_metrics(fund["data"])
    assert metrics["observations"] == 600
    assert metrics["latest_nav"] > 0
    assert "max_drawdown_pct" in metrics


def test_analysis_returns_one_result_per_fund():
    results = analyse_funds([make_fund("A", 0.0, 1600), make_fund("B", 0.4, 1600)])
    assert len(results) == 2
    assert {x["fund_name"] for x in results} == {"A", "B"}
    assert all(len(x["historical"]) == 1600 for x in results)
    assert all(x["forecast"] for x in results)
