from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def calculate_metrics(
    history: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calculate historical NAV performance metrics only.
    No forecasting or machine-learning operations are performed.
    """

    group = (
        history.sort_values("date")
        .reset_index(drop=True)
        .copy()
    )

    if group.empty:
        raise ValueError("Cannot calculate metrics for an empty NAV history.")

    nav = group["nav"].astype(float)

    daily_returns = (
        nav.pct_change(fill_method=None)
        .dropna()
    )

    elapsed_days = max(
        (
            group["date"].iloc[-1]
            - group["date"].iloc[0]
        ).days,
        1,
    )

    total_return = float(
        (nav.iloc[-1] / nav.iloc[0] - 1.0) * 100.0
    )

    annualised_return = float(
        (
            (nav.iloc[-1] / nav.iloc[0])
            ** (365.25 / elapsed_days)
            - 1.0
        )
        * 100.0
    )

    annualised_volatility = (
        float(
            daily_returns.std()
            * np.sqrt(TRADING_DAYS_PER_YEAR)
            * 100.0
        )
        if len(daily_returns) > 1
        else None
    )

    drawdown = nav / nav.cummax() - 1.0

    def period_return(observations: int) -> float | None:
        if len(nav) <= observations:
            return None

        return float(
            (
                nav.iloc[-1]
                / nav.iloc[-1 - observations]
                - 1.0
            )
            * 100.0
        )

    return {
        "observations": int(len(group)),
        "start_date": pd.Timestamp(
            group["date"].iloc[0]
        ).date().isoformat(),
        "end_date": pd.Timestamp(
            group["date"].iloc[-1]
        ).date().isoformat(),
        "starting_nav": float(nav.iloc[0]),
        "latest_nav": float(nav.iloc[-1]),
        "total_return_pct": total_return,
        "annualised_return_pct": annualised_return,
        "annualised_volatility_pct": annualised_volatility,
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "return_30_observations_pct": period_return(30),
        "return_90_observations_pct": period_return(90),
        "return_252_observations_pct": period_return(252),
    }


def analyse_funds(
    funds: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Analyse uploaded funds using historical NAV data only.

    This function deliberately does not:
    - Build forecasting features
    - Train machine-learning models
    - Select or validate prediction models
    - Generate future NAV estimates

    Each fund is analysed independently.
    """

    results = []

    for item in funds:
        fund_name = item["fund_name"]

        history = (
            item["data"]
            .sort_values("date")
            .reset_index(drop=True)
            .copy()
        )

        if history.empty:
            raise ValueError(
                f"No valid NAV observations found for {fund_name}."
            )

        history["date"] = pd.to_datetime(
            history["date"],
            errors="raise",
        )
        history["nav"] = pd.to_numeric(
            history["nav"],
            errors="raise",
        )

        metrics = calculate_metrics(history)

        historical = [
            {
                "date": pd.Timestamp(date).date().isoformat(),
                "nav": float(nav),
            }
            for date, nav in zip(
                history["date"],
                history["nav"],
            )
        ]

        results.append(
            {
                "fund_name": fund_name,
                "source_file": item["source_file"],
                "historical": historical,
                "metrics": metrics,
                "warnings": list(item.get("warnings") or []),
                "cleaning": item.get("cleaning") or {},
            }
        )

    return results