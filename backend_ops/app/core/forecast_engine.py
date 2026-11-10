from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TRADING_DAYS_PER_YEAR = 252
MIN_FORECAST_OBSERVATIONS = 273
RANDOM_STATE = 42

MIN_TRAINING_ROWS = 400
MIN_VALIDATION_ROWS = 60
MIN_TEST_ROWS = 60

MODEL_FEATURES = [
    "momentum_30d_pct",
    "momentum_90d_pct",
    "momentum_180d_pct",
    "momentum_1y_pct",
    "momentum_acceleration_pct",
    "momentum_90d_vs_1y_pct",
    "nav_vs_ma_30d_pct",
    "nav_vs_ma_90d_pct",
    "nav_vs_ma_180d_pct",
    "nav_vs_ma_1y_pct",
    "ma_30d_slope_pct",
    "ma_90d_slope_pct",
    "ma_180d_slope_pct",
    "ma_1y_slope_pct",
    "volatility_30d_pct",
    "volatility_90d_pct",
    "volatility_1y_pct",
    "volatility_regime",
    "drawdown_pct",
    "drawdown_recovery_90d_pct",
    "positive_day_ratio_90d",
    "positive_day_ratio_1y",
    "avg_daily_return_30d_pct",
    "avg_daily_return_90d_pct",
    "relative_to_fund_median_pct",
    "trend_score",
]


@dataclass
class ModelBundle:
    model: Any | None
    model_name: str
    validation_mae_pct: float | None
    validation_rmse_pct: float | None
    validation_rows: int
    residual_q10: float
    residual_q90: float
    training_rows: int
    status: str

    # Metrics from the separate, untouched holdout test period.
    test_mae_pct: float | None = None
    test_rmse_pct: float | None = None
    test_rows: int = 0


def _pct_change(series: pd.Series, periods: int) -> pd.Series:
    return series.pct_change(periods=periods, fill_method=None) * 100.0


def build_features(all_funds: pd.DataFrame) -> pd.DataFrame:
    """
    Build time-series features independently for each fund.

    Features use current and historical NAV only.
    No future NAV values are used in feature calculations.

    This implementation supports independent fund analysis.
    The relative-to-median feature is zero because funds are not
    compared with one another.
    """

    pieces = []

    for fund_name, original in all_funds.groupby(
        "fund_name",
        sort=False,
    ):
        group = (
            original.sort_values("date")
            .copy()
            .reset_index(drop=True)
        )

        nav = group["nav"].astype(float)
        daily_return = nav.pct_change(fill_method=None)

        group["daily_return"] = daily_return

        # Momentum features.
        for window, label in [
            (30, "30d"),
            (90, "90d"),
            (180, "180d"),
            (252, "1y"),
        ]:
            group[f"momentum_{label}_pct"] = _pct_change(
                nav,
                window,
            )

        group["momentum_acceleration_pct"] = (
            group["momentum_30d_pct"]
            - group["momentum_90d_pct"] / 3.0
        )

        group["momentum_90d_vs_1y_pct"] = (
            group["momentum_90d_pct"]
            - group["momentum_1y_pct"] / 4.0
        )

        # Moving averages and moving-average slopes.
        for window, label in [
            (30, "30d"),
            (90, "90d"),
            (180, "180d"),
            (252, "1y"),
        ]:
            moving_average = nav.rolling(
                window,
                min_periods=window,
            ).mean()

            group[f"nav_vs_ma_{label}_pct"] = (
                nav / moving_average - 1.0
            ) * 100.0

            group[f"ma_{label}_slope_pct"] = (
                moving_average
                / moving_average.shift(21)
                - 1.0
            ) * 100.0

        # Annualised historical volatility.
        for window, label in [
            (30, "30d"),
            (90, "90d"),
            (252, "1y"),
        ]:
            group[f"volatility_{label}_pct"] = (
                daily_return.rolling(
                    window,
                    min_periods=window,
                ).std()
                * np.sqrt(TRADING_DAYS_PER_YEAR)
                * 100.0
            )

        group["volatility_regime"] = (
            group["volatility_30d_pct"]
            / group["volatility_1y_pct"].replace(0, np.nan)
        )

        # Drawdown features.
        running_peak = nav.cummax()

        group["drawdown_pct"] = (
            nav / running_peak - 1.0
        ) * 100.0

        group["drawdown_recovery_90d_pct"] = (
            group["drawdown_pct"]
            - group["drawdown_pct"].shift(90)
        )

        # Positive-return ratios.
        group["positive_day_ratio_90d"] = (
            daily_return.gt(0)
            .rolling(90, min_periods=90)
            .mean()
        )

        group["positive_day_ratio_1y"] = (
            daily_return.gt(0)
            .rolling(252, min_periods=252)
            .mean()
        )

        # Average daily returns.
        group["avg_daily_return_30d_pct"] = (
            daily_return.rolling(
                30,
                min_periods=30,
            ).mean()
            * 100.0
        )

        group["avg_daily_return_90d_pct"] = (
            daily_return.rolling(
                90,
                min_periods=90,
            ).mean()
            * 100.0
        )

        # Trend score: descriptive indicator, not investment advice.
        trend_signals = pd.concat(
            [
                group["momentum_90d_pct"].gt(0).astype(float),
                group["momentum_1y_pct"].gt(0).astype(float),
                group["nav_vs_ma_90d_pct"].gt(0).astype(float),
                group["nav_vs_ma_1y_pct"].gt(0).astype(float),
            ],
            axis=1,
        )

        group["trend_score"] = (
            trend_signals.mean(axis=1) * 100.0
        )

        # Normalisation is retained for consistency.
        # It is not used to compare different funds.
        first_nav = float(nav.iloc[0])

        group["normalized_nav"] = (
            nav / first_nav * 100.0
        )

        group["relative_to_fund_median_pct"] = 0.0
        group["fund_name"] = fund_name

        pieces.append(group)

    if not pieces:
        return pd.DataFrame()

    features = pd.concat(
        pieces,
        ignore_index=True,
    )

    features = features.sort_values(
        ["fund_name", "date"]
    ).reset_index(drop=True)

    # The target is the one-year forward return.
    # Shift is performed within each fund.
    features["future_return_1y"] = (
        features.groupby("fund_name")["nav"].shift(
            -TRADING_DAYS_PER_YEAR
        )
        / features["nav"]
        - 1.0
    )

    features["target_date"] = (
        features.groupby("fund_name")["date"].shift(
            -TRADING_DAYS_PER_YEAR
        )
    )

    return features


def _make_models() -> dict[str, Any]:
    """
    Return independent candidate models.

    Each candidate includes its own preprocessing pipeline.
    """

    return {
        "Ridge": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    Ridge(alpha=10.0),
                ),
            ]
        ),

        "Random Forest": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=10,
                        min_samples_leaf=10,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "Gradient Boosting": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=250,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def _calculate_errors(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float]:
    """
    Calculate MAE and RMSE.

    Inputs and outputs are decimal returns.
    Returned metrics are percentages.
    """

    residuals = actual - predicted

    mae = float(
        np.mean(np.abs(residuals)) * 100.0
    )

    rmse = float(
        np.sqrt(np.mean(np.square(residuals))) * 100.0
    )

    return mae, rmse


def train_fund_model(features: pd.DataFrame) -> ModelBundle:
    """
    Train and evaluate a model for ONE fund only.

    Chronological process:

    1. Training period:
       Model fitting.

    2. Validation period:
       Candidate model selection using MAE.

    3. Holdout test period:
       Independent evaluation and residual estimation.

    4. Final model:
       Refit the selected candidate using all available
       supervised historical observations.

    Purging:
       Training target dates must be before validation start.
       Validation target dates must be before test start.
    """

    supervised = features.dropna(
        subset=MODEL_FEATURES + [
            "future_return_1y",
            "target_date",
        ]
    ).copy()

    if supervised.empty:
        return ModelBundle(
            None,
            "Historical trend fallback",
            None,
            None,
            0,
            -0.15,
            0.15,
            0,
            "insufficient_training_history",
        )

    supervised = supervised.sort_values(
        "date"
    ).reset_index(drop=True)

    unique_dates = np.array(
        sorted(supervised["date"].unique())
    )

    if len(unique_dates) < 500:
        return ModelBundle(
            None,
            "Historical trend fallback",
            None,
            None,
            0,
            -0.15,
            0.15,
            len(supervised),
            "insufficient_training_history",
        )

    # Chronological cutoffs:
    # 60% for training period boundary.
    # 75% for validation period boundary.
    train_end_index = int(
        len(unique_dates) * 0.55
    )

    validation_end_index = int(
        len(unique_dates) * 0.80
    )

    train_end = pd.Timestamp(
        unique_dates[train_end_index]
    )

    validation_end = pd.Timestamp(
        unique_dates[validation_end_index]
    )

    latest_target_date = pd.Timestamp(
        supervised["target_date"].max()
    )

    # Training observations must have labels entirely
    # before the validation period.
    train = supervised[
        supervised["target_date"] < train_end
    ].copy()

    # Validation observations must:
    # - Start at or after the training cutoff.
    # - Have their complete one-year target before test starts.
    validation = supervised[
        (supervised["date"] >= train_end)
        & (
            supervised["target_date"]
            < validation_end
        )
    ].copy()

    # Test observations must:
    # - Start at or after the validation cutoff.
    # - Have their one-year target available by the latest date.
    test = supervised[
        (supervised["date"] >= validation_end)
        & (
            supervised["target_date"]
            <= latest_target_date
        )
    ].copy()

    if (
        len(train) < MIN_TRAINING_ROWS
        or len(validation) < MIN_VALIDATION_ROWS
        or len(test) < MIN_TEST_ROWS
    ):
        return ModelBundle(
            None,
            "Historical trend fallback",
            None,
            None,
            len(validation),
            -0.15,
            0.15,
            len(train),
            "insufficient_temporal_splits",
        )

    X_train = train[MODEL_FEATURES]
    y_train = train["future_return_1y"].astype(float)

    X_validation = validation[MODEL_FEATURES]
    y_validation = validation[
        "future_return_1y"
    ].astype(float)

    X_test = test[MODEL_FEATURES]
    y_test = test["future_return_1y"].astype(float)

    # --------------------------------------------------
    # Stage 1: Select candidate model using validation MAE.
    # --------------------------------------------------

    best_name = None
    best_validation_mae = float("inf")
    best_validation_rmse = None

    for name, candidate in _make_models().items():

        candidate.fit(
            X_train,
            y_train,
        )

        predictions = candidate.predict(
            X_validation
        )

        mae, rmse = _calculate_errors(
            y_validation.to_numpy(),
            predictions,
        )

        if mae < best_validation_mae:
            best_name = name
            best_validation_mae = mae
            best_validation_rmse = rmse

    # --------------------------------------------------
    # Stage 2: Evaluate the selected model on the
    # untouched holdout test period.
    # --------------------------------------------------

    train_and_validation = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    evaluation_model = _make_models()[best_name]

    evaluation_model.fit(
        train_and_validation[MODEL_FEATURES],
        train_and_validation[
            "future_return_1y"
        ].astype(float),
    )

    test_predictions = evaluation_model.predict(
        X_test
    )

    test_actual = y_test.to_numpy()

    test_residuals = (
        test_actual - test_predictions
    )

    test_mae, test_rmse = _calculate_errors(
        test_actual,
        test_predictions,
    )

    # Residual quantiles from the untouched test period.
    q10, q90 = np.quantile(
        test_residuals,
        [0.10, 0.90],
    )

    # --------------------------------------------------
    # Stage 3: Refit the selected model using all
    # supervised observations for the current forecast.
    # --------------------------------------------------

    final_model = _make_models()[best_name]

    final_model.fit(
        supervised[MODEL_FEATURES],
        supervised[
            "future_return_1y"
        ].astype(float),
    )

    return ModelBundle(
        model=final_model,
        model_name=best_name,
        validation_mae_pct=best_validation_mae,
        validation_rmse_pct=best_validation_rmse,
        validation_rows=len(validation),
        residual_q10=float(q10),
        residual_q90=float(q90),
        training_rows=len(supervised),
        status="trained",
        test_mae_pct=test_mae,
        test_rmse_pct=test_rmse,
        test_rows=len(test),
    )


def _ordered_return_bounds(
    expected_return: float,
    lower_return: float,
    upper_return: float,
) -> tuple[float, float, float]:
    """
    Ensure the forecast range is mathematically ordered.

    All inputs are decimal returns.
    """

    expected_return = max(
        float(expected_return),
        -0.99,
    )

    lower_return = max(
        float(lower_return),
        -0.99,
    )

    upper_return = max(
        float(upper_return),
        -0.99,
    )

    lower_return = min(
        lower_return,
        expected_return,
    )

    upper_return = max(
        upper_return,
        expected_return,
    )

    return (
        expected_return,
        lower_return,
        upper_return,
    )


def _fallback_return(
    latest: pd.Series,
    fund_history: pd.DataFrame,
) -> tuple[float, float, float]:
    """
    Trend-based fallback when a validated ML model
    cannot be trained.

    This is an estimate, not a validated prediction.
    """

    candidates = []

    periods = {
        "momentum_1y_pct": 1.0,
        "momentum_180d_pct": (
            TRADING_DAYS_PER_YEAR / 180.0
        ),
        "momentum_90d_pct": (
            TRADING_DAYS_PER_YEAR / 90.0
        ),
    }

    weights = {
        "momentum_1y_pct": 0.65,
        "momentum_180d_pct": 0.25,
        "momentum_90d_pct": 0.10,
    }

    for column, exponent in periods.items():

        value = latest.get(column)

        if pd.notna(value):

            period_return = float(value) / 100.0

            annualised = (
                (1.0 + period_return) ** exponent
                - 1.0
            )

            candidates.append(
                (
                    annualised,
                    weights[column],
                )
            )

    if not candidates:
        return 0.0, -0.20, 0.20

    weight_sum = sum(
        weight for _, weight in candidates
    )

    predicted = sum(
        value * weight
        for value, weight in candidates
    ) / weight_sum

    # Historical rolling one-year returns.
    rolling_annual = (
        fund_history["nav"]
        / fund_history["nav"].shift(
            TRADING_DAYS_PER_YEAR
        )
        - 1.0
    ).dropna()

    if len(rolling_annual) >= 30:

        low, high = np.quantile(
            rolling_annual,
            [0.10, 0.90],
        )

    else:

        volatility = float(
            fund_history["nav"]
            .pct_change(fill_method=None)
            .std()
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        )

        if not np.isfinite(volatility):
            volatility = 0.20

        low = predicted - volatility
        high = predicted + volatility

    return _ordered_return_bounds(
        predicted,
        float(low),
        float(high),
    )


def _monthly_path(
    last_date: pd.Timestamp,
    last_nav: float,
    expected_return: float,
    lower_return: float,
    upper_return: float,
) -> list[dict[str, Any]]:
    """
    Generate 12 monthly display points.

    The model predicts a 12-month endpoint.
    Intermediate monthly values are geometrically interpolated.
    They are not independent monthly predictions.
    """

    (
        expected_return,
        lower_return,
        upper_return,
    ) = _ordered_return_bounds(
        expected_return,
        lower_return,
        upper_return,
    )

    points = []

    for month in range(1, 13):

        fraction = month / 12.0

        forecast_date = (
            last_date + pd.DateOffset(months=month)
        )

        expected_nav = (
            last_nav
            * ((1.0 + expected_return) ** fraction)
        )

        lower_nav = (
            last_nav
            * ((1.0 + lower_return) ** fraction)
        )

        upper_nav = (
            last_nav
            * ((1.0 + upper_return) ** fraction)
        )

        points.append(
            {
                "date": forecast_date.date().isoformat(),
                "nav": float(expected_nav),
                "lower_nav": float(lower_nav),
                "upper_nav": float(upper_nav),
            }
        )

    return points


def calculate_metrics(
    history: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calculate historical NAV performance metrics.
    """

    group = history.sort_values(
        "date"
    ).reset_index(drop=True)

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
        (nav.iloc[-1] / nav.iloc[0] - 1.0)
        * 100.0
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

    drawdown = (
        nav / nav.cummax() - 1.0
    )

    def period_return(
        observations: int,
    ) -> float | None:

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
        "start_date": (
            group["date"].iloc[0].date().isoformat()
        ),
        "end_date": (
            group["date"].iloc[-1].date().isoformat()
        ),
        "starting_nav": float(nav.iloc[0]),
        "latest_nav": float(nav.iloc[-1]),
        "total_return_pct": total_return,
        "annualised_return_pct": annualised_return,
        "annualised_volatility_pct": annualised_volatility,
        "max_drawdown_pct": float(
            drawdown.min() * 100.0
        ),
        "return_30_observations_pct": period_return(30),
        "return_90_observations_pct": period_return(90),
        "return_252_observations_pct": period_return(252),
    }


def analyse_funds(
    funds: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Analyse multiple independently uploaded fund files.

    IMPORTANT:
    Every fund gets its own:
    - Feature engineering
    - Model training
    - Validation
    - Holdout testing
    - Forecast

    No fund's NAV history is used to train another fund's model.
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

        # Build features for this fund alone.
        fund_features = build_features(
            history
        )

        # Train a separate model for this fund.
        bundle = train_fund_model(
            fund_features
        )

        latest_feature = fund_features.iloc[-1]

        last_date = pd.Timestamp(
            history["date"].iloc[-1]
        )

        last_nav = float(
            history["nav"].iloc[-1]
        )

        forecast = []

        forecast_status = "insufficient_history"

        forecast_return = None
        lower_return = None
        upper_return = None

        model_name = bundle.model_name

        # A 273-observation minimum allows the one-year
        # moving-average slope feature to become available.
        if len(history) >= MIN_FORECAST_OBSERVATIONS:

            if bundle.model is not None:

                latest_features = pd.DataFrame(
                    [
                        latest_feature[
                            MODEL_FEATURES
                        ].to_dict()
                    ],
                    columns=MODEL_FEATURES,
                )

                raw_prediction = float(
                    bundle.model.predict(
                        latest_features
                    )[0]
                )

                # Limit extreme predictions to the observed
                # historical one-year target-return range.
                training_returns = (
                    fund_features[
                        "future_return_1y"
                    ].dropna()
                )

                if len(training_returns):

                    q01, q99 = training_returns.quantile(
                        [0.01, 0.99]
                    )

                    forecast_return = float(
                        np.clip(
                            raw_prediction,
                            q01,
                            q99,
                        )
                    )

                else:
                    forecast_return = raw_prediction

                lower_return = (
                    forecast_return
                    + bundle.residual_q10
                )

                upper_return = (
                    forecast_return
                    + bundle.residual_q90
                )

                (
                    forecast_return,
                    lower_return,
                    upper_return,
                ) = _ordered_return_bounds(
                    forecast_return,
                    lower_return,
                    upper_return,
                )

                forecast_status = "model_forecast"

            else:

                (
                    forecast_return,
                    lower_return,
                    upper_return,
                ) = _fallback_return(
                    latest_feature,
                    history,
                )

                model_name = "Historical trend fallback"

                forecast_status = (
                    "unvalidated_trend_fallback"
                )

            forecast = _monthly_path(
                last_date=last_date,
                last_nav=last_nav,
                expected_return=forecast_return,
                lower_return=lower_return,
                upper_return=upper_return,
            )

        metrics = calculate_metrics(
            history
        )

        forecast_summary = {
            "status": forecast_status,
            "model": model_name if forecast else None,
            "horizon_months": 12 if forecast else None,

            "expected_return_pct": (
                float(forecast_return * 100.0)
                if forecast
                else None
            ),

            "lower_return_pct": (
                float(lower_return * 100.0)
                if forecast
                else None
            ),

            "upper_return_pct": (
                float(upper_return * 100.0)
                if forecast
                else None
            ),

            "validation_mae_pct": (
                bundle.validation_mae_pct
            ),

            "validation_rmse_pct": (
                bundle.validation_rmse_pct
            ),

            "validation_rows": (
                bundle.validation_rows
            ),

            "holdout_test_mae_pct": (
                bundle.test_mae_pct
            ),

            "holdout_test_rmse_pct": (
                bundle.test_rmse_pct
            ),

            "holdout_test_rows": (
                bundle.test_rows
            ),

            "training_rows": (
                bundle.training_rows
            ),

            "uncertainty_note": (
                "Forecast range uses empirical residual quantiles "
                "from the chronological holdout test period. "
                "It is not a guaranteed confidence interval."
                if bundle.status == "trained"
                else
                "Historical range / trend-based estimate only. "
                "No validated ML forecast was available."
            ),

            "validation_note": (
                "Models were selected using a chronological "
                "validation period and evaluated on a separate "
                "chronological holdout test period. "
                "One-year target windows overlap, so evaluation "
                "observations are not statistically independent."
                if bundle.status == "trained"
                else
                "The ML model could not satisfy the minimum "
                "chronological training, validation and test "
                "requirements for this fund."
            ),

            "path_note": (
                "The model predicts a 12-month endpoint. "
                "Intermediate monthly points are geometrically "
                "interpolated, not separately predicted month by month."
            ),
        }

        warnings = list(
            item["warnings"]
        )

        if forecast_status == "unvalidated_trend_fallback":

            warnings.append(
                "Forecast uses a trend fallback because the fund "
                "did not have sufficient data for the required "
                "chronological ML evaluation."
            )

        if forecast_status == "insufficient_history":

            warnings.append(
                f"At least {MIN_FORECAST_OBSERVATIONS} NAV "
                "observations are required to generate a "
                "12-month forecast."
            )

        results.append(
            {
                "fund_name": fund_name,
                "source_file": item["source_file"],

                "historical": [
                    {
                        "date": row.date().isoformat(),
                        "nav": float(nav),
                    }
                    for row, nav in zip(
                        history["date"],
                        history["nav"],
                    )
                ],

                "forecast": forecast,

                "metrics": metrics,

                "forecast_summary": forecast_summary,

                "warnings": warnings,

                "cleaning": item["cleaning"],
            }
        )

    return results