from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from finapp.ml.metrics import mae, wape

LOCAL_METHODS = ["naive", "seasonal_naive", "moving_average", "ets"]
LAGS = [1, 2, 3, 6, 12]
SEASON = 12


def naive(y: np.ndarray, h: int) -> np.ndarray:
    _check(y, h)
    return np.repeat(y[-1], h).astype(float)


def seasonal_naive(y: np.ndarray, h: int, m: int = SEASON) -> np.ndarray:
    _check(y, h)
    if len(y) < m:
        return naive(y, h)
    return np.array([y[len(y) - m + (i % m)] for i in range(h)], dtype=float)


def moving_average(y: np.ndarray, h: int, window: int = 3) -> np.ndarray:
    _check(y, h)
    return np.repeat(np.mean(y[-window:]), h).astype(float)


def ets(y: np.ndarray, h: int, m: int = SEASON) -> np.ndarray:
    _check(y, h)
    y = np.asarray(y, float)
    if len(y) < 4 or np.all(y == y[0]):
        return naive(y, h)
    seasonal = "add" if len(y) >= 2 * m else None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(
                y, trend="add", damped_trend=True, seasonal=seasonal, seasonal_periods=m if seasonal else None,
                initialization_method="estimated",
            ).fit(optimized=True)
        fc = fit.forecast(h)
    except (ValueError, np.linalg.LinAlgError):
        return moving_average(y, h)
    return np.clip(np.nan_to_num(fc, nan=y[-1]), 0, None)


def _check(y: np.ndarray, h: int) -> None:
    if len(y) == 0:
        raise ValueError("Порожній часовий ряд")
    if h < 1:
        raise ValueError("Горизонт прогнозу має бути ≥ 1")


LOCAL = {"naive": naive, "seasonal_naive": seasonal_naive, "moving_average": moving_average, "ets": ets}


def _series_features(y: np.ndarray, months: np.ndarray) -> pd.DataFrame:
    s = pd.Series(y, dtype=float)
    shifted = s.shift(1)
    scale = shifted.rolling(12, min_periods=1).mean().fillna(0) + 1.0
    feats = {f"lag_{k}": s.shift(k) / scale for k in LAGS}
    feats["roll_mean_3"] = shifted.rolling(3, min_periods=1).mean() / scale
    feats["roll_mean_6"] = shifted.rolling(6, min_periods=1).mean() / scale
    feats["roll_std_3"] = shifted.rolling(3, min_periods=2).std() / scale
    feats["zero_share_6"] = (shifted == 0).rolling(6, min_periods=1).mean()
    feats["month"] = months
    feats["log_scale"] = np.log(scale)
    out = pd.DataFrame(feats)
    out["scale"] = scale
    out["target"] = s / scale
    return out


class GlobalLGBMForecaster:
    """Одна LightGBM-модель для всіх рядів «користувач × категорія»; рекурсивний багатокроковий прогноз."""

    def __init__(self, objective: str = "l1", seed: int = 0, **params) -> None:
        self.params = {
            "objective": objective, "n_estimators": 400, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 20,
            "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "random_state": seed, "verbose": -1,
        } | params
        self.model: LGBMRegressor | None = None
        self.categories: list[str] = []

    def _frame(self, panel: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for (user, cat), g in panel.groupby(["user_id", "category"], sort=False):
            g = g.sort_values("month")
            f = _series_features(g["amount"].to_numpy(), g["month"].dt.month.to_numpy())
            f["category_code"] = self.categories.index(cat) if cat in self.categories else -1
            f["user_id"], f["category"], f["month_p"] = user, cat, g["month"].to_numpy()
            parts.append(f)
        return pd.concat(parts, ignore_index=True)

    @property
    def feature_names(self) -> list[str]:
        return [f"lag_{k}" for k in LAGS] + ["roll_mean_3", "roll_mean_6", "roll_std_3", "zero_share_6",
                                             "month", "log_scale", "category_code"]

    def fit(self, panel: pd.DataFrame) -> GlobalLGBMForecaster:
        self.categories = sorted(panel["category"].unique())
        frame = self._frame(panel).dropna(subset=["lag_1"])
        if len(frame) < 10:
            raise ValueError("Недостатньо історії для навчання глобальної моделі")
        self.model = LGBMRegressor(**self.params).fit(frame[self.feature_names], frame["target"])
        return self

    def predict(self, panel: pd.DataFrame, h: int) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Модель не навчена")
        out = []
        for (user, cat), g in panel.groupby(["user_id", "category"], sort=False):
            g = g.sort_values("month")
            y = g["amount"].to_numpy(dtype=float)
            months = g["month"].dt.month.to_numpy()
            last = g["month"].iloc[-1]
            for step in range(1, h + 1):
                nxt = last + step
                f = _series_features(np.append(y, np.nan), np.append(months, nxt.month)).iloc[[-1]]
                f["category_code"] = self.categories.index(cat) if cat in self.categories else -1
                pred = max(float(self.model.predict(f[self.feature_names])[0] * f["scale"].iloc[0]), 0.0)
                y, months = np.append(y, pred), np.append(months, nxt.month)
                out.append({"user_id": user, "category": cat, "month": nxt, "h": step, "forecast": pred})
        return pd.DataFrame(out)


def forecast_panel(panel: pd.DataFrame, method: str, h: int, lgbm_params: dict | None = None) -> pd.DataFrame:
    if method == "lgbm":
        return GlobalLGBMForecaster(**(lgbm_params or {})).fit(panel).predict(panel, h)
    fn = LOCAL[method]
    out = []
    for (user, cat), g in panel.groupby(["user_id", "category"], sort=False):
        g = g.sort_values("month")
        fc = fn(g["amount"].to_numpy(dtype=float), h)
        last = g["month"].iloc[-1]
        out += [{"user_id": user, "category": cat, "month": last + i + 1, "h": i + 1, "forecast": float(v)}
                for i, v in enumerate(fc)]
    return pd.DataFrame(out)


def rolling_origin_backtest(panel: pd.DataFrame, methods: list[str], h: int = 3, n_origins: int = 10,
                            min_train: int = 24, lgbm_params: dict | None = None) -> pd.DataFrame:
    months = sorted(panel["month"].unique())
    last_origin = len(months) - h - 1
    first_origin = max(min_train - 1, last_origin - n_origins + 1)
    if first_origin > last_origin:
        raise ValueError("Ряд закороткий для обраних min_train і горизонту")
    results = []
    for oi in range(first_origin, last_origin + 1):
        origin = months[oi]
        train = panel[panel["month"] <= origin]
        actual = panel[(panel["month"] > origin) & (panel["month"] <= origin + h)]
        for method in methods:
            fc = forecast_panel(train, method, h, lgbm_params)
            merged = fc.merge(actual, on=["user_id", "category", "month"], how="inner")
            merged["method"], merged["origin"] = method, origin
            results.append(merged)
    return pd.concat(results, ignore_index=True)


def score_backtest(bt: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    by = by or ["method"]
    return (
        bt.groupby(by)
        .apply(lambda g: pd.Series({"MAE": mae(g["amount"], g["forecast"]), "WAPE": wape(g["amount"], g["forecast"])}),
               include_groups=False)
        .reset_index()
    )


def plan_backtest(n_months: int, h: int = 3, max_origins: int = 10, min_train: int = 24) -> tuple[int, int]:
    """Підбирає (min_train, n_origins) під довжину історії: для 36 міс. — (24, 10), для коротших — менше."""
    available = n_months - h
    min_train = min(min_train, max(6, available - max_origins + 1))
    n_origins = min(max_origins, available - min_train + 1)
    if n_origins < 1:
        raise ValueError(f"Ряд закороткий для бектестингу: {n_months} міс. при горизонті {h}")
    return min_train, n_origins
