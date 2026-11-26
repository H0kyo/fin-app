from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

MAD_TO_SIGMA = 0.6745


class RobustZScoreDetector:
    """Залишок log-суми відносно медіани групи «користувач × категорія», нормований на MAD."""

    def __init__(self, threshold: float = 3.5, min_group: int = 8) -> None:
        self.threshold = threshold
        self.min_group = min_group
        self.stats: pd.DataFrame | None = None
        self.global_stats: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame, category_col: str = "category") -> RobustZScoreDetector:
        x = df.assign(log_amount=np.log1p(df["amount"]))
        agg = x.groupby(["user_id", category_col])["log_amount"]
        self.stats = pd.DataFrame({"median": agg.median(), "mad": agg.apply(_mad), "n": agg.size()})
        cat = x.groupby(category_col)["log_amount"]
        self.global_stats = pd.DataFrame({"median": cat.median(), "mad": cat.apply(_mad)})
        self.category_col = category_col
        return self

    def score(self, df: pd.DataFrame) -> pd.Series:
        if self.stats is None:
            raise RuntimeError("Детектор не навчений")
        key = pd.MultiIndex.from_arrays([df["user_id"], df[self.category_col]])
        st = self.stats.reindex(key)
        small = (st["n"].fillna(0) < self.min_group).to_numpy()
        gl = self.global_stats.reindex(df[self.category_col])
        med = np.where(small, gl["median"], st["median"])
        mad = np.where(small, gl["mad"], st["mad"])
        mad = np.where((mad > 0) & np.isfinite(mad), mad, 1.0)
        z = MAD_TO_SIGMA * (np.log1p(df["amount"].to_numpy()) - med) / mad
        return pd.Series(np.nan_to_num(z, nan=0.0), index=df.index)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return self.score(df) > self.threshold


def _mad(s: pd.Series) -> float:
    return float(np.median(np.abs(s - np.median(s))))


class IsolationForestDetector:
    def __init__(self, contamination: float = 0.01, seed: int = 0) -> None:
        self.contamination = contamination
        self.seed = seed
        self.ref = RobustZScoreDetector()
        self.model: IsolationForest | None = None

    def _features(self, df: pd.DataFrame) -> np.ndarray:
        hour = df["datetime"].dt.hour
        return np.column_stack([
            np.log1p(df["amount"]),
            self.ref.score(df),
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            df["datetime"].dt.weekday,
        ])

    def fit(self, df: pd.DataFrame, category_col: str = "category") -> IsolationForestDetector:
        self.ref.fit(df, category_col)
        self.model = IsolationForest(n_estimators=300, contamination=self.contamination, random_state=self.seed)
        self.model.fit(self._features(df))
        return self

    def score(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(-self.model.score_samples(self._features(df)), index=df.index)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(self._features(df)) == -1, index=df.index)


def monthly_overspend(actual: pd.DataFrame, forecast: pd.DataFrame, residual_std: pd.Series, k: float = 2.0) -> pd.DataFrame:
    m = actual.merge(forecast, on=["user_id", "category", "month"], how="inner")
    sd = residual_std.reindex(m["category"]).to_numpy()
    m["excess"] = m["amount"] - m["forecast"]
    m["is_overspend"] = m["excess"] > k * np.nan_to_num(sd, nan=np.inf)
    return m
