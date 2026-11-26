from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import LinearSVC

from finapp.ml.rules import apply_rules

MODEL_NAMES = ["logreg_text", "logreg", "linsvc", "lgbm"]
RULE_MODES = ["feature", "override", "off"]
FEATURE_GROUPS = ("text", "meta", "numeric", "rule")
NUMERIC = ["log_amount", "hour_sin", "hour_cos", "weekday", "is_weekend", "day_of_month"]
META = ["mcc_str", "source", "user_id"]
MIN_CLASS_SIZE = 3


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hour = out["datetime"].dt.hour + out["datetime"].dt.minute / 60
    out["log_amount"] = np.log1p(out["amount"].astype(float))
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["weekday"] = out["datetime"].dt.weekday
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    out["day_of_month"] = out["datetime"].dt.day
    out["mcc_str"] = out["mcc"].astype("string").fillna("none")
    out["rule_category"] = apply_rules(out)["rule_category"].fillna("none")
    return out


def _char_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)


def _word_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")


def _categorical(groups: tuple[str, ...]) -> list[str]:
    return (META if "meta" in groups else []) + (["rule_category"] if "rule" in groups else [])


def _linear_features(groups: tuple[str, ...]) -> ColumnTransformer:
    parts = []
    if "text" in groups:
        parts += [("char", _char_tfidf(), "description_norm"), ("word", _word_tfidf(), "description_norm")]
    if cats := _categorical(groups):
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), cats))
    if "numeric" in groups:
        parts.append(("num", StandardScaler(), NUMERIC))
    return ColumnTransformer(parts)


def _tree_features(groups: tuple[str, ...], svd_components: int) -> ColumnTransformer:
    parts = []
    if "text" in groups:
        parts.append(("text", make_pipeline(_char_tfidf(), TruncatedSVD(svd_components, random_state=0)), "description_norm"))
    if cats := _categorical(groups):
        parts.append(("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cats))
    if "numeric" in groups:
        parts.append(("num", FunctionTransformer(), NUMERIC))
    return ColumnTransformer(parts)


def build_model(name: str, *, c: float = 10.0, calibrate: bool = True, seed: int = 0,
                groups: tuple[str, ...] = FEATURE_GROUPS) -> Pipeline:
    if name == "logreg_text":
        return Pipeline([("features", _linear_features(("text",))), ("clf", LogisticRegression(C=c, max_iter=3000))])
    if name == "logreg":
        return Pipeline([("features", _linear_features(groups)), ("clf", LogisticRegression(C=c, max_iter=3000))])
    if name == "linsvc":
        svc = LinearSVC(C=c / 10, max_iter=20000)
        clf = CalibratedClassifierCV(svc, method="sigmoid", cv=3) if calibrate else svc
        return Pipeline([("features", _linear_features(groups)), ("clf", clf)])
    if name == "lgbm":
        lgbm = LGBMClassifier(
            n_estimators=300, learning_rate=0.08, num_leaves=31, min_child_samples=10,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
            random_state=seed, verbose=-1, n_jobs=-1,
        )
        clf = CalibratedClassifierCV(lgbm, method="isotonic", cv=3) if calibrate else lgbm
        return Pipeline([("features", _tree_features(groups, 120)), ("clf", clf)])
    raise ValueError(f"Невідома модель: {name}")


@dataclass
class CascadeCategorizer:
    model_name: str = "logreg"
    threshold: float = 0.7
    model_params: dict = field(default_factory=dict)
    rule_mode: str = "feature"
    model: Pipeline | None = None
    rule_precision: dict[str, float] = field(default_factory=dict)
    dropped_classes: list[str] = field(default_factory=list)

    def fit(self, train: pd.DataFrame) -> CascadeCategorizer:
        if self.rule_mode not in RULE_MODES:
            raise ValueError(f"Невідомий режим правил: {self.rule_mode}")
        labeled = add_features(train[train["category"].notna()])
        counts = labeled["category"].value_counts()
        self.dropped_classes = counts[counts < MIN_CLASS_SIZE].index.tolist()
        labeled = labeled[~labeled["category"].isin(self.dropped_classes)]
        if labeled["category"].nunique() < 2:
            raise ValueError("Для навчання потрібно щонайменше дві категорії з мітками")
        params = dict(self.model_params)
        if self.rule_mode != "feature":
            params.setdefault("groups", tuple(g for g in FEATURE_GROUPS if g != "rule"))
        self.model = build_model(self.model_name, **params).fit(labeled, labeled["category"])
        rules = apply_rules(labeled)
        for rtype in ("mcc", "keyword"):
            m = rules["rule_type"] == rtype
            self.rule_precision[rtype] = float((rules.loc[m, "rule_category"] == labeled.loc[m, "category"]).mean()) if m.any() else 0.0
        return self

    def predict_model(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Модель не навчена")
        proba = self.model.predict_proba(add_features(df))
        classes = self.model.classes_
        return classes[proba.argmax(axis=1)], proba.max(axis=1), proba

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        ml_cat, ml_conf, _ = self.predict_model(df)
        out = pd.DataFrame(index=df.index)
        out["pred_category"] = ml_cat
        out["confidence"] = ml_conf
        out["category_source"] = "ml"

        if self.rule_mode == "override":
            rules = apply_rules(df)
            hit = rules["rule_category"].notna().to_numpy()
            out.loc[hit, "pred_category"] = rules.loc[hit, "rule_category"]
            out.loc[hit, "confidence"] = rules.loc[hit, "rule_type"].map(self.rule_precision).fillna(0.0)
            out.loc[hit, "category_source"] = "rule"

        manual = ((df["source"] == "manual") & df["category"].notna()).to_numpy()
        out.loc[manual, "pred_category"] = df.loc[manual, "category"]
        out.loc[manual, "confidence"] = 1.0
        out.loc[manual, "category_source"] = "manual"

        out["auto_assigned"] = out["confidence"] >= self.threshold
        return out


def fill_missing_categories(df: pd.DataFrame, min_labels: int = 50) -> tuple[pd.DataFrame, int]:
    """Заповнює категорії нерозмічених витрат моделлю, навченою на наявних мітках (або правилами, якщо міток мало)."""
    out = df.copy()
    todo = out.index[(out["direction"] == "expense") & out["category"].isna()]
    if len(todo) == 0:
        return out, 0
    labeled = out[(out["direction"] == "expense") & out["category"].notna()]
    if len(labeled) >= min_labels and labeled["category"].nunique() >= 2:
        pred, _, _ = CascadeCategorizer(model_name="logreg", model_params={"c": 1.0}).fit(labeled).predict_model(out.loc[todo])
        out.loc[todo, "category"] = pred
    else:
        out.loc[todo, "category"] = apply_rules(out.loc[todo])["rule_category"].fillna("other")
    return out, len(todo)
