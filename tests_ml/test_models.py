import numpy as np
import pandas as pd
import pytest

from finapp.ml.anomaly import IsolationForestDetector, RobustZScoreDetector
from finapp.ml.categorizer import CascadeCategorizer
from finapp.ml.data import expenses, monthly_spend, validate_transactions
from finapp.ml.forecasting import GlobalLGBMForecaster, ets, naive, rolling_origin_backtest, seasonal_naive
from finapp.ml.metrics import expected_calibration_error, macro_f1, mae, threshold_report, wape
from finapp.ml.rules import apply_rules
from tests_ml.conftest import tx


def test_tc11_metrics_known_values():
    assert mae([100, 200], [110, 180]) == 15
    assert wape([100, 0, 100], [90, 10, 100]) == pytest.approx(0.1)
    assert macro_f1(["a", "a", "b", "b"], ["a", "a", "b", "a"]) == pytest.approx((0.8 + 2 / 3) / 2)
    assert expected_calibration_error([1, 1, 0, 0], [1.0, 1.0, 0.0, 0.0]) == 0


def test_tc12_metrics_edge_cases():
    assert np.isnan(wape([0, 0], [1, 2]))
    with pytest.raises(ValueError):
        mae([], [])
    r = threshold_report(["a"], ["a"], [0.5], threshold=1.0)
    assert r["coverage"] == 0 and np.isnan(r["accuracy_auto"])
    assert threshold_report(["a"], ["a"], [0.5], threshold=0.0)["coverage"] == 1


def test_tc13_rules_mcc_and_keywords():
    df = pd.DataFrame([tx(), tx(mcc=None, description="таксі додому"), tx(mcc=7399, description="LIQPAY*TOV X")])
    r = apply_rules(df)
    assert r["rule_category"].tolist() == ["groceries", "transport", None]
    assert r["rule_type"].tolist() == ["mcc", "keyword", None]


def test_tc14_cascade_outputs(dataset):
    exp = expenses(dataset)
    cz = CascadeCategorizer(model_name="logreg", rule_mode="override", threshold=0.7).fit(exp.iloc[:3000])
    out = cz.predict(exp.iloc[3000:3500])
    assert out["confidence"].between(0, 1).all()
    assert set(out["category_source"]) <= {"manual", "rule", "ml"}
    manual = exp.iloc[3000:3500]["source"] == "manual"
    assert (out.loc[manual, "category_source"] == "manual").all() and (out.loc[manual, "confidence"] == 1).all()
    assert (out["auto_assigned"] == (out["confidence"] >= 0.7)).all()


def test_tc15_categorizer_quality_above_rules(dataset):
    exp = expenses(dataset)
    train, test = exp.iloc[: int(len(exp) * 0.8)], exp.iloc[int(len(exp) * 0.8):]
    test = test[test["source"] != "manual"]
    pred, _conf, _ = CascadeCategorizer(model_name="logreg").fit(train).predict_model(test)
    rules = apply_rules(test)["rule_category"].fillna("none")
    assert macro_f1(test["category"], pred) > macro_f1(test["category"], rules)


def test_tc16_categorizer_needs_two_classes():
    df, _ = validate_transactions(pd.DataFrame([tx(transaction_id=str(i)) for i in range(5)]))
    with pytest.raises(ValueError, match="дві категорії"):
        CascadeCategorizer().fit(df)


def test_tc17_baselines_and_short_series():
    y = np.arange(1, 25, dtype=float)
    assert seasonal_naive(y, 3).tolist() == [13, 14, 15]
    assert seasonal_naive(np.array([5.0, 7.0]), 2).tolist() == [7, 7]
    assert naive(np.array([3.0]), 2).tolist() == [3, 3]
    assert ets(np.zeros(30), 3).tolist() == [0, 0, 0]
    with pytest.raises(ValueError):
        naive(np.array([]), 1)
    with pytest.raises(ValueError):
        naive(y, 0)


def test_tc18_lgbm_forecast_nonnegative(dataset):
    panel = monthly_spend(dataset)
    fc = GlobalLGBMForecaster().fit(panel).predict(panel, 3)
    assert len(fc) == panel.groupby(["user_id", "category"]).ngroups * 3
    assert (fc["forecast"] >= 0).all()


def test_tc19_backtest_too_short_series(dataset):
    panel = monthly_spend(dataset)
    with pytest.raises(ValueError, match="закороткий"):
        rolling_origin_backtest(panel, ["naive"], h=3, min_train=40)


def test_tc20_anomaly_detected_and_constant_group():
    rows = [tx(transaction_id=str(i), amount=-(100 + i)) for i in range(30)]
    rows += [tx(transaction_id="big", amount=-5000), tx(transaction_id="c1", category="cash", amount=-500, mcc=6011)]
    df, _ = validate_transactions(pd.DataFrame(rows))
    flags = RobustZScoreDetector(threshold=3.5).fit(df).predict(df)
    assert flags[df["transaction_id"] == "big"].item()
    assert flags.sum() == 1
    iso = IsolationForestDetector(contamination=0.05).fit(df)
    assert iso.score(df)[df["transaction_id"] == "big"].item() == iso.score(df).max()
