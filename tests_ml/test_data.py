import io

import pandas as pd
import pytest

from finapp.ml.data import (
    DataValidationError,
    load_transactions,
    monthly_spend,
    normalize_description,
    validate_transactions,
)
from tests_ml.conftest import tx


def test_tc01_valid_rows_loaded():
    df, rep = validate_transactions(pd.DataFrame([tx(), tx(transaction_id="t2", amount=2500, description="Зарплата")]))
    assert len(df) == 2 and rep.rows_out == 2 and not rep.dropped
    assert set(df["direction"]) == {"expense", "income"}
    assert (df["amount"] > 0).all()


def test_tc02_invalid_values_dropped_or_fixed():
    rows = [tx(), tx(transaction_id="bad_date", datetime="31.02.2024"), tx(transaction_id="nan", amount=None),
            tx(transaction_id="zero", amount=0), tx(transaction_id="mcc", mcc=99999), tx(transaction_id="neg_mcc", mcc=-5),
            tx(transaction_id="nodesc", description=None)]
    df, rep = validate_transactions(pd.DataFrame(rows))
    assert rep.dropped == {"некоректна дата": 1, "відсутня або нульова сума": 2}
    assert rep.fixed["некоректний MCC замінено пропуском"] == 2
    assert df.loc[df["transaction_id"] == "nodesc", "description"].item() == ""
    assert df.loc[df["transaction_id"] == "mcc", "mcc"].isna().item()


def test_tc03_duplicates_removed():
    df, rep = validate_transactions(pd.DataFrame([tx(), tx(), tx(source="manual")]))
    assert len(df) == 2 and rep.dropped["дублікат (source, transaction_id)"] == 1


def test_tc04_empty_file():
    with pytest.raises(DataValidationError, match="порожній"):
        load_transactions(io.BytesIO(b""))


def test_tc05_missing_required_columns():
    with pytest.raises(DataValidationError, match="amount"):
        load_transactions(io.BytesIO(b"transaction_id,user_id,datetime,description\n1,u,2025-01-01,x\n"))


def test_tc06_all_rows_invalid():
    with pytest.raises(DataValidationError, match="жодного коректного"):
        validate_transactions(pd.DataFrame([tx(amount=0), tx(transaction_id="t2", datetime="not a date")]))


def test_tc07_monobank_json_statement():
    raw = b'[{"id":"ZuHWzqE7cbrY","time":1735725600,"description":"\xd0\x90\xd0\xa2\xd0\x91","mcc":5411,"amount":-12550,"currencyCode":980}]'
    df, _ = load_transactions(io.BytesIO(raw))
    row = df.iloc[0]
    assert row["amount"] == 125.5 and row["direction"] == "expense" and row["mcc"] == 5411 and row["currency"] == "UAH"


def test_tc08_boundary_amounts():
    df, _ = validate_transactions(pd.DataFrame([tx(amount=-0.01), tx(transaction_id="t2", amount=-1e9)]))
    assert df["amount"].tolist() == [0.01, 1e9]


def test_tc09_normalize_description():
    assert normalize_description("SILPO 202 Львів!!") == "silpo львів"
    assert normalize_description("М’ЯСНА  лавка") == "мясна лавка"
    assert normalize_description(None) == ""


def test_tc10_monthly_panel_fills_gaps():
    rows = [tx(), tx(transaction_id="t2", datetime="2025-03-10")]
    df, _ = validate_transactions(pd.DataFrame(rows))
    panel = monthly_spend(df)
    assert panel["amount"].tolist() == [100.0, 0.0, 100.0]


def test_tc21_mixed_date_formats():
    rows = [tx(), tx(transaction_id="t2", datetime="2025-03-10"), tx(transaction_id="t3", datetime="05.04.2025 08:30")]
    df, rep = validate_transactions(pd.DataFrame(rows))
    assert not rep.dropped
    assert df["datetime"].dt.month.tolist() == [1, 3, 4]
