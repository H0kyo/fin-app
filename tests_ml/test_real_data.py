from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from finapp.ml import monobank_export as export_monobank
from finapp.ml.categorizer import fill_missing_categories
from finapp.ml.data import expenses, validate_transactions
from finapp.ml.forecasting import plan_backtest
from finapp.monobank import Client


class FakeClient(Client):
    def __init__(self, items: list[dict]) -> None:
        super().__init__(token="test")
        self.items, self.calls = items, []

    def get_statement(self, account_id, frm, to):
        self.calls.append((frm, to))
        hits = [i for i in self.items if frm.timestamp() <= i["time"] <= to.timestamp()]
        return sorted(hits, key=lambda i: -i["time"])[:500]


def test_tc22_statement_pagination_over_500():
    start = datetime(2025, 1, 1, tzinfo=UTC)
    items = [{"id": f"t{i}", "time": int((start + timedelta(minutes=i)).timestamp())} for i in range(1200)]
    client = FakeClient(items)
    got = list(client.iter_statement("acc", start, start + timedelta(days=1)))
    assert len(got) == 1200 and len({i["id"] for i in got}) == 1200
    assert len(client.calls) == 3


def test_tc23_internal_transfers_marked():
    acc_black = {"id": "a1", "type": "black", "currencyCode": 980, "maskedPan": ["5375****1111"]}
    acc_white = {"id": "a2", "type": "white", "currencyCode": 980, "maskedPan": ["4441****2222"]}
    t = 1735725600
    rows = export_monobank.to_rows(acc_black, [{"id": "x1", "time": t, "amount": -50000, "description": "На білу картку"},
                                               {"id": "x2", "time": t + 3600, "amount": -12550, "description": "АТБ", "mcc": 5411}], "me")
    rows += export_monobank.to_rows(acc_white, [{"id": "y1", "time": t + 30, "amount": 50000, "description": "З чорної картки"}], "me")
    df = export_monobank.mark_internal(pd.DataFrame(rows), own_ibans=set())
    assert set(df.loc[df["is_internal"] == 1, "transaction_id"]) == {"x1", "y1"}
    atb = df[df["transaction_id"] == "x2"].iloc[0]
    assert atb["amount"] == -125.5 and atb["direction"] == "expense" and atb["datetime"].hour == 13


@pytest.mark.parametrize(("months", "expected"), [(36, (24, 10)), (24, (12, 10)), (12, (6, 4)), (10, (6, 2))])
def test_tc24_plan_backtest(months, expected):
    assert plan_backtest(months, h=3) == expected


def test_tc25_plan_backtest_too_short():
    with pytest.raises(ValueError, match="закороткий"):
        plan_backtest(8, h=3)


def test_tc26_fill_missing_categories(dataset):
    df = dataset.copy()
    exp_idx = df.index[df["direction"] == "expense"]
    hidden = exp_idx[::3]
    truth = df.loc[hidden, "category"].copy()
    df.loc[hidden, "category"] = None
    filled, n = fill_missing_categories(df)
    assert n == len(hidden) and expenses(filled)["category"].notna().all()
    assert (filled.loc[hidden, "category"] == truth).mean() > 0.85


def test_tc27_fill_falls_back_to_rules():
    rows = [{"transaction_id": "1", "user_id": "me", "datetime": "2025-01-01 10:00", "amount": -100,
             "description": "АТБ", "mcc": 5411, "category": None},
            {"transaction_id": "2", "user_id": "me", "datetime": "2025-01-02 10:00", "amount": -100,
             "description": "LIQPAY*Невідомо", "mcc": 7399, "category": None}]
    df, _ = validate_transactions(pd.DataFrame(rows))
    filled, n = fill_missing_categories(df)
    assert n == 2 and filled["category"].tolist() == ["groceries", "other"]


def test_tc28_backfill_envelope_matches_bronze_schema():
    import json
    import re
    from pathlib import Path

    from finapp.mono_backfill import to_jsonl

    bronze_src = (Path(__file__).resolve().parents[1] / "src/finapp_etl/transformations/bronze_mono_raw.py").read_text()
    item_fields = set(re.findall(r'StructField\("(\w+)"', bronze_src)) - {"type", "data", "account", "statementItem"}
    item = {"id": "ZuHWzqE7cbrY", "time": 1735725600, "description": "АТБ", "mcc": 5411, "amount": -12550,
            "currencyCode": 980, "balance": 100000}
    body, written, skipped = to_jsonl("acc1", [item, {**item, "id": "old"}], skip_ids={"old"})
    assert (written, skipped) == (1, 1)
    env = json.loads(body.splitlines()[0])
    assert env["type"] == "StatementItem" and env["data"]["account"] == "acc1"
    assert set(env["data"]["statementItem"]) <= item_fields


def test_tc29_select_accounts():
    from finapp.mono_backfill import select_accounts

    info = {"accounts": [{"id": "uah", "currencyCode": 980}, {"id": "usd", "currencyCode": 840}],
            "jars": [{"id": "jar1", "currencyCode": 980}]}
    assert [a["id"] for a in select_accounts(info, None, False)] == ["uah"]
    assert [a["id"] for a in select_accounts(info, None, True)] == ["uah", "jar1"]
    assert [a["id"] for a in select_accounts(info, ["usd"], False)] == ["usd"]


def test_tc30_from_bronze_matches_direct_export():
    bronze = pd.DataFrame([{"account_id": "acc0001", "transaction_id": "t1", "time_unix": 1735725600, "amount": -12550,
                            "description": "АТБ", "mcc": 5411, "currency_code": 980, "comment": None,
                            "counter_name": None, "counter_iban": None}])
    df = export_monobank.from_bronze(bronze)
    row = df.iloc[0]
    assert (row["transaction_id"], row["amount"], row["direction"], row["mcc"], row["currency"]) == \
        ("t1", -125.5, "expense", 5411, "UAH")
    clean, _ = validate_transactions(df)
    assert clean["amount"].iloc[0] == 125.5


def test_tc31_client_retries_dropped_connection(monkeypatch):
    import requests

    from finapp import monobank

    monkeypatch.setattr(monobank, "MIN_REQUEST_INTERVAL_SECONDS", 0)
    calls = {"n": 0}

    class Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return [{"id": "ok"}]

    def flaky_get(url, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.ConnectionError("Remote end closed connection without response")
        return Resp()

    client = monobank.Client(token="test")
    monkeypatch.setattr(client.session, "get", flaky_get)
    assert client._get("/personal/statement/a/1/2") == [{"id": "ok"}] and calls["n"] == 2
    assert client.session.headers["Connection"] == "close"


MONO_CSV_UA = (
    '"Дата i час операції","Деталі операції",MCC,"Сума в валюті картки (UAH)","Сума в валюті операції",'
    'Валюта,Курс,"Сума комісій (UAH)","Сума кешбеку (UAH)","Залишок після операції"\n'
    '"02.11.2026 15:50:16",Кінотеатр,7832,-150.0,-150.0,UAH,—,—,—,850.0\n'
    '"01.11.2026 10:00:00",Зарплата,4829,1000.0,1000.0,UAH,—,—,—,1000.0\n'
)
MONO_CSV_EN = (
    '"Date and time",Description,MCC,"Card currency amount, (USD)","Operation amount","Operation currency",'
    '"Exchange rate","Commission, (USD)","Cashback amount, (USD)",Balance\n'
    '"02.11.2026 15:50:16",Cinema,7832,-15.0,-600.0,UAH,40.0,—,—,85.0\n'
)


@pytest.mark.parametrize("raw", [MONO_CSV_UA, MONO_CSV_EN])
def test_tc32_monobank_app_csv(raw):
    from io import BytesIO

    from finapp.ml.data import load_transactions

    df, report = load_transactions(BytesIO(("﻿" + raw).encode()))
    first, _ = load_transactions(BytesIO(raw.encode()))
    assert report.rows_out == raw.count("\n") - 1 and not report.dropped
    assert df["transaction_id"].tolist() == first["transaction_id"].tolist()
    assert df["transaction_id"].is_unique
    cinema = df.iloc[-1]
    assert cinema["direction"] == "expense" and cinema["mcc"] == 7832
    assert cinema["datetime"] == pd.Timestamp("2026-11-02 15:50:16")
    assert cinema["currency"] == ("USD" if "USD" in raw else "UAH")
