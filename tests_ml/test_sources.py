from datetime import UTC, datetime, timedelta
from io import BytesIO

import pandas as pd

from finapp.ml import sources
from finapp.ml.data import load_transactions, validate_transactions
from finapp.monobank import Client

BLACK = {"id": "a1", "type": "black", "currencyCode": 980, "maskedPan": ["5375****1111"], "balance": 1}
WHITE = {"id": "a2", "type": "white", "currencyCode": 980, "maskedPan": ["4441****2222"]}
CSV = (
    '"Date and time",Description,MCC,"Card currency amount, (UAH)","Operation amount","Operation currency",'
    '"Exchange rate","Commission, (UAH)","Cashback amount, (UAH)",Balance\n'
    '"01.11.2026 12:00:00",Сільпо,5411,-300.0,-300.0,UAH,—,—,—,700.0\n'
)


class FakeClient(Client):
    def __init__(self, items: dict[str, list[dict]]) -> None:
        super().__init__(token="test")
        self.items, self.calls = items, []

    def get_statement(self, account_id, frm, to):
        self.calls.append((account_id, frm, to))
        return [i for i in self.items.get(account_id, []) if frm.timestamp() <= i["time"] <= to.timestamp()]


def kyiv_unix(s: str) -> int:
    return int(pd.Timestamp(s).tz_localize("Europe/Kyiv").timestamp())


def csv_store() -> pd.DataFrame:
    df, _ = load_transactions(BytesIO(CSV.encode()))
    return df.drop(columns=["description_norm"])


def test_tc33_incremental_sync_dedups_csv_overlap_and_internal():
    store = csv_store()
    now = datetime(2026, 11, 3, 12, tzinfo=UTC)
    client = FakeClient({
        "a1": [{"id": "m-dup", "time": kyiv_unix("2026-11-01 12:00:00"), "amount": -30000, "description": "Сільпо"},
               {"id": "m-new", "time": kyiv_unix("2026-11-02 09:15:00"), "amount": -12000, "description": "Аптека"},
               {"id": "m-out", "time": kyiv_unix("2026-11-02 10:00:00"), "amount": -50000, "description": "На білу"}],
        "a2": [{"id": "m-in", "time": kyiv_unix("2026-11-02 10:00:30"), "amount": 50000, "description": "З чорної"}],
    })
    since = sources.sync_start(store, now)
    new = sources.fetch_statements(client, [BLACK, WHITE], since, now)
    assert set(new["transaction_id"]) == {"m-dup", "m-new"}
    new, _ = validate_transactions(new)
    merged, added = sources.merge_transactions(store, new.drop(columns=["description_norm"]))
    assert added == 1 and len(merged) == 2
    assert merged.iloc[-1]["transaction_id"] == "m-new"


def test_tc34_sync_window_and_request_estimate():
    now = datetime(2026, 11, 3, tzinfo=UTC)
    assert sources.sync_start(None, now) == now - timedelta(days=sources.FIRST_SYNC_DAYS)
    since = sources.sync_start(csv_store(), now)
    assert since == datetime(2026, 11, 1, 10, tzinfo=UTC) - sources.SYNC_OVERLAP
    assert sources.statement_requests(now - timedelta(days=2), now) == 1
    assert sources.statement_requests(now - timedelta(days=62), now) == 3


class FakeKeyring:
    def __init__(self):
        self.store = {}

    def get_password(self, s, u):
        return self.store.get((s, u))

    def set_password(self, s, u, p):
        self.store[(s, u)] = p

    def delete_password(self, s, u):
        del self.store[(s, u)]


def test_tc35_token_env_then_keychain(monkeypatch):
    kr = FakeKeyring()
    monkeypatch.setattr(sources, "_keyring", lambda: kr)
    monkeypatch.delenv(sources.TOKEN_ENV, raising=False)
    assert sources.load_token() is None and sources.token_origin() is None
    sources.save_token("  secret-token \n")
    assert sources.load_token() == "secret-token" and sources.token_origin() == "keychain"
    monkeypatch.setenv(sources.TOKEN_ENV, "from-env")
    assert sources.load_token() == "from-env" and sources.token_origin() == "env"
    monkeypatch.delenv(sources.TOKEN_ENV)
    sources.forget_token()
    sources.forget_token()
    assert sources.load_token() is None


def test_tc36_store_and_prefs_roundtrip(tmp_path):
    store_path, prefs_path = tmp_path / "s.csv", tmp_path / "p.json"
    sources.write_store(csv_store(), store_path)
    back = sources.read_store(store_path)
    df, report = validate_transactions(back)
    assert len(df) == 1 and not report.dropped and back["transaction_id"].iloc[0].startswith("csv-")
    assert sources.read_prefs(prefs_path) == {"accounts": [], "auto": True}
    sources.write_prefs({"accounts": [BLACK], "auto": False}, prefs_path)
    prefs = sources.read_prefs(prefs_path)
    assert prefs["auto"] is False and "balance" not in prefs["accounts"][0]
