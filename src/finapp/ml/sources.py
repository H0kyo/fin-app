"""Джерела транзакцій для застосунку: файл, Monobank API (інкрементально), Databricks (bronze.mono_raw)."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from finapp.ml.monobank_export import KYIV, from_bronze, mark_internal, to_rows
from finapp.monobank import STATEMENT_WINDOW, Client

TOKEN_ENV = "MONO_TOKEN"
KEYRING_SERVICE = "finapp-monobank"
KEYRING_USER = "personal-token"
SYNC_PATH = Path("data/private/monobank_sync.csv")
PREFS_PATH = Path("data/private/monobank_prefs.json")
SYNC_OVERLAP = timedelta(days=1)
FIRST_SYNC_DAYS = 31
WAREHOUSE_ENV = "DATABRICKS_WAREHOUSE_ID"
CATALOG_ENV = "FINAPP_CATALOG"

KEY_COLUMNS = ["datetime", "direction", "amount", "description"]


def _keyring():
    try:
        import keyring
        from keyring.backends.fail import Keyring as NoBackend
    except ImportError:
        return None
    return None if isinstance(keyring.get_keyring(), NoBackend) else keyring


def _keychain_token() -> str | None:
    from keyring.errors import KeyringError

    kr = _keyring()
    try:
        return kr.get_password(KEYRING_SERVICE, KEYRING_USER) if kr else None
    except KeyringError:
        return None


def token_origin() -> str | None:
    if os.environ.get(TOKEN_ENV):
        return "env"
    return "keychain" if _keychain_token() else None


def load_token() -> str | None:
    return os.environ.get(TOKEN_ENV) or _keychain_token()


def keychain_available() -> bool:
    return _keyring() is not None


def save_token(token: str) -> None:
    kr = _keyring()
    if kr is None:
        raise RuntimeError("Системне сховище секретів (keyring) недоступне")
    kr.set_password(KEYRING_SERVICE, KEYRING_USER, token.strip())


def forget_token() -> None:
    from keyring.errors import PasswordDeleteError

    kr = _keyring()
    if kr is not None and _keychain_token():
        try:
            kr.delete_password(KEYRING_SERVICE, KEYRING_USER)
        except PasswordDeleteError:
            return


def statement_requests(since: datetime, until: datetime) -> int:
    return max(1, -(-int((until - since).total_seconds()) // int(STATEMENT_WINDOW.total_seconds())))


def sync_start(store: pd.DataFrame | None, now: datetime) -> datetime:
    if store is None or store.empty:
        return now - timedelta(days=FIRST_SYNC_DAYS)
    last = pd.Timestamp(store["datetime"].max()).tz_localize(KYIV, ambiguous=True, nonexistent="shift_forward")
    last = last.tz_convert("UTC").to_pydatetime()
    return min(last - SYNC_OVERLAP, now)


def fetch_statements(client: Client, accounts: list[dict], since: datetime, until: datetime | None = None,
                     keep_internal: bool = False, on_account: Callable[[dict], None] | None = None) -> pd.DataFrame:
    until = until or datetime.now(UTC)
    rows: list[dict] = []
    for acc in accounts:
        if on_account:
            on_account(acc)
        rows += to_rows(acc, list(client.iter_statement(acc["id"], since, until)), user_id="me")
    if not rows:
        return pd.DataFrame(columns=["transaction_id", "user_id", "datetime", "amount", "description"])
    df = mark_internal(pd.DataFrame(rows), own_ibans={a["iban"] for a in accounts if a.get("iban")})
    if not keep_internal:
        df = df[df["is_internal"] == 0]
    return df.drop(columns=["counter_iban"]).reset_index(drop=True)


def _match_key(df: pd.DataFrame) -> pd.Series:
    amount = pd.to_numeric(df["amount"], errors="coerce").abs().round(2)
    return (pd.to_datetime(df["datetime"]).dt.floor("s").astype(str) + "|" + df["direction"].astype(str) + "|"
            + amount.astype(str) + "|" + df["description"].fillna("").astype(str).str.strip().str.lower())


def merge_transactions(base: pd.DataFrame | None, new: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Додає нові операції; дублікати відкидаються за transaction_id та за ключем «час, напрям, сума, опис»."""
    if base is None or base.empty:
        return new.reset_index(drop=True), len(new)
    fresh = new[~new["transaction_id"].astype(str).isin(base["transaction_id"].astype(str))]
    fresh = fresh[~_match_key(fresh).isin(set(_match_key(base)))]
    return pd.concat([base, fresh], ignore_index=True), len(fresh)


def read_store(path: Path = SYNC_PATH) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"transaction_id": str, "user_id": str}, parse_dates=["datetime"])


def write_store(df: pd.DataFrame, path: Path = SYNC_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in df.columns if c not in {"description_norm"}]
    df[cols].sort_values("datetime").to_csv(path, index=False)


ACCOUNT_FIELDS = ("id", "type", "currencyCode", "maskedPan", "iban")


def read_prefs(path: Path = PREFS_PATH) -> dict:
    if not path.exists():
        return {"accounts": [], "auto": True}
    return json.loads(path.read_text(encoding="utf-8"))


def write_prefs(prefs: dict, path: Path = PREFS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prefs = {**prefs, "accounts": [{k: a[k] for k in ACCOUNT_FIELDS if k in a} for a in prefs.get("accounts", [])]}
    path.write_text(json.dumps(prefs, ensure_ascii=False, indent=2), encoding="utf-8")


def databricks_available() -> bool:
    return bool(os.environ.get(WAREHOUSE_ENV))


def read_databricks(catalog: str | None = None) -> pd.DataFrame:
    from databricks import sql
    from databricks.sdk.core import Config

    cfg = Config()
    catalog = catalog or os.environ.get(CATALOG_ENV, "dev")
    with sql.connect(server_hostname=cfg.host, http_path=f"/sql/1.0/warehouses/{os.environ[WAREHOUSE_ENV]}",
                     credentials_provider=lambda: cfg.authenticate) as conn, conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {catalog}.bronze.mono_raw QUALIFY row_number() OVER "
                    "(PARTITION BY transaction_id ORDER BY _ingested_at DESC) = 1")
        bronze = cur.fetchall_arrow().to_pandas()
    if bronze.empty:
        raise ValueError(f"{catalog}.bronze.mono_raw порожня")
    df = mark_internal(from_bronze(bronze), own_ibans=set())
    return df[df["is_internal"] == 0].drop(columns=["counter_iban"]).reset_index(drop=True)
