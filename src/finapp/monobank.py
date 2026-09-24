from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import requests

BASE_URL = "https://api.monobank.ua"
STATEMENT_WINDOW = timedelta(days=31) - timedelta(seconds=1)
MIN_REQUEST_INTERVAL_SECONDS = 60
SECRET_SCOPE = "finapp"
TOKEN_SECRET_KEY = "mono_api_token"


def get_api_token() -> str:
    from databricks.sdk.runtime import dbutils

    return dbutils.secrets.get(scope=SECRET_SCOPE, key=TOKEN_SECRET_KEY)


@dataclass
class Client:
    token: str
    session: requests.Session | None = None
    _last_request_at: float = 0.0

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()
        self.session.headers.update({"X-Token": self.token})

    def _throttle(self) -> None:
        wait = MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)

    def _get(self, path: str) -> Any:
        self._throttle()
        resp = self.session.get(f"{BASE_URL}{path}", timeout=30)

        self._last_request_at = time.monotonic()
        resp.raise_for_status()

        return resp.json()

    def get_client_info(self) -> dict:
        return self._get("/personal/client-info")

    def get_statement(self, account_id: str, frm: datetime, to: datetime) -> list[dict]:
        if to - frm > STATEMENT_WINDOW:
            raise ValueError("Monobank personal/statement window is limited to 31 days")
        return self._get(f"/personal/statement/{account_id}/{int(frm.timestamp())}/{int(to.timestamp())}")

    def iter_statement(self, account_id: str, frm: datetime, to: datetime) -> Iterator[dict]:
        window_start = frm
        while window_start < to:
            window_end = min(window_start + STATEMENT_WINDOW, to)
            yield from self.get_statement(account_id, window_start, window_end)
            window_start = window_end


def statement_item_to_bronze_row(account_id: str, item: dict, source: str = "monobank") -> dict:
    return {
        "account_id": account_id,
        "transaction_id": item["id"],
        "time_unix": item["time"],
        "description": item.get("description"),
        "mcc": item.get("mcc"),
        "original_mcc": item.get("originalMcc"),
        "amount": item["amount"],
        "operation_amount": item.get("operationAmount"),
        "currency_code": item.get("currencyCode"),
        "commission_rate": item.get("commissionRate"),
        "cashback_amount": item.get("cashbackAmount"),
        "balance": item.get("balance"),
        "comment": item.get("comment"),
        "receipt_id": item.get("receiptId"),
        "counter_edrpou": item.get("counterEdrpou"),
        "counter_iban": item.get("counterIban"),
        "counter_name": item.get("counterName"),
        "_ingested_at": datetime.now(UTC),
        "_source": source,
    }


def parse_webhook_event(raw_json: dict) -> dict | None:
    if raw_json.get("type") != "StatementItem":
        return None
    data = raw_json["data"]
    return statement_item_to_bronze_row(account_id=data["account"], item=data["statementItem"])
