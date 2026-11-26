"""Перетворення виписки Monobank у схему транзакцій finapp.ml і пошук переказів між власними рахунками."""

from __future__ import annotations

import pandas as pd

from finapp.ml.data import ISO4217

KYIV = "Europe/Kyiv"
INTERNAL_WINDOW_SECONDS = 180


def account_label(acc: dict) -> str:
    pan = (acc.get("maskedPan") or ["—"])[0]
    return f"{acc.get('type', '?')} {ISO4217.get(acc['currencyCode'], acc['currencyCode'])} {pan}"


def to_rows(account: dict, items: list[dict], user_id: str) -> list[dict]:
    currency = ISO4217.get(account["currencyCode"], str(account["currencyCode"]))
    return [
        {
            "transaction_id": it["id"],
            "user_id": user_id,
            "datetime": pd.Timestamp(it["time"], unit="s", tz="UTC").tz_convert(KYIV).tz_localize(None),
            "amount": it["amount"] / 100,
            "currency": currency,
            "description": it.get("description", ""),
            "mcc": it.get("mcc"),
            "source": "monobank",
            "direction": "expense" if it["amount"] < 0 else "income",
            "category": None,
            "account": account_label(account),
            "comment": it.get("comment"),
            "counter_name": it.get("counterName"),
            "counter_iban": it.get("counterIban"),
            "is_internal": 0,
        }
        for it in items
    ]


def mark_internal(df: pd.DataFrame, own_ibans: set[str]) -> pd.DataFrame:
    """Перекази між власними рахунками: протилежні суми на різних рахунках у межах 3 хв або IBAN власного рахунку."""
    df = df.sort_values("datetime").reset_index(drop=True)
    df.loc[df["counter_iban"].isin(own_ibans), "is_internal"] = 1
    out, inc = df[df["amount"] < 0], df[df["amount"] > 0]
    for i, row in out.iterrows():
        cand = inc[(inc["account"] != row["account"]) & (inc["amount"] == -row["amount"])
                   & ((inc["datetime"] - row["datetime"]).abs().dt.total_seconds() <= INTERNAL_WINDOW_SECONDS)]
        if len(cand):
            df.loc[[i, cand.index[0]], "is_internal"] = 1
    return df


BRONZE_TO_ITEM = {"transaction_id": "id", "time_unix": "time", "amount": "amount", "description": "description",
                  "mcc": "mcc", "comment": "comment", "counter_name": "counterName", "counter_iban": "counterIban"}


def from_bronze(bronze: pd.DataFrame, user_id: str = "me") -> pd.DataFrame:
    """Рядки bronze.mono_raw → схема finapp.ml; рахунок позначається останніми 4 символами його ID."""
    rows: list[dict] = []
    for (account_id, currency), g in bronze.groupby(["account_id", "currency_code"], dropna=False):
        account = {"id": account_id, "type": "acc", "currencyCode": int(currency) if pd.notna(currency) else 980,
                   "maskedPan": [f"…{str(account_id)[-4:]}"]}
        items = g.rename(columns=BRONZE_TO_ITEM)[list(BRONZE_TO_ITEM.values())]
        items = items.astype(object).where(items.notna(), None).to_dict("records")
        rows += to_rows(account, items, user_id)
    return pd.DataFrame(rows)
