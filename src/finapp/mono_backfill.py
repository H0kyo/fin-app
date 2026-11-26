"""Backfill історії Monobank у landing-зону вебхука: далі bronze.mono_raw підхоплює файли через Auto Loader.

Токен читається з Secret Scope всередині Databricks і не залишає workspace.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

from finapp.monobank import SECRET_SCOPE, TOKEN_SECRET_KEY, Client


def webhook_envelope(account_id: str, item: dict) -> dict:
    return {"type": "StatementItem", "data": {"account": account_id, "statementItem": item}}


def to_jsonl(account_id: str, items: Iterable[dict], skip_ids: set[str]) -> tuple[str, int, int]:
    lines, skipped = [], 0
    for item in items:
        if item["id"] in skip_ids:
            skipped += 1
            continue
        lines.append(json.dumps(webhook_envelope(account_id, item), ensure_ascii=False))
    return "\n".join(lines), len(lines), skipped


def select_accounts(info: dict, account_ids: list[str] | None, include_jars: bool) -> list[dict]:
    accounts = list(info.get("accounts", []))
    if include_jars:
        accounts += [{**j, "type": "jar"} for j in info.get("jars", [])]
    if account_ids:
        return [a for a in accounts if a["id"] in account_ids]
    return [a for a in accounts if a.get("currencyCode") == 980]


def existing_ids(spark, table: str) -> set[str]:
    if not spark.catalog.tableExists(table):
        return set()
    return {r[0] for r in spark.table(table).select("transaction_id").distinct().collect()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--landing-path", required=True)
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--accounts", default="", help="ID рахунків через кому; порожньо — усі гривневі")
    ap.add_argument("--include-jars", default="false")
    ap.add_argument("--secret-scope", default=SECRET_SCOPE)
    ap.add_argument("--secret-key", default=TOKEN_SECRET_KEY)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from databricks.sdk.runtime import dbutils, spark

    args = parse_args(argv)
    client = Client(dbutils.secrets.get(scope=args.secret_scope, key=args.secret_key))
    accounts = select_accounts(client.get_client_info(), [a for a in args.accounts.split(",") if a],
                               args.include_jars.lower() == "true")
    skip = existing_ids(spark, f"{args.catalog}.bronze.mono_raw")
    to = datetime.now(UTC)
    frm = to - timedelta(days=30.44 * args.months)
    run_ts = to.strftime("%Y%m%dT%H%M%SZ")
    print(f"Рахунків: {len(accounts)}; вже в bronze: {len(skip)} операцій; період {frm:%Y-%m-%d}..{to:%Y-%m-%d}")

    for acc in accounts:
        items: list[dict] = []
        error = None
        try:
            items.extend(client.iter_statement(acc["id"], frm, to))
        except Exception as e:  # noqa: BLE001
            error = e
        body, written, skipped = to_jsonl(acc["id"], items, skip)
        if written:
            dbutils.fs.put(f"{args.landing_path.rstrip('/')}/backfill-{run_ts}-{acc['id']}.json", body, overwrite=False)
        status = f"; зупинено: {error}" if error else ""
        print(f"{acc.get('type', '?')} {acc.get('currencyCode')}: отримано {len(items)}, записано {written}, "
              f"пропущено як наявні {skipped}{status}")
