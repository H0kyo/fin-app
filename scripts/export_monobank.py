"""Вивантаження власної історії транзакцій Monobank у CSV-схему модуля finapp.ml.

Токен: https://api.monobank.ua/ → змінна оточення MONO_TOKEN або прихований ввід.
Ліміт API — 1 запит на 60 с, тому рік історії одного рахунку займає ~12–13 хв.

uv run --group ml python scripts/export_monobank.py --months 24
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from finapp.ml.monobank_export import account_label, mark_internal, to_rows
from finapp.monobank import STATEMENT_WINDOW, Client

OUT_DIR = Path("data/private")
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=12, help="Глибина історії, місяців")
    ap.add_argument("--accounts", nargs="*", help="ID рахунків; за замовчуванням — усі гривневі картки")
    ap.add_argument("--include-jars", action="store_true", help="Додати банки (накопичення)")
    ap.add_argument("--keep-internal", action="store_true", help="Не видаляти перекази між власними рахунками")
    ap.add_argument("--user-id", default="me")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "transactions_monobank.csv")
    args = ap.parse_args()

    token = os.environ.get("MONO_TOKEN") or getpass.getpass("Monobank X-Token (ввід приховано): ").strip()
    client = Client(token)
    info = client.get_client_info()
    accounts = info.get("accounts", [])
    if args.include_jars:
        accounts += [{**j, "type": "jar", "maskedPan": [j.get("title", "")]} for j in info.get("jars", [])]
    print("Рахунки:")
    for acc in accounts:
        print(f"  {acc['id']}  {account_label(acc)}")
    chosen = [a for a in accounts if (a["id"] in args.accounts if args.accounts else a["currencyCode"] == 980)]
    if not chosen:
        raise SystemExit("Не обрано жодного рахунку")
    own_ibans = {a["iban"] for a in info.get("accounts", []) if a.get("iban")}

    to = datetime.now(UTC)
    frm = to - timedelta(days=30.44 * args.months)
    windows = int((to - frm) / STATEMENT_WINDOW) + 1
    print(f"Обрано {len(chosen)} рах.; ≈{windows * len(chosen)} запитів, ≈{windows * len(chosen)} хв (ліміт 1/хв)")

    raw_dir = args.out.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for acc in chosen:
        items: list[dict] = []
        try:
            items.extend(client.iter_statement(acc["id"], frm, to))
        except Exception as e:  # noqa: BLE001
            print(f"  {account_label(acc)}: зупинено на {len(items)} операціях ({e})")
        (raw_dir / f"{acc['id']}.json").write_text(json.dumps(items, ensure_ascii=False))
        rows += to_rows(acc, items, args.user_id)
        print(f"  {account_label(acc)}: {len(items)} операцій")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("API не повернув жодної операції")
    df = mark_internal(df, own_ibans)
    if not args.keep_internal:
        print(f"Видалено переказів між власними рахунками: {int(df['is_internal'].sum())}")
        df = df[df["is_internal"] == 0]
    df.drop(columns=["counter_iban"]).to_csv(args.out, index=False)
    print(f"Збережено {len(df)} операцій у {args.out} (період {df['datetime'].min():%d.%m.%Y}–{df['datetime'].max():%d.%m.%Y})")


if __name__ == "__main__":
    main()
