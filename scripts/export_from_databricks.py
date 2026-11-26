"""Вивантаження історії з {catalog}.bronze.mono_raw (після mono_backfill) у CSV для finapp.ml через Databricks Connect.

uv run --group dev --group ml python scripts/export_from_databricks.py --catalog dev
"""

from __future__ import annotations

import argparse
from pathlib import Path

from databricks.connect import DatabricksSession

from finapp.ml.monobank_export import from_bronze, mark_internal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="dev")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--keep-internal", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("data/private/transactions_monobank.csv"))
    args = ap.parse_args()

    builder = DatabricksSession.builder.serverless(True)
    spark = (builder.profile(args.profile) if args.profile else builder).getOrCreate()
    bronze = spark.table(f"{args.catalog}.bronze.mono_raw").dropDuplicates(["transaction_id"]).toPandas()
    if bronze.empty:
        raise SystemExit("bronze.mono_raw порожня — спершу запустіть job mono_backfill")
    df = mark_internal(from_bronze(bronze), own_ibans=set())
    df = df[df["currency"] == "UAH"]
    if not args.keep_internal:
        print(f"Видалено переказів між власними рахунками: {int(df['is_internal'].sum())}")
        df = df[df["is_internal"] == 0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["counter_iban"]).to_csv(args.out, index=False)
    print(f"Збережено {len(df)} операцій у {args.out} ({df['datetime'].min():%d.%m.%Y}–{df['datetime'].max():%d.%m.%Y})")


if __name__ == "__main__":
    main()
