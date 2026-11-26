"""Zero-shot Chronos-Bolt на тих самих точках відліку, що й rolling_origin_backtest.

uv run --group ml --with chronos-forecasting python scripts/chronos_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from chronos import BaseChronosPipeline

from finapp.ml import DEMO_DATASET
from finapp.ml.categorizer import fill_missing_categories
from finapp.ml.data import load_transactions, monthly_spend
from finapp.ml.forecasting import plan_backtest, score_backtest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEMO_DATASET))
    ap.add_argument("--out", default="Claude outputs/Курсова/results")
    ap.add_argument("--model", default="amazon/chronos-bolt-small")
    args = ap.parse_args()

    df, _ = load_transactions(args.data)
    panel = monthly_spend(fill_missing_categories(df)[0])
    months = sorted(panel["month"].unique())
    h = 3
    min_train, n_origins = plan_backtest(len(months), h)
    last_origin = len(months) - h - 1
    first_origin = max(min_train - 1, last_origin - n_origins + 1)

    pipe = BaseChronosPipeline.from_pretrained(args.model, device_map="cpu", torch_dtype=torch.float32)
    rows = []
    t0 = time.perf_counter()
    for oi in range(first_origin, last_origin + 1):
        origin = months[oi]
        keys, ctx = [], []
        for (user, cat), g in panel[panel["month"] <= origin].groupby(["user_id", "category"]):
            keys.append((user, cat))
            ctx.append(torch.tensor(g.sort_values("month")["amount"].to_numpy(), dtype=torch.float32))
        _, mean = pipe.predict_quantiles(ctx, prediction_length=h, quantile_levels=[0.5])
        for (user, cat), fc in zip(keys, mean.numpy()):
            for step in range(h):
                rows.append({"user_id": user, "category": cat, "month": origin + step + 1, "h": step + 1,
                             "forecast": max(float(fc[step]), 0.0), "origin": origin})
    elapsed = time.perf_counter() - t0
    fc = pd.DataFrame(rows).merge(panel, on=["user_id", "category", "month"], how="inner").assign(method="chronos")
    overall = score_backtest(fc).iloc[0].to_dict()
    byh = score_backtest(fc, ["method", "h"])
    out = Path(args.out)
    byh.round(4).to_csv(out / "tables" / "fc_chronos_by_h.csv", index=False)
    fc.to_csv(out / "tables" / "fc_chronos_raw.csv", index=False)
    res = json.loads((out / "results.json").read_text())
    res.setdefault("forecasting", {})["chronos"] = {"model": args.model, "MAE": overall["MAE"], "WAPE": overall["WAPE"],
                                                     "seconds": elapsed}
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float))
    print(overall, elapsed)


if __name__ == "__main__":
    main()
