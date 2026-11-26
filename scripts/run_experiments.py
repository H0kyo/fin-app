"""Відтворювані експерименти для розділів 2–3: EDA, категоризація, прогнозування, аномалії.

uv run --group ml python scripts/run_experiments.py --out "Claude outputs/Курсова/results"
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from finapp.ml import DEMO_DATASET
from finapp.ml.anomaly import IsolationForestDetector, RobustZScoreDetector
from finapp.ml.categories import CATEGORY_IDS, label
from finapp.ml.categorizer import CascadeCategorizer, add_features, build_model, fill_missing_categories
from finapp.ml.data import expenses, load_transactions, monthly_spend
from finapp.ml.forecasting import GlobalLGBMForecaster, plan_backtest, rolling_origin_backtest, score_backtest
from finapp.ml.metrics import classification_summary, macro_f1, threshold_report
from finapp.ml.rules import apply_rules

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
plt.rcParams.update({"font.size": 11, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})

METHOD_LABELS = {
    "naive": "Наївний", "seasonal_naive": "Сезонний наївний", "moving_average": "Ковзне середнє (3)",
    "ets": "ETS", "lgbm": "LightGBM (глобальна)",
}
MODEL_LABELS = {
    "rules": "Лише правила", "logreg_text": "TF-IDF + LogReg (текст)", "logreg": "LogReg (змішані ознаки)",
    "linsvc": "LinearSVC + калібрування", "lgbm": "LightGBM + калібрування",
}


class Run:
    def __init__(self, out: Path) -> None:
        self.out = out
        (out / "fig").mkdir(parents=True, exist_ok=True)
        (out / "tables").mkdir(parents=True, exist_ok=True)
        self.results: dict = {}

    def fig(self, name: str) -> None:
        plt.savefig(self.out / "fig" / f"{name}.png")
        plt.close()

    def table(self, name: str, df: pd.DataFrame) -> None:
        df.to_csv(self.out / "tables" / f"{name}.csv", index=False)

    def save(self) -> None:
        (self.out / "results.json").write_text(json.dumps(self.results, ensure_ascii=False, indent=2, default=float))


def eda(run: Run, raw: pd.DataFrame, df: pd.DataFrame, report) -> None:
    r = {"rows_raw": len(raw), "rows_clean": len(df), "dropped": report.dropped, "fixed": report.fixed,
         "columns": list(raw.columns), "users": int(df["user_id"].nunique()),
         "period": [str(df["datetime"].min().date()), str(df["datetime"].max().date())],
         "missing_raw": {c: int(v) for c, v in raw.isna().sum().items()},
         "source_counts": df["source"].value_counts().to_dict(),
         "direction_counts": df["direction"].value_counts().to_dict(),
         "anomalies": int(df["is_anomaly"].sum()),
         "unique_descriptions": int(df["description_norm"].nunique()),
         "mcc_missing_share": float(df["mcc"].isna().mean())}
    exp = expenses(df)

    stats = exp.groupby("category")["amount"].agg(["count", "mean", "median", "std", "min", "max", "skew"])
    stats["share"] = stats["count"] / stats["count"].sum()
    stats = stats.sort_values("count", ascending=False).reset_index()
    stats["category"] = stats["category"].map(label)
    run.table("eda_category_stats", stats.round(2))
    a = exp["amount"]
    r["amount_stats"] = {"mean": a.mean(), "median": a.median(), "mode": float(a.round(-1).mode().iloc[0]),
                         "std": a.std(), "var": a.var(), "min": a.min(), "max": a.max(), "skew": a.skew(),
                         "log_skew": np.log1p(a).skew(), "q25": a.quantile(0.25), "q75": a.quantile(0.75)}
    r["imbalance_ratio"] = float(stats["count"].max() / stats["count"].min())

    sample_ids = raw.loc[raw["datetime"].astype(str).str.contains("31.02") | raw["amount"].isna()
                         | (pd.to_numeric(raw["mcc"], errors="coerce") > 9999) | raw["description"].isna()].head(6)
    run.table("eda_dirty_examples", sample_ids)
    before = raw[raw["description"].astype(str).str.contains(r"\d{3,}|[A-Z]{4,}", regex=True, na=False)].head(6)
    after = df.set_index("transaction_id").reindex(before["transaction_id"])[
        ["datetime", "amount", "direction", "description_norm", "mcc"]].reset_index()
    run.table("eda_before", before[["transaction_id", "datetime", "amount", "description", "mcc"]])
    run.table("eda_after", after)

    _, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].hist(a, bins=80, color="#4C72B0")
    ax[0].set(title="Сума витрати, грн", xlabel="грн", ylabel="Кількість")
    ax[1].hist(np.log10(a), bins=60, color="#55A868")
    ax[1].set(title="log10(сума)", xlabel="log10(грн)", ylabel="Кількість")
    run.fig("eda_amount_hist")

    order = stats["category"].tolist()
    exp_l = exp.assign(cat_label=exp["category"].map(label))
    _, ax = plt.subplots(figsize=(11, 4.6))
    ax.boxplot([np.log10(exp_l.loc[exp_l["cat_label"] == c, "amount"]) for c in order], showfliers=True,
               flierprops={"markersize": 2})
    ax.set_xticks(range(1, len(order) + 1), order, rotation=35, ha="right")
    ax.set(ylabel="log10(сума, грн)", title="Розподіл сум витрат за категоріями")
    run.fig("eda_boxplot")

    _, ax = plt.subplots(figsize=(10, 4))
    ax.barh(order[::-1], stats["count"][::-1], color="#4C72B0")
    ax.set(xlabel="Кількість транзакцій", title="Розподіл транзакцій за категоріями")
    run.fig("eda_category_counts")

    m = exp.assign(month=exp["datetime"].dt.to_period("M").dt.to_timestamp())
    piv = m.pivot_table(index="month", columns="user_id", values="amount", aggfunc="sum")
    inc = df[df["direction"] == "income"].assign(month=lambda x: x["datetime"].dt.to_period("M").dt.to_timestamp())
    _, ax = plt.subplots(figsize=(11, 4))
    for c in piv.columns:
        ax.plot(piv.index, piv[c] / 1000, label=c)
    ax.set(ylabel="тис. грн", title="Місячні витрати користувачів")
    ax.legend(ncol=5, fontsize=9)
    run.fig("eda_monthly_users")
    r["income_vs_expense"] = {"income_total": float(inc["amount"].sum()), "expense_total": float(exp["amount"].sum())}

    cat_month = m.pivot_table(index="month", columns="category", values="amount", aggfunc="sum").fillna(0)
    seas = cat_month.groupby(cat_month.index.month).mean()
    seas = seas / seas.mean()
    _, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(seas.T.values, aspect="auto", cmap="RdYlBu_r", vmin=0.4, vmax=1.9)
    ax.set_yticks(range(len(seas.columns)), [label(c) for c in seas.columns])
    ax.set_xticks(range(12), ["Січ", "Лют", "Бер", "Кві", "Тра", "Чер", "Лип", "Сер", "Вер", "Жов", "Лис", "Гру"])
    plt.colorbar(im, ax=ax, label="Відношення до середнього")
    ax.set_title("Сезонний профіль витрат за категоріями")
    run.fig("eda_seasonality")

    hw = exp.groupby([exp["datetime"].dt.weekday, exp["datetime"].dt.hour]).size().unstack(fill_value=0)
    _, ax = plt.subplots(figsize=(11, 3.4))
    im = ax.imshow(hw.values, aspect="auto", cmap="Blues")
    ax.set_yticks(range(7), ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"])
    ax.set_xticks(range(len(hw.columns)), hw.columns)
    ax.set(xlabel="Година", title="Кількість транзакцій за днем тижня та годиною")
    plt.colorbar(im, ax=ax)
    run.fig("eda_weekday_hour")

    f = add_features(exp)
    num = pd.DataFrame({
        "log_сума": f["log_amount"], "година": f["datetime"].dt.hour, "день_тижня": f["weekday"],
        "вихідний": f["is_weekend"], "день_місяця": f["day_of_month"], "є_MCC": f["mcc"].notna().astype(int),
        "ручний": (f["source"] == "manual").astype(int), "довжина_опису": f["description"].str.len(),
    })
    corr = num.corr()
    run.table("eda_corr", corr.round(3).reset_index())
    _fig, ax = plt.subplots(figsize=(7, 5.6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax)
    ax.set_title("Кореляційна матриця числових ознак")
    run.fig("eda_corr")

    corr_cat = cat_month.corr()
    r["top_category_corr"] = (corr_cat.where(np.triu(np.ones(corr_cat.shape), 1).astype(bool)).stack()
                              .sort_values(ascending=False).head(5).round(3).to_dict())
    r["top_category_corr"] = {f"{label(a)} — {label(b)}": v for (a, b), v in r["top_category_corr"].items()}

    panel = monthly_spend(df)
    r["panel"] = {"series": int(panel.groupby(["user_id", "category"]).ngroups), "months": int(panel["month"].nunique()),
                  "zero_share": float((panel["amount"] == 0).mean())}
    run.results["eda"] = r


def time_split(exp: pd.DataFrame, q: float = 0.8):
    cut = exp["datetime"].quantile(q)
    return exp[exp["datetime"] <= cut], exp[exp["datetime"] > cut], cut


def categorization(run: Run, df: pd.DataFrame) -> None:
    r: dict = {}
    exp = expenses(df).dropna(subset=["category"])
    train, test, cut = time_split(exp)
    ev = test[test["source"] != "manual"]
    r["split"] = {"cutoff": str(cut), "train": len(train), "test": len(test), "test_eval": len(ev),
                  "test_manual": int((test["source"] == "manual").sum())}

    rules = apply_rules(ev)
    hit = rules["rule_category"].notna()
    r["rules"] = {"coverage": float(hit.mean()),
                  "precision": float((rules.loc[hit, "rule_category"] == ev.loc[hit, "category"]).mean()),
                  "coverage_mcc": float((rules["rule_type"] == "mcc").mean()),
                  "coverage_keyword": float((rules["rule_type"] == "keyword").mean())}
    rows = [{"model": "rules", **classification_summary(ev["category"], rules["rule_category"].fillna("none")),
             "fit_s": 0.0, "predict_ms_per_1k": 0.0}]

    tscv = TimeSeriesSplit(n_splits=3)
    tr_lab = add_features(train)
    tuning = []
    for name, grid in {"logreg": [0.1, 1, 10, 100], "linsvc": [0.1, 1, 10, 100]}.items():
        for c in grid:
            scores = []
            for a, b in tscv.split(tr_lab):
                m = build_model(name, c=c, calibrate=False).fit(tr_lab.iloc[a], tr_lab["category"].iloc[a])
                scores.append(macro_f1(tr_lab["category"].iloc[b], m.predict(tr_lab.iloc[b])))
            tuning.append({"model": name, "C": c, "macro_f1_mean": np.mean(scores), "macro_f1_std": np.std(scores)})
    tuning = pd.DataFrame(tuning)
    run.table("cat_tuning", tuning.round(4))
    best_c = tuning.loc[tuning.groupby("model")["macro_f1_mean"].idxmax()].set_index("model")["C"].to_dict()
    r["best_C"] = best_c

    fitted = {}
    for name in ["logreg_text", "logreg", "linsvc", "lgbm"]:
        params = {"c": best_c.get(name, 10.0)}
        t0 = time.perf_counter()
        cz = CascadeCategorizer(model_name=name, model_params=params, rule_mode="feature").fit(train)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred, conf, _ = cz.predict_model(ev)
        pred_ms = (time.perf_counter() - t0) / len(ev) * 1000 * 1000
        fitted[name] = (cz, pred, conf)
        rows.append({"model": name, **classification_summary(ev["category"], pred, conf), "fit_s": fit_s,
                     "predict_ms_per_1k": pred_ms})
        print(f"  {name}: {rows[-1]}")
    comp = pd.DataFrame(rows)
    run.table("cat_models", comp.round(4))
    best = comp.iloc[1:].sort_values("macro_f1", ascending=False).iloc[0]["model"]
    r["best_model"] = best
    cz, pred, conf = fitted[best]

    ab = []
    for groups in [("text",), ("text", "meta"), ("text", "meta", "numeric"), ("text", "meta", "numeric", "rule"),
                   ("meta", "numeric", "rule")]:
        m = build_model("logreg", c=best_c["logreg"], groups=groups).fit(tr_lab, tr_lab["category"])
        p = m.predict(add_features(ev))
        ab.append({"groups": " + ".join(groups), **classification_summary(ev["category"], p)})
    run.table("cat_ablation", pd.DataFrame(ab).round(4))

    modes = []
    for mode in ["off", "feature", "override"]:
        c2 = CascadeCategorizer(model_name="logreg", model_params={"c": best_c["logreg"]}, rule_mode=mode).fit(train)
        out = c2.predict(ev)
        modes.append({"mode": mode, **classification_summary(ev["category"], out["pred_category"], out["confidence"]),
                      "share_rule": float((out["category_source"] == "rule").mean())})
    full = CascadeCategorizer(model_name=best, model_params={"c": best_c.get(best, 10.0)}).fit(train).predict(test)
    r["cascade_full_test"] = {**classification_summary(test["category"], full["pred_category"]),
                              "share_manual": float((full["category_source"] == "manual").mean())}
    run.table("cat_rule_modes", pd.DataFrame(modes).round(4))

    th = pd.DataFrame([threshold_report(ev["category"], pred, conf, t) for t in np.round(np.arange(0.3, 1.0, 0.05), 2)])
    run.table("cat_threshold", th.round(4))
    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(th["threshold"], th["coverage"], "o-", label="Частка автоматичної категоризації")
    ax.plot(th["threshold"], th["accuracy_auto"], "s-", label="Точність на автоматично категоризованих")
    ax.axvline(0.7, color="grey", ls="--", lw=1)
    ax.set(xlabel="Поріг впевненості θ", ylim=(0.6, 1.005), title=f"Вплив порогу θ ({MODEL_LABELS[best]})")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    run.fig("cat_threshold")

    correct = (ev["category"].to_numpy() == pred).astype(float)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, 9)
    rel = pd.DataFrame({"bin": idx, "conf": conf, "correct": correct}).groupby("bin").agg(conf=("conf", "mean"), acc=("correct", "mean"), n=("conf", "size"))
    _, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Ідеальне калібрування")
    ax.plot(rel["conf"], rel["acc"], "o-", label=MODEL_LABELS[best])
    ax.set(xlabel="Середня впевненість", ylabel="Частка правильних", title="Діаграма надійності")
    ax.legend()
    ax.grid(alpha=0.3)
    run.fig("cat_reliability")

    lc = []
    for frac in [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        for seed in range(3):
            sub = train.sample(frac=frac, random_state=seed) if frac < 1 else train
            if sub["category"].nunique() < train["category"].nunique():
                continue
            m = build_model("logreg", c=best_c["logreg"]).fit(add_features(sub), sub["category"])
            lc.append({"frac": frac, "n": len(sub), "seed": seed, "macro_f1": macro_f1(ev["category"], m.predict(add_features(ev)))})
            if frac == 1.0:
                break
    if not lc:
        lc = [{"frac": 1.0, "n": len(train), "seed": 0,
               "macro_f1": macro_f1(ev["category"], build_model("logreg", c=best_c["logreg"]).fit(tr_lab, tr_lab["category"]).predict(add_features(ev)))}]
    lc = pd.DataFrame(lc).groupby(["frac", "n"], as_index=False)["macro_f1"].agg(["mean", "std"]).fillna(0)
    run.table("cat_learning_curve", lc.round(4))
    _, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(lc["n"], lc["mean"], yerr=lc["std"], fmt="o-", capsize=3, label="LogReg (змішані ознаки)")
    ax.axhline(comp.loc[comp["model"] == "rules", "macro_f1"].iloc[0], color="C3", ls="--", label="Лише правила")
    ax.set(xscale="log", xlabel="Кількість розмічених транзакцій у навчанні", ylabel="Макро-F1", title="Крива навчання")
    ax.legend()
    ax.grid(alpha=0.3)
    run.fig("cat_learning_curve")

    seen = set(train["description_norm"])
    new = ~ev["description_norm"].isin(seen)
    r["unseen"] = {"share": float(new.mean()), "n": int(new.sum()),
                   "rules_acc": float((rules.loc[new, "rule_category"] == ev.loc[new, "category"]).mean()),
                   "model_acc": float((pred[new.to_numpy()] == ev.loc[new, "category"]).mean()),
                   "seen_model_acc": float((pred[~new.to_numpy()] == ev.loc[~new, "category"]).mean())}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    counts = exp["category"].value_counts()
    lab = add_features(exp[exp["category"].isin(counts[counts >= 5].index)])
    cv = []
    for name in ["logreg", "linsvc"]:
        s = []
        for a, b in skf.split(lab, lab["category"]):
            m = build_model(name, c=best_c[name]).fit(lab.iloc[a], lab["category"].iloc[a])
            s.append(macro_f1(lab["category"].iloc[b], m.predict(lab.iloc[b])))
        cv.append({"model": name, "macro_f1_mean": np.mean(s), "macro_f1_std": np.std(s)})
    run.table("cat_cv5", pd.DataFrame(cv).round(4))

    rep = classification_report(ev["category"], pred, output_dict=True, zero_division=0)
    per = pd.DataFrame([{"category": label(k), "precision": v["precision"], "recall": v["recall"], "f1": v["f1-score"],
                         "support": v["support"]} for k, v in rep.items() if k in set(ev["category"]) | set(pred)])
    run.table("cat_per_class", per.round(3))
    labels_order = [c for c in CATEGORY_IDS if c in set(ev["category"])] + sorted(set(ev["category"]) - set(CATEGORY_IDS))
    cm = confusion_matrix(ev["category"], pred, labels=labels_order, normalize="true")
    _fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels_order)), [label(c) for c in labels_order], rotation=45, ha="right")
    ax.set_yticks(range(len(labels_order)), [label(c) for c in labels_order])
    for i in range(len(cm)):
        for j in range(len(cm)):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.6 else "black")
    ax.set(xlabel="Передбачена категорія", ylabel="Справжня категорія", title=f"Нормована матриця помилок ({MODEL_LABELS[best]})")
    plt.colorbar(im, ax=ax)
    run.fig("cat_confusion")

    errs = ev.assign(pred=pred, conf=conf)
    errs = errs[errs["pred"] != errs["category"]].sort_values("conf", ascending=False)
    run.table("cat_errors", errs[["description", "mcc", "amount", "user_id", "category", "pred", "conf"]].head(15).round(3))
    oks = ev.assign(pred=pred, conf=conf)
    oks = oks[(oks["pred"] == oks["category"]) & oks["mcc"].isin([4215, 5399, 7399, 5411, 4829])]
    run.table("cat_correct_hard", oks[["description", "mcc", "amount", "user_id", "category", "pred", "conf"]]
              .drop_duplicates("description").head(10).round(3))
    run.results["categorization"] = r


def forecasting(run: Run, df: pd.DataFrame) -> None:
    r: dict = {}
    panel = monthly_spend(df)
    methods = ["naive", "seasonal_naive", "moving_average", "ets", "lgbm"]
    t0 = time.perf_counter()
    min_train, n_origins = plan_backtest(panel["month"].nunique(), h=3)
    r["min_train"], r["n_origins"] = min_train, n_origins
    bt = rolling_origin_backtest(panel, methods, h=3, n_origins=n_origins, min_train=min_train)
    r["backtest_seconds"] = time.perf_counter() - t0
    r["origins"] = [str(bt["origin"].min()), str(bt["origin"].max())]
    overall = score_backtest(bt)
    overall["method_label"] = overall["method"].map(METHOD_LABELS)
    run.table("fc_overall", overall.round(4))
    byh = score_backtest(bt, ["method", "h"])
    run.table("fc_by_h", byh.round(4))
    bycat = score_backtest(bt, ["method", "category"]).pivot(index="category", columns="method", values="WAPE")
    bycat.index = bycat.index.map(label)
    run.table("fc_by_category", bycat.round(3).reset_index())

    tot = bt.groupby(["method", "origin", "user_id", "month"], as_index=False)[["amount", "forecast"]].sum()
    total = score_backtest(tot)
    run.table("fc_total_level", total.round(4))

    bt[bt["method"] == "seasonal_naive"].set_index(["origin", "user_id", "category", "month"])["amount"]
    errs = {m: (bt[bt["method"] == m].set_index(["origin", "user_id", "category", "month"])
                .eval("abs(amount - forecast)")) for m in methods}
    rel = {m: float((errs[m] < errs["seasonal_naive"]).mean()) for m in methods}
    r["share_better_than_snaive"] = rel
    diff = (errs["lgbm"] - errs["ets"]).dropna()
    from scipy.stats import wilcoxon
    r["wilcoxon_lgbm_vs_ets"] = {"stat": float(wilcoxon(diff).statistic), "p": float(wilcoxon(diff).pvalue),
                                 "mean_diff": float(diff.mean())}
    diff2 = (errs["lgbm"] - errs["seasonal_naive"]).dropna()
    r["wilcoxon_lgbm_vs_snaive"] = {"stat": float(wilcoxon(diff2).statistic), "p": float(wilcoxon(diff2).pvalue),
                                    "mean_diff": float(diff2.mean())}

    _, ax = plt.subplots(figsize=(8, 4))
    for m in methods:
        g = byh[byh["method"] == m]
        ax.plot(g["h"], g["WAPE"], "o-", label=METHOD_LABELS[m])
    ax.set(xlabel="Горизонт h, міс.", ylabel="WAPE", xticks=[1, 2, 3], title="Похибка прогнозу залежно від горизонту")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    run.fig("fc_by_h")

    origin = panel["month"].max() - min(12, panel["month"].nunique() - min_train)
    train = panel[panel["month"] <= origin]
    from finapp.ml.forecasting import forecast_panel
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    top = panel.groupby(["user_id", "category"])["amount"].sum().sort_values(ascending=False).index[:2]
    for ax, (user, cat) in zip(axes, top):
        hist = panel[(panel["user_id"] == user) & (panel["category"] == cat)]
        ax.plot(hist["month"].dt.to_timestamp(), hist["amount"], "k.-", label="Факт")
        for m in ["seasonal_naive", "ets", "lgbm"]:
            steps = int(panel["month"].max().ordinal - origin.ordinal)
            fc = forecast_panel(train[(train["user_id"] == user) & (train["category"] == cat)], m, steps) if m != "lgbm" else \
                GlobalLGBMForecaster().fit(train).predict(train[(train["user_id"] == user) & (train["category"] == cat)], steps)
            ax.plot(fc["month"].dt.to_timestamp(), fc["forecast"], "--", label=METHOD_LABELS[m])
        ax.axvline(origin.to_timestamp(), color="grey", lw=1)
        ax.set(title=f"{user}: {label(cat)}", ylabel="грн/міс")
        ax.tick_params(axis="x", rotation=30)
    axes[0].legend(fontsize=8)
    run.fig("fc_examples")

    model = GlobalLGBMForecaster().fit(panel)
    imp = pd.Series(model.model.booster_.feature_importance("gain"), index=model.feature_names).sort_values()
    imp = imp / imp.sum()
    run.table("fc_importance", imp.round(4).rename("gain_share").reset_index())
    _fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp.index, imp.values, color="#4C72B0")
    ax.set(xlabel="Частка приросту (gain)", title="Важливість ознак глобальної LightGBM-моделі")
    run.fig("fc_importance")

    obj = []
    for o in ["l1", "l2", "huber"]:
        b = rolling_origin_backtest(panel, ["lgbm"], h=3, n_origins=n_origins, min_train=min_train,
                                    lgbm_params={"objective": o})
        obj.append({"objective": o, **score_backtest(b).iloc[0][["MAE", "WAPE"]].to_dict()})
    run.table("fc_objective", pd.DataFrame(obj).round(4))
    run.results["forecasting"] = r


def anomalies(run: Run, df: pd.DataFrame) -> None:
    r: dict = {}
    exp = expenses(df).dropna(subset=["category"])
    _, test_period, _ = time_split(exp)
    test = exp
    y = test["is_anomaly"].fillna(0).astype(int).to_numpy()
    r["anomalies"] = int(y.sum())
    if y.sum() == 0:
        zs = RobustZScoreDetector().fit(exp).score(test)
        r["flagged_z35"] = int((zs > 3.5).sum())
        run.table("an_top", test.assign(z=zs).nlargest(15, "z")[["datetime", "description", "category", "amount", "z"]].round(2))
        run.results["anomaly"] = r
        print("  немає еталонних аномалій (is_anomaly): метрики P/R пропущено, див. --inject-anomalies")
        return
    r["size"] = len(test)
    rows = []
    z = RobustZScoreDetector().fit(exp)
    zs = z.score(test).to_numpy()
    for k in [2.5, 3.0, 3.5, 4.0, 5.0]:
        p, rc, f, _ = precision_recall_fscore_support(y, zs > k, average="binary", zero_division=0)
        rows.append({"method": "robust_z", "param": k, "precision": p, "recall": rc, "f1": f, "flagged": int((zs > k).sum())})
    for c in [0.005, 0.01, 0.02, 0.05]:
        iso = IsolationForestDetector(contamination=c).fit(exp)
        pr = iso.predict(test).to_numpy()
        p, rc, f, _ = precision_recall_fscore_support(y, pr, average="binary", zero_division=0)
        rows.append({"method": "isolation_forest", "param": c, "precision": p, "recall": rc, "f1": f, "flagged": int(pr.sum())})
    res = pd.DataFrame(rows)
    run.table("an_results", res.round(4))

    train_p = exp[exp["datetime"] < test_period["datetime"].min()]
    yt = test_period["is_anomaly"].fillna(0).astype(int).to_numpy()
    zt = RobustZScoreDetector().fit(train_p).score(test_period).to_numpy() > 3.5
    p, rc, f, _ = precision_recall_fscore_support(yt, zt, average="binary", zero_division=0)
    r["holdout_z35"] = {"anomalies": int(yt.sum()), "precision": p, "recall": rc, "f1": f}

    noisy = test.assign(z=zs)
    fp = noisy[(noisy["z"] > 3.5) & (noisy["is_anomaly"] == 0)]
    run.table("an_false_positives", fp[["datetime", "user_id", "description", "category", "amount", "z"]].head(10).round(2))
    from sklearn.metrics import average_precision_score, precision_recall_curve
    iso = IsolationForestDetector(contamination=0.01).fit(exp)
    isc = iso.score(test).to_numpy()
    r["ap"] = {"robust_z": float(average_precision_score(y, zs)), "isolation_forest": float(average_precision_score(y, isc))}
    _, ax = plt.subplots(figsize=(6, 4.5))
    for name, s in [("Робастна z-оцінка", zs), ("Ізоляційний ліс", isc)]:
        p, rc, _ = precision_recall_curve(y, s)
        ax.plot(rc, p, label=name)
    ax.set(xlabel="Повнота", ylabel="Точність", title="Криві точність–повнота для виявлення аномалій")
    ax.legend()
    ax.grid(alpha=0.3)
    run.fig("an_pr")

    t = test.assign(z=zs, flag=zs > 3.5)
    _fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(t["datetime"], t["amount"], s=4, c=np.where(t["flag"], "C3", "C0"), alpha=0.6)
    ax.scatter(t.loc[t["is_anomaly"] == 1, "datetime"], t.loc[t["is_anomaly"] == 1, "amount"], s=40,
               facecolors="none", edgecolors="k", label="Внесені аномалії")
    ax.set(yscale="log", ylabel="Сума, грн", title="Транзакції користувачів: червоні — позначені детектором (z > 3,5)")
    ax.legend()
    run.fig("an_scatter")
    run.results["anomaly"] = r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEMO_DATASET))
    ap.add_argument("--out", default="Claude outputs/Курсова/results")
    ap.add_argument("--only", nargs="*", default=["eda", "cat", "fc", "an"])
    ap.add_argument("--inject-anomalies", type=float, default=0.0,
                    help="Частка витрат, які штучно збільшуються в 5–15 разів для оцінювання детекторів (реальні дані)")
    args = ap.parse_args()
    run = Run(Path(args.out))
    prev = Path(args.out) / "results.json"
    if prev.exists():
        run.results = json.loads(prev.read_text())
    raw = pd.read_csv(args.data, dtype={"transaction_id": str})
    df, report = load_transactions(args.data)
    run.results["labeled_share"] = float(expenses(df)["category"].notna().mean())
    if args.inject_anomalies and not df["is_anomaly"].fillna(0).any():
        rng = np.random.default_rng(0)
        idx = df.index[df["direction"] == "expense"]
        hit = rng.choice(idx, max(1, int(args.inject_anomalies * len(idx))), replace=False)
        df.loc[hit, "amount"] *= rng.uniform(5, 15, len(hit))
        df["is_anomaly"] = 0
        df.loc[hit, "is_anomaly"] = 1
        run.results["injected_anomalies"] = len(hit)
    filled, n_filled = fill_missing_categories(df)
    run.results["filled_by_model"] = n_filled
    steps = {"eda": lambda: eda(run, raw, filled, report), "cat": lambda: categorization(run, df),
             "fc": lambda: forecasting(run, filled), "an": lambda: anomalies(run, filled)}
    for s in args.only:
        t0 = time.perf_counter()
        print(f"[{s}] ...")
        steps[s]()
        run.save()
        print(f"[{s}] {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
