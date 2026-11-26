"""FinApp — модуль категоризації транзакцій та прогнозування витрат.

uv run --group ml streamlit run src/finapp_app/app.py
"""

from __future__ import annotations

import io
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.metrics import confusion_matrix

from finapp.ml import DEMO_DATASET, sources
from finapp.ml.anomaly import IsolationForestDetector, RobustZScoreDetector
from finapp.ml.categories import CATEGORIES, CATEGORY_IDS, label
from finapp.ml.categorizer import CascadeCategorizer
from finapp.ml.data import DataValidationError, expenses, load_transactions, monthly_spend, validate_transactions
from finapp.ml.forecasting import forecast_panel, plan_backtest, rolling_origin_backtest, score_backtest
from finapp.ml.metrics import classification_summary, threshold_report
from finapp.ml.monobank_export import account_label
from finapp.ml.rules import apply_rules
from finapp.mono_backfill import select_accounts
from finapp.monobank import Client

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

MODELS = {"logreg": "Логістична регресія", "linsvc": "LinearSVC + калібрування", "lgbm": "LightGBM + калібрування"}
RULE_MODES = {"feature": "Правило як ознака моделі", "override": "Правило має пріоритет", "off": "Без правил"}
FC_METHODS = {"lgbm": "LightGBM (глобальна)", "ets": "ETS", "seasonal_naive": "Сезонний наївний",
              "moving_average": "Ковзне середнє", "naive": "Наївний"}
DATA_SOURCES = {"file": "Файл", "mono": "Monobank API", "demo": "Демо", "databricks": "Databricks"}
SOURCE_LABELS = {"manual": "користувач", "rule": "правило", "ml": "модель"}
LABEL_TO_ID = {v: k for k, v in CATEGORIES.items()}
MIN_LABELS = 50
LOCAL_LABELS_PATH = Path("data/private/transactions_labeled.csv")

st.set_page_config(page_title="FinApp — категоризація та прогноз", page_icon="💳", layout="wide")


@st.cache_data(show_spinner="Завантаження та перевірка даних…")
def load(content: bytes, name: str):
    df, report = load_transactions(io.BytesIO(content))
    return df, report.__dict__


@st.cache_resource(show_spinner="Навчання моделі категоризації…")
def train_categorizer(df: pd.DataFrame, model: str, rule_mode: str, threshold: float):
    exp = expenses(df).dropna(subset=["category"])
    cut = exp["datetime"].quantile(0.8)
    train, test = exp[exp["datetime"] <= cut], exp[exp["datetime"] > cut]
    cz = CascadeCategorizer(model_name=model, rule_mode=rule_mode, threshold=threshold,
                            model_params={"c": 1.0 if model == "logreg" else 10.0}).fit(train)
    final = CascadeCategorizer(model_name=model, rule_mode=rule_mode, threshold=threshold,
                               model_params={"c": 1.0 if model == "logreg" else 10.0}).fit(exp)
    return cz, final, train, test


def enough_labels(df: pd.DataFrame) -> bool:
    lab = expenses(df)["category"].dropna()
    return len(lab) >= MIN_LABELS and lab.nunique() >= 2


def with_labels(df: pd.DataFrame) -> pd.DataFrame:
    labels = st.session_state.get("labels", {})
    if not labels:
        return df
    df = df.copy()
    hit = df["transaction_id"].isin(labels) & (df["direction"] == "expense")
    df.loc[hit, "category"] = df.loc[hit, "transaction_id"].map(labels)
    return df


@st.cache_resource(show_spinner="Категоризація нерозмічених витрат…")
def completed(df: pd.DataFrame) -> pd.DataFrame:
    """Мітки користувача; для решти витрат — модель (якщо міток достатньо) або правила."""
    out = df.copy()
    exp_idx = out.index[out["direction"] == "expense"]
    todo = exp_idx[out.loc[exp_idx, "category"].isna()]
    out["category_origin"] = np.where(out["category"].notna(), "користувач", None)
    out["suggestion_conf"] = np.nan
    if len(todo) == 0:
        return out
    if enough_labels(df):
        cz = CascadeCategorizer(model_name="logreg", model_params={"c": 1.0}).fit(expenses(df).dropna(subset=["category"]))
        pred, conf, _ = cz.predict_model(out.loc[todo])
        out.loc[todo, "category"], out.loc[todo, "suggestion_conf"] = pred, conf
        out.loc[todo, "category_origin"] = "модель"
    else:
        rules = apply_rules(out.loc[todo])["rule_category"]
        out.loc[todo, "category"] = rules.fillna("other")
        out.loc[todo, "category_origin"] = np.where(rules.notna(), "правило", "без категорії")
    return out


def source_file() -> tuple[bytes, str]:
    up = st.sidebar.file_uploader("Набір транзакцій (CSV або JSON-виписка Monobank)", type=["csv", "json"])
    if up is None:
        st.info("Завантажте файл із транзакціями на бічній панелі: CSV або JSON-виписку Monobank.")
        st.stop()
    return up.getvalue(), up.name


def mono_client(token: str) -> Client:
    if st.session_state.get("mono_client_token") != hash(token):
        st.session_state["mono_client"] = Client(token)
        st.session_state["mono_client_token"] = hash(token)
    return st.session_state["mono_client"]


def mono_token() -> str:
    sb = st.sidebar
    token = sources.load_token()
    if token is None:
        if not sources.keychain_available():
            sb.error("Keychain недоступний: задайте токен у змінній середовища MONO_TOKEN.")
            st.stop()
        with sb.form("mono_token"):
            entered = st.text_input("Особистий токен Monobank", type="password",
                                    help="Отримати на api.monobank.ua. Зберігається лише в Keychain цього комп'ютера.")
            submitted = st.form_submit_button("Зберегти в Keychain")
        if submitted and entered.strip():
            sources.save_token(entered)
            st.rerun()
        st.info("Вставте особистий токен Monobank на бічній панелі.")
        st.stop()
    origin = sources.token_origin()
    c = sb.columns([3, 2])
    c[0].caption("🔑 Токен: " + ("змінна MONO_TOKEN" if origin == "env" else "Keychain"))
    if origin == "keychain" and c[1].button("Забути"):
        sources.forget_token()
        for k in ("mono_client", "mono_client_token", "mono_info", "mono_synced"):
            st.session_state.pop(k, None)
        st.rerun()
    return token


def mono_sync(client: Client, accounts: list[dict], store: pd.DataFrame | None) -> pd.DataFrame | None:
    now = datetime.now(UTC)
    since = sources.sync_start(store, now)
    with st.status(f"Monobank: операції з {since.astimezone():%d.%m.%Y %H:%M}…", expanded=True) as s:
        try:
            new = sources.fetch_statements(client, accounts, since, now,
                                           on_account=lambda a: s.write(f"Картка {account_label(a)}"))
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else "?"
            s.update(label=f"Monobank відповів помилкою {code}", state="error")
            if code in (401, 403):
                st.sidebar.error("Токен недійсний — натисніть «Забути» і введіть новий.")
            return store
        except requests.RequestException as e:
            s.update(label=f"Немає з'єднання з Monobank: {type(e).__name__}", state="error")
            return store
        if new.empty:
            s.update(label="Нових операцій немає", state="complete", expanded=False)
            return store
        new, _ = validate_transactions(new)
        merged, added = sources.merge_transactions(store, new.drop(columns=["description_norm"]))
        sources.write_store(merged)
        s.update(label=f"Додано нових операцій: {added}", state="complete", expanded=False)
    return merged


def source_mono() -> tuple[bytes, str]:
    sb = st.sidebar
    token = mono_token()
    client = mono_client(token)
    prefs = sources.read_prefs()
    store = sources.read_store()

    with sb.expander("Картки", expanded=not prefs["accounts"]):
        if st.button("Отримати список карток" if "mono_info" not in st.session_state else "Оновити список карток"):
            try:
                st.session_state["mono_info"] = client.get_client_info()
            except requests.RequestException as e:
                st.error(f"Не вдалося отримати картки: {type(e).__name__}")
        info = st.session_state.get("mono_info")
        if info:
            available = {a["id"]: a for a in select_accounts(info, None, include_jars=False) or info.get("accounts", [])}
            saved = [i for i in (a["id"] for a in prefs["accounts"]) if i in available]
            chosen = st.multiselect("Гривневі картки для синхронізації", list(available),
                                    format_func=lambda i: account_label(available[i]), default=saved or list(available))
            if st.button("Зберегти вибір"):
                prefs["accounts"] = [available[i] for i in chosen]
                sources.write_prefs(prefs)
                st.rerun()
        elif prefs["accounts"]:
            st.caption(", ".join(account_label(a) for a in prefs["accounts"]))

    with sb.expander("Початкова історія", expanded=store is None):
        st.caption("Історію зручніше взяти з CSV-виписки додатку, а API довантажить лише нові операції.")
        seed = st.file_uploader("CSV або JSON-виписка", type=["csv", "json"], key="mono_seed")
        if seed is not None and st.button("Додати до історії"):
            try:
                hist, _ = load_transactions(io.BytesIO(seed.getvalue()))
            except DataValidationError as e:
                st.error(f"Помилка даних: {e}")
            else:
                store, added = sources.merge_transactions(store, hist.drop(columns=["description_norm"]))
                sources.write_store(store)
                st.success(f"Додано операцій: {added}")

    if not prefs["accounts"]:
        st.info("Оберіть картки Monobank для синхронізації на бічній панелі.")
        st.stop()

    auto = sb.toggle("Оновлювати при відкритті", value=prefs.get("auto", True))
    if auto != prefs.get("auto", True):
        sources.write_prefs({**prefs, "auto": auto})
    since = sources.sync_start(store, datetime.now(UTC))
    minutes = sources.statement_requests(since, datetime.now(UTC)) * len(prefs["accounts"]) - 1
    manual = sb.button("Оновити з Monobank", type="primary", use_container_width=True,
                       help=f"З {since.astimezone():%d.%m.%Y}; ліміт API — 1 запит/хв, орієнтовно {max(minutes, 0)} хв")
    if manual or (auto and not st.session_state.get("mono_synced")):
        st.session_state["mono_synced"] = True
        store = mono_sync(client, prefs["accounts"], store)

    if store is None or store.empty:
        st.info("Локальна історія порожня: додайте виписку або оновіть дані з Monobank.")
        st.stop()
    sb.caption(f"Локально: {len(store)} операцій, остання — {pd.Timestamp(store['datetime'].max()):%d.%m.%Y %H:%M}")
    return sources.SYNC_PATH.read_bytes(), f"monobank-{sources.SYNC_PATH.stat().st_mtime_ns}"


def source_demo() -> tuple[bytes, str]:
    st.sidebar.caption("Синтетичний набір: 5 користувачів, 01.2023–12.2025, схема Monobank API + ручні записи.")
    return DEMO_DATASET.read_bytes(), "demo"


@st.cache_data(ttl=600, show_spinner="Читання bronze.mono_raw з Databricks…")
def databricks_csv() -> bytes:
    return sources.read_databricks().to_csv(index=False).encode()


def source_databricks() -> tuple[bytes, str]:
    if st.sidebar.button("Перечитати з Databricks"):
        databricks_csv.clear()
    try:
        return databricks_csv(), "databricks"
    except Exception as e:  # noqa: BLE001
        st.error(f"Не вдалося прочитати дані з Databricks: {e}")
        st.stop()


def sidebar():
    st.sidebar.title("💳 FinApp")
    st.sidebar.caption("Категоризація транзакцій і прогнозування витрат")
    kinds = ["file", "mono", "demo"] + (["databricks"] if sources.databricks_available() else [])
    kind = st.sidebar.radio("Джерело даних", kinds, format_func=DATA_SOURCES.get, horizontal=True)
    content, name = {"file": source_file, "mono": source_mono, "demo": source_demo,
                     "databricks": source_databricks}[kind]()
    try:
        df, report = load(content, name)
    except DataValidationError as e:
        st.error(f"Помилка даних: {e}")
        st.stop()
    users = sorted(df["user_id"].unique())
    user = st.sidebar.selectbox("Користувач", ["Усі"] + users)
    return df, report, user


def tab_data(df: pd.DataFrame, report: dict) -> None:
    c = st.columns(5)
    exp = expenses(df)
    c[0].metric("Записів після очищення", f"{report['rows_out']:,}".replace(",", " "), f"-{report['rows_in'] - report['rows_out']}")
    c[1].metric("Користувачів", df["user_id"].nunique())
    c[2].metric("Період, міс.", df["datetime"].dt.to_period("M").nunique(),
                help=f"{df['datetime'].min():%d.%m.%Y} – {df['datetime'].max():%d.%m.%Y}")
    c[3].metric("Витрати, грн", f"{exp['amount'].sum():,.0f}".replace(",", " "))
    c[4].metric("З мітками категорій", f"{exp['category'].notna().mean():.0%}")
    if report["dropped"] or report["fixed"]:
        with st.expander("Звіт про очищення даних", expanded=True):
            st.dataframe(pd.DataFrame(
                [{"Дія": "Видалено", "Причина": k, "Кількість": v} for k, v in report["dropped"].items()]
                + [{"Дія": "Виправлено", "Причина": k, "Кількість": v} for k, v in report["fixed"].items()]),
                hide_index=True)
    st.dataframe(df.drop(columns=["description_norm"]).head(500), hide_index=True, height=380)


def tab_eda(exp: pd.DataFrame) -> None:
    exp = exp.assign(Категорія=exp["category"].map(label).fillna("Без категорії"),
                     Місяць=exp["datetime"].dt.to_period("M").dt.to_timestamp())
    left, right = st.columns([2, 1])
    monthly = exp.groupby(["Місяць", "Категорія"], as_index=False)["amount"].sum()
    monthly["Місяць"] = monthly["Місяць"].dt.strftime("%Y-%m")
    bars = go.Figure([go.Bar(x=g["Місяць"], y=g["amount"], name=c) for c, g in monthly.groupby("Категорія")])
    bars.update_layout(barmode="stack", bargap=0.15, title="Витрати за місяцями", yaxis_title="грн",
                       legend_title="Категорія")
    bars.update_xaxes(type="category", nticks=12)
    left.plotly_chart(bars, use_container_width=True)
    share = exp.groupby("Категорія", as_index=False)["amount"].sum()
    pie = px.pie(share, names="Категорія", values="amount", hole=0.45, title="Структура витрат")
    pie.update_traces(textposition="inside", textinfo="percent", insidetextorientation="horizontal")
    pie.update_layout(uniformtext_minsize=10, uniformtext_mode="hide", legend={"font": {"size": 11}})
    right.plotly_chart(pie, use_container_width=True)
    left, right = st.columns(2)
    left.plotly_chart(px.histogram(exp.assign(lg=np.log10(exp["amount"])), x="lg", nbins=60, title="Розподіл сум (log10 грн)",
                                   labels={"lg": "log10(сума, грн)", "count": "Кількість"}), use_container_width=True)
    hw = exp.groupby([exp["datetime"].dt.weekday, exp["datetime"].dt.hour]).size().unstack(fill_value=0)
    right.plotly_chart(px.imshow(hw, labels={"x": "Година", "y": "День тижня", "color": "К-сть"},
                                 y=["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"][: len(hw)], aspect="auto",
                                 color_continuous_scale="Blues", title="Активність за днем і годиною"),
                       use_container_width=True)
    stats = exp.groupby("Категорія")["amount"].agg(["count", "mean", "median", "std", "min", "max"]).round(1)
    stats.columns = ["Кількість", "Середнє", "Медіана", "Ст. відхилення", "Мін", "Макс"]
    st.subheader("Описові статистики за категоріями")
    st.dataframe(stats.sort_values("Кількість", ascending=False))


def tab_labeling(df: pd.DataFrame) -> None:
    labels: dict = st.session_state.setdefault("labels", {})
    exp = expenses(df)
    sugg = completed(df)
    sugg = sugg.loc[sugg["category_origin"] != "користувач", ["source", "transaction_id", "category", "suggestion_conf"]]
    exp = exp.merge(sugg.rename(columns={"category": "suggestion", "suggestion_conf": "conf"}),
                    on=["source", "transaction_id"], how="left")
    exp["suggestion"] = exp["suggestion"].map(label, na_action="ignore")
    c = st.columns(4)
    c[0].metric("Розмічено витрат", f"{exp['category'].notna().sum()} з {len(exp)}")
    c[1].metric("Частка", f"{exp['category'].notna().mean():.0%}")
    c[2].metric("Категорій у мітках", exp["category"].nunique())
    c[3].metric("Нерозмічених торговців", exp.loc[exp["category"].isna(), "description_norm"].nunique())
    st.caption(f"Підказки дають правила, а після {MIN_LABELS} міток — модель, навчена на ваших мітках. Для чесного "
               "оцінювання краще обирати категорію самостійно, а не механічно приймати підказку.")
    left, mid, right = st.columns(3)
    unit = left.radio("Одиниця розмітки", ["Торговець", "Транзакція"], horizontal=True,
                      help="Торговець: категорія застосовується до всіх операцій з однаковим нормалізованим описом.")
    only_new = mid.toggle("Лише нерозмічені", value=True)
    prefill = right.toggle("Підставити підказки в порожні клітинки", value=False)
    options = [CATEGORIES[c] for c in CATEGORY_IDS]

    if unit == "Торговець":
        g = exp.groupby("description_norm", as_index=False).agg(
            Опис=("description", "first"), MCC=("mcc", "first"), Операцій=("amount", "size"), Сума=("amount", "sum"),
            Категорія=("category", lambda s: label(s.mode().iloc[0]) if s.notna().any() else None),
            Підказка=("suggestion", lambda s: s.mode().iloc[0] if s.notna().any() else None))
        key_col = "description_norm"
    else:
        g = exp.assign(Опис=exp["description"], MCC=exp["mcc"], Сума=exp["amount"], Дата=exp["datetime"],
                       Категорія=exp["category"].map(label), Підказка=exp["suggestion"])
        g = g[["transaction_id", "Дата", "Опис", "MCC", "Сума", "Категорія", "Підказка"]]
        key_col = "transaction_id"
    if only_new:
        g = g[g["Категорія"].isna()]
    g = g.sort_values("Операцій" if unit == "Торговець" else "Дата", ascending=unit != "Транзакція")
    if unit == "Торговець":
        g = g.sort_values("Операцій", ascending=False)
    n = st.slider("Показати рядків", 10, max(10, min(500, len(g))), min(50, max(10, len(g))), 10) if len(g) > 10 else len(g)
    g = g.head(n).reset_index(drop=True)
    if prefill:
        g["Категорія"] = g["Категорія"].fillna(g["Підказка"])
    edited = st.data_editor(
        g, hide_index=True, height=420, key=f"editor_{unit}",
        column_order=[c for c in g.columns if c != key_col],
        disabled=[c for c in g.columns if c != "Категорія"],
        column_config={"Категорія": st.column_config.SelectboxColumn(options=options, required=False),
                       "Сума": st.column_config.NumberColumn(format="%.2f")})
    if st.button("Зберегти розмітку", type="primary"):
        chosen = edited.dropna(subset=["Категорія"])
        mapping = dict(zip(chosen[key_col], chosen["Категорія"].map(LABEL_TO_ID)))
        hit = exp[exp[key_col].isin(mapping)]
        for tid, k in zip(hit["transaction_id"], hit[key_col]):
            labels[tid] = mapping[k]
        st.success(f"Збережено мітки для {len(hit)} операцій")
        st.rerun()

    labeled = df.drop(columns=["description_norm"])
    st.divider()
    a, b = st.columns(2)
    a.download_button("Завантажити розмічений набір (CSV)", labeled.to_csv(index=False).encode("utf-8-sig"),
                      "transactions_labeled.csv", "text/csv")
    if b.button(f"Зберегти у {LOCAL_LABELS_PATH}"):
        LOCAL_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        labeled.to_csv(LOCAL_LABELS_PATH, index=False)
        b.success("Збережено")


def tab_categorization(df: pd.DataFrame, user: str) -> None:
    if not enough_labels(df):
        st.info(f"Для навчання й оцінювання моделі потрібно щонайменше {MIN_LABELS} розмічених витрат двох і більше "
                "категорій. Розмітьте операції на вкладці «Розмітка».")
        return
    c = st.columns(3)
    model = c[0].selectbox("Модель", list(MODELS), format_func=MODELS.get)
    mode = c[1].selectbox("Використання правил", list(RULE_MODES), format_func=RULE_MODES.get)
    theta = c[2].slider("Поріг впевненості θ", 0.3, 0.99, 0.7, 0.01)
    if model == "lgbm":
        st.caption("Навчання LightGBM з калібруванням триває близько хвилини.")
    cz, final, train, test = train_categorizer(df, model, mode, theta)

    ev = test[test["source"] != "manual"]
    pred, conf, _ = cz.predict_model(ev)
    s = classification_summary(ev["category"], pred, conf)
    t = threshold_report(ev["category"], pred, conf, theta)
    st.markdown(f"**Оцінювання на відкладеному періоді** (навчання до {train['datetime'].max():%d.%m.%Y}, "
                f"тест — {len(ev)} транзакцій з банківської виписки)")
    m = st.columns(5)
    m[0].metric("Accuracy", f"{s['accuracy']:.3f}")
    m[1].metric("Макро-F1", f"{s['macro_f1']:.3f}")
    m[2].metric("ECE", f"{s['ece']:.3f}")
    m[3].metric("Частка авто (≥ θ)", f"{t['coverage']:.1%}")
    m[4].metric("Точність авто", f"{t['accuracy_auto']:.3f}")

    labels = sorted(set(ev["category"]) | set(pred))
    cm = confusion_matrix(ev["category"], pred, labels=labels, normalize="true")
    with st.expander("Матриця помилок"):
        st.plotly_chart(px.imshow(cm, x=[label(x) for x in labels], y=[label(x) for x in labels], text_auto=".2f",
                                  color_continuous_scale="Blues", labels={"x": "Передбачено", "y": "Факт"}),
                        use_container_width=True)

    st.subheader("Категоризація всіх витрат")
    exp = expenses(df)
    if user != "Усі":
        exp = exp[exp["user_id"] == user]
    out = final.predict(exp)
    res = exp[["datetime", "user_id", "description", "mcc", "amount", "category"]].join(out)
    res["Категорія"] = res["pred_category"].map(label)
    res["Джерело"] = res["category_source"].map(SOURCE_LABELS)
    review = st.toggle("Лише ті, що потребують підтвердження (впевненість < θ)")
    view = res[~res["auto_assigned"]] if review else res
    st.dataframe(view[["datetime", "user_id", "description", "mcc", "amount", "Категорія", "Джерело", "confidence"]]
                 .rename(columns={"datetime": "Дата", "user_id": "Користувач", "description": "Опис", "amount": "Сума",
                                  "confidence": "Впевненість"}),
                 hide_index=True, height=360,
                 column_config={"Впевненість": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
    st.caption(f"Потребують підтвердження: {(~res['auto_assigned']).sum()} з {len(res)}")
    st.download_button("Завантажити результат (CSV)", res.to_csv(index=False).encode("utf-8-sig"),
                       "categorized.csv", "text/csv")


def tab_forecast(df: pd.DataFrame, user: str) -> None:
    panel = monthly_spend(df)
    if panel.empty:
        st.warning("Немає витрат із категоріями для побудови рядів.")
        return
    c = st.columns(3)
    method = c[0].selectbox("Метод", list(FC_METHODS), format_func=FC_METHODS.get)
    h = c[1].slider("Горизонт, міс.", 1, 6, 3)
    users = sorted(panel["user_id"].unique())
    u = user if user != "Усі" else c[2].selectbox("Користувач для графіка", users)
    up = panel[panel["user_id"] == u]
    cats = up.groupby("category")["amount"].sum().sort_values(ascending=False).index.tolist()
    chosen = st.multiselect("Категорії", cats, default=cats[:3], format_func=label)
    try:
        fc = forecast_panel(panel, method, h)
    except ValueError as e:
        st.warning(f"{e}. Зараз історії — {panel['month'].nunique()} міс.: додайте виписку за довший період "
                   "або оберіть простіший метод.")
        return
    fig = go.Figure()
    for i, cat in enumerate(chosen):
        hist = panel[(panel["user_id"] == u) & (panel["category"] == cat)]
        f = fc[(fc["user_id"] == u) & (fc["category"] == cat)]
        color = px.colors.qualitative.Plotly[i % 10]
        fig.add_scatter(x=hist["month"].dt.to_timestamp(), y=hist["amount"], name=label(cat), line={"color": color})
        fig.add_scatter(x=f["month"].dt.to_timestamp(), y=f["forecast"], name=f"{label(cat)} — прогноз",
                        line={"color": color, "dash": "dash"})
    fig.update_layout(title=f"Історія та прогноз витрат ({u}, {FC_METHODS[method]})", yaxis_title="грн/міс")
    st.plotly_chart(fig, use_container_width=True)
    nxt = fc[fc["user_id"] == u].pivot_table(index="category", columns="month", values="forecast").round(0)
    nxt.index = nxt.index.map(label).rename("Категорія")
    nxt.columns = [str(x) for x in nxt.columns]
    nxt.loc["Разом"] = nxt.sum()
    st.subheader("Прогноз витрат за категоріями, грн")
    st.dataframe(nxt)
    if st.button("Оцінити точність методів (ковзна точка відліку)"):
        with st.spinner("Бектестинг…"):
            min_train, n_origins = plan_backtest(panel["month"].nunique(), h=3)
            bt = rolling_origin_backtest(panel, list(FC_METHODS), h=3, n_origins=n_origins, min_train=min_train)
            sc = score_backtest(bt)
            sc["method"] = sc["method"].map(FC_METHODS)
            st.dataframe(sc.round(3).sort_values("WAPE").rename(columns={"method": "Метод"}), hide_index=True)


def tab_anomalies(df: pd.DataFrame, user: str) -> None:
    exp = expenses(df).dropna(subset=["category"])
    c = st.columns(2)
    method = c[0].radio("Метод", ["Робастна z-оцінка", "Ізоляційний ліс"], horizontal=True)
    if method == "Робастна z-оцінка":
        k = c[1].slider("Поріг z", 2.0, 6.0, 3.5, 0.1)
        det = RobustZScoreDetector(threshold=k).fit(exp)
    else:
        k = c[1].slider("Очікувана частка аномалій", 0.001, 0.05, 0.005, 0.001, format="%.3f")
        det = IsolationForestDetector(contamination=k).fit(exp)
    view = exp if user == "Усі" else exp[exp["user_id"] == user]
    flags = det.predict(view)
    score = det.score(view)
    v = view.assign(Аномалія=flags.map({True: "так", False: "ні"}), score=score, Категорія=view["category"].map(label))
    st.plotly_chart(px.scatter(v, x="datetime", y="amount", color="Аномалія", log_y=True, hover_data=["description", "Категорія"],
                               color_discrete_map={"так": "#d62728", "ні": "#9ecae1"}, title="Транзакції та виявлені аномалії",
                               labels={"datetime": "Дата", "amount": "Сума, грн"}), use_container_width=True)
    st.subheader(f"Позначені транзакції: {int(flags.sum())}")
    st.dataframe(v[flags][["datetime", "user_id", "description", "Категорія", "amount", "score"]]
                 .sort_values("score", ascending=False).round(2)
                 .rename(columns={"datetime": "Дата", "user_id": "Користувач", "description": "Опис", "amount": "Сума, грн",
                                  "score": "Оцінка аномальності"}), hide_index=True)


def main() -> None:
    raw, report, user = sidebar()
    df = with_labels(raw)
    full = completed(df)
    view = full if user == "Усі" else full[full["user_id"] == user]
    tabs = st.tabs(["📄 Дані", "📊 Аналіз", "✍️ Розмітка", "🏷️ Категоризація", "📈 Прогноз", "⚠️ Аномалії"])
    with tabs[0]:
        tab_data(df if user == "Усі" else df[df["user_id"] == user], report)
    with tabs[1]:
        tab_eda(expenses(view))
    with tabs[2]:
        tab_labeling(df)
    with tabs[3]:
        tab_categorization(df, user)
    with tabs[4]:
        tab_forecast(full, user)
    with tabs[5]:
        tab_anomalies(full, user)


main()
