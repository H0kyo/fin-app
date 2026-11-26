from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["transaction_id", "user_id", "datetime", "amount", "description"]
OPTIONAL_COLUMNS = ["currency", "mcc", "source", "direction", "category", "is_anomaly"]
VALID_SOURCES = {"monobank", "manual", "privat"}
VALID_DIRECTIONS = {"expense", "income"}

ISO4217 = {980: "UAH", 840: "USD", 978: "EUR"}


class DataValidationError(ValueError):
    pass


@dataclass
class LoadReport:
    rows_in: int = 0
    rows_out: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    fixed: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str, n: int) -> None:
        if n:
            self.dropped[reason] = self.dropped.get(reason, 0) + int(n)

    def fix(self, reason: str, n: int) -> None:
        if n:
            self.fixed[reason] = self.fixed.get(reason, 0) + int(n)


_DIGITS = re.compile(r"\d+")
_NON_WORD = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACES = re.compile(r"\s+")
_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "`": "'", "‘": "'"})


def normalize_description(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.translate(_APOSTROPHES).lower().replace("'", "")
    t = _DIGITS.sub(" ", t)
    t = _NON_WORD.sub(" ", t).replace("_", " ")
    return _SPACES.sub(" ", t).strip()


def _read_any(source: str | Path | IO) -> pd.DataFrame:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise DataValidationError(f"Файл не знайдено: {path}")
        if path.stat().st_size == 0:
            raise DataValidationError("Файл порожній")
        if path.suffix.lower() == ".json":
            return from_monobank_statement(json.loads(path.read_text(encoding="utf-8")))
        source = io.BytesIO(path.read_bytes())
    raw = source.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8-sig")
    if not raw.strip():
        raise DataValidationError("Файл порожній")
    if raw.lstrip().startswith("["):
        return from_monobank_statement(json.loads(raw))
    if is_monobank_csv(raw):
        return from_monobank_csv(raw)
    return pd.read_csv(io.StringIO(raw), dtype={"transaction_id": str, "user_id": str})


MONOBANK_CSV_HEADERS = ("Дата i час операції", "Дата і час операції", "Date and time")
_CARD_CURRENCY = re.compile(r"\(([A-Z]{3})\)")


def is_monobank_csv(raw: str) -> bool:
    first = raw.lstrip().split("\n", 1)[0].lstrip('"')
    return first.startswith(MONOBANK_CSV_HEADERS)


def from_monobank_csv(raw: str, user_id: str = "me") -> pd.DataFrame:
    src = pd.read_csv(io.StringIO(raw), dtype=str, keep_default_na=False)
    if src.shape[1] < 10:
        raise DataValidationError("Невідомий формат CSV-виписки Monobank: очікується 10 колонок")
    dt, desc, mcc, amount, balance = (src.iloc[:, i] for i in (0, 1, 2, 3, 9))
    m = _CARD_CURRENCY.search(src.columns[3])
    key = dt + "|" + amount + "|" + balance + "|" + desc
    return pd.DataFrame({
        "transaction_id": key.map(lambda k: "csv-" + hashlib.sha1(k.encode()).hexdigest()[:16]),
        "user_id": user_id,
        "datetime": dt,
        "amount": amount,
        "currency": m.group(1) if m else "UAH",
        "description": desc,
        "mcc": mcc,
        "source": "monobank",
        "direction": np.where(pd.to_numeric(amount, errors="coerce") < 0, "expense", "income"),
    })


def from_monobank_statement(items: list[dict], user_id: str = "user-1") -> pd.DataFrame:
    rows = [
        {
            "transaction_id": it["id"],
            "user_id": user_id,
            "datetime": pd.to_datetime(it["time"], unit="s"),
            "amount": it["amount"] / 100,
            "currency": ISO4217.get(it.get("currencyCode", 980), str(it.get("currencyCode"))),
            "description": it.get("description", ""),
            "mcc": it.get("mcc"),
            "source": "monobank",
        }
        for it in items
    ]
    return pd.DataFrame(rows)


def load_transactions(source: str | Path | IO) -> tuple[pd.DataFrame, LoadReport]:
    try:
        df = _read_any(source)
    except pd.errors.EmptyDataError as e:
        raise DataValidationError("Файл порожній") from e
    except (pd.errors.ParserError, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise DataValidationError(f"Не вдалося розібрати файл: {e}") from e
    return validate_transactions(df)


def parse_datetime(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    text = s.astype("string").str.strip()
    out = pd.to_datetime(text, errors="coerce", format="ISO8601")
    for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%d.%m.%Y"):
        rest = out.isna() & text.notna()
        if not rest.any():
            break
        out[rest] = pd.to_datetime(text[rest], errors="coerce", format=fmt)
    return out


def validate_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, LoadReport]:
    report = LoadReport(rows_in=len(df))
    if df.empty:
        raise DataValidationError("Набір даних не містить жодного запису")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Відсутні обов'язкові колонки: {', '.join(missing)}")

    df = df.copy()
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["datetime"] = parse_datetime(df["datetime"])
    bad = df["datetime"].isna()
    report.drop("некоректна дата", bad.sum())
    df = df[~bad]

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    bad = df["amount"].isna() | (df["amount"] == 0) | ~np.isfinite(df["amount"])
    report.drop("відсутня або нульова сума", bad.sum())
    df = df[~bad]

    direction = df["direction"].where(df["direction"].isin(VALID_DIRECTIONS))
    derived = np.where(df["amount"] < 0, "expense", "income")
    report.fix("напрям визначено за знаком суми", direction.isna().sum())
    df["direction"] = direction.fillna(pd.Series(derived, index=df.index))
    df["amount"] = df["amount"].abs()

    df["mcc"] = pd.to_numeric(df["mcc"], errors="coerce")
    bad_mcc = df["mcc"].notna() & ((df["mcc"] < 0) | (df["mcc"] > 9999) | (df["mcc"] % 1 != 0))
    report.fix("некоректний MCC замінено пропуском", bad_mcc.sum())
    df.loc[bad_mcc, "mcc"] = np.nan
    df["mcc"] = df["mcc"].astype("Int64")

    report.fix("порожній опис", df["description"].isna().sum())
    df["description"] = df["description"].fillna("").astype(str)
    df["currency"] = df["currency"].fillna("UAH").astype(str)
    df["source"] = df["source"].where(df["source"].isin(VALID_SOURCES), "monobank")
    df["transaction_id"] = df["transaction_id"].astype(str)
    df["user_id"] = df["user_id"].astype(str)
    df["category"] = df["category"].where(df["category"].notna(), None)
    df["is_anomaly"] = pd.to_numeric(df["is_anomaly"], errors="coerce")

    dup = df.duplicated(subset=["source", "transaction_id"], keep="first")
    report.drop("дублікат (source, transaction_id)", dup.sum())
    df = df[~dup]

    if df.empty:
        raise DataValidationError("Після очищення не залишилось жодного коректного запису")

    df["description_norm"] = df["description"].map(normalize_description)
    df = df.sort_values("datetime").reset_index(drop=True)
    report.rows_out = len(df)
    return df, report


def expenses(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["direction"] == "expense"].reset_index(drop=True)


def monthly_spend(df: pd.DataFrame, category_col: str = "category") -> pd.DataFrame:
    exp = expenses(df).dropna(subset=[category_col])
    exp = exp.assign(month=exp["datetime"].dt.to_period("M"))
    agg = exp.groupby(["user_id", category_col, "month"], observed=True)["amount"].sum()
    out = []
    for (user, cat), s in agg.groupby(level=[0, 1]):
        s = s.droplevel([0, 1])
        idx = pd.period_range(exp["month"].min(), exp["month"].max(), freq="M")
        s = s.reindex(idx, fill_value=0.0)
        out.append(pd.DataFrame({"user_id": user, "category": cat, "month": idx, "amount": s.values}))
    return pd.concat(out, ignore_index=True)
