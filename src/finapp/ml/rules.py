from __future__ import annotations

import re

import numpy as np
import pandas as pd

MCC_RULES: dict[int, str] = {
    5411: "groceries", 5462: "groceries",
    5812: "cafe", 5814: "cafe",
    4111: "transport", 4112: "transport", 4121: "transport", 4131: "transport",
    5541: "fuel", 5542: "fuel",
    4900: "utilities",
    4814: "telecom",
    5815: "subscriptions", 5818: "subscriptions",
    5912: "health", 8021: "health", 8062: "health", 8071: "health",
    5651: "clothing", 5661: "clothing", 5691: "clothing",
    7832: "entertainment", 7922: "entertainment", 7933: "entertainment", 5942: "entertainment",
    5200: "home", 5712: "home",
    6010: "cash", 6011: "cash",
    8398: "donations",
}

KEYWORD_RULES: list[tuple[str, str]] = [
    (r"netflix|spotify|youtube premium|megogo|sweet\.?tv|google one|chatgpt|openai", "subscriptions"),
    (r"kyivstar|київстар|vodafone|lifecell|воля кабель|triolan|datagroup", "telecom"),
    (r"аптек|apteka|подорожник|synevo|сінево", "health"),
    (r"uber|uklon|bolt(?!\s*food)|метро|metro\b|таксі|маршрутк", "transport"),
    (r"okko|wog|socar|shell|upg\b", "fuel"),
    (r"сільпо|silpo|атб|atb|novus|фора|fora\b|varus|ашан|auchan|ринок|продукти", "groceries"),
    (r"кава|kava|coffee|mcdonald|пузата|kfc|glovo(?!\s*market)|bolt food|ресторан|кафе", "cafe"),
    (r"епіцентр|epicentr|jysk|ikea|comfy|фокстрот", "home"),
    (r"банкомат|atm\b|готівка", "cash"),
    (r"донат|збір на|благодійн|повернись живим|притул|united24|фонд", "donations"),
    (r"переказ|з чорної картки|на банку", "transfers"),
]
_COMPILED = [(re.compile(p, re.IGNORECASE), c) for p, c in KEYWORD_RULES]


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Повертає rule_category (або None) і rule_type ('mcc' | 'keyword' | None) для кожного рядка."""
    mcc_cat = df["mcc"].map(lambda m: MCC_RULES.get(int(m)) if pd.notna(m) else None)
    text = df["description"].fillna("")

    def kw(t: str) -> str | None:
        for rx, cat in _COMPILED:
            if rx.search(t):
                return cat
        return None

    kw_cat = text.map(kw)
    rule_cat = mcc_cat.where(mcc_cat.notna(), kw_cat)
    rule_type = np.where(mcc_cat.notna(), "mcc", np.where(kw_cat.notna(), "keyword", None))
    return pd.DataFrame({"rule_category": rule_cat, "rule_type": rule_type}, index=df.index)
