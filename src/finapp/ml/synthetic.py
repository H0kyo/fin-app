from __future__ import annotations

import string
from dataclasses import dataclass

import numpy as np
import pandas as pd

CITIES = ["KYIV", "Київ", "LVIV", "Львів", "Kyiv UA", "ODESA", "Бровари", ""]

MERCHANTS: dict[str, list[tuple[str, int | None, float]]] = {
    "groceries": [
        ("Сільпо", 5411, 3.0), ("SILPO {n} {city}", 5411, 1.5), ("АТБ", 5411, 3.0), ("ATB-MARKET {n}", 5411, 1.5),
        ("Novus", 5411, 1.0), ("NOVUS UKRAINA {city}", 5411, 0.7), ("Фора", 5411, 1.0), ("FORA {n}", 5411, 0.6),
        ("Varus", 5411, 0.6), ("Ашан", 5411, 0.8), ("AUCHAN {city}", 5411, 0.4), ("Таврія В", 5411, 0.3),
        ("Метро", 5411, 0.4), ("Близенько", 5499, 0.8), ("M'YASNA LAVKA {n}", 5499, 0.4),
        ("Хлібна хата", 5462, 0.4), ("Glovo Market", 5499, 0.4),
        ("Glovo", 5814, 0.5), ("LIQPAY*Ferma {n}", 7399, 0.4), ("Prom.ua", 5399, 0.2),
    ],
    "cafe": [
        ("McDonald's", 5814, 2.0), ("MCDONALDS {n} {city}", 5814, 1.0), ("Пузата Хата", 5812, 1.0),
        ("Aroma Kava", 5814, 1.2), ("AROMA KAVA {n}", 5814, 0.8), ("One Love espresso bar", 5814, 0.6),
        ("Львівські круасани", 5814, 0.8), ("LVIVSKI KRUASANY", 5814, 0.5), ("KFC {city}", 5814, 0.6),
        ("Glovo", 5814, 1.2), ("GLOVO*{n}", 5814, 0.6), ("Bolt Food", 5814, 0.7), ("Ресторан Кримська Перлина", 5812, 0.2),
        ("Starbucks", 5814, 0.3), ("Вареничная Катюша", 5812, 0.3), ("Coffee Point {n}", 5814, 0.4),
        ("Сільпо", 5411, 0.8), ("WFP*Kavarnia {n}", 7399, 0.3),
    ],
    "transport": [
        ("Uber", 4121, 2.0), ("UBER *TRIP", 4121, 1.2), ("Bolt", 4121, 2.0), ("BOLT.EU/O/{n}", 4121, 1.0),
        ("Uklon", 4121, 1.2), ("Київ Цифровий", 4111, 1.5), ("KYIV METRO", 4111, 0.8),
        ("Укрзалізниця", 4112, 0.6), ("UZ.GOV.UA", 4112, 0.4), ("Lviv transport", 4111, 0.4),
        ("BlaBlaCar", 4131, 0.2), ("FlixBus", 4131, 0.2), ("LIQPAY*Avtovokzal", 7399, 0.2),
    ],
    "fuel": [
        ("OKKO", 5541, 2.0), ("OKKO AZK {n}", 5541, 1.0), ("WOG", 5542, 2.0), ("WOG {n} {city}", 5542, 0.8),
        ("SOCAR", 5541, 1.0), ("UPG", 5541, 0.5), ("Shell", 5541, 0.5), ("AMIC ENERGY", 5541, 0.3),
    ],
    "utilities": [
        ("Київводоканал", 4900, 1.0), ("YASNO", 4900, 1.0), ("Київгаз Енерджи", 4900, 1.0),
        ("Нафтогаз України", 4900, 0.6), ("ОСББ Комфорт", 4900, 0.5), ("Комунальні послуги через Portmone", 4900, 0.5),
        ("Львівенергозбут", 4900, 0.3), ("Квартплата", 4900, 0.3),
    ],
    "telecom": [
        ("Kyivstar", 4814, 2.0), ("Київстар поповнення", 4814, 1.0), ("Vodafone", 4814, 1.5),
        ("lifecell", 4814, 1.0), ("Воля кабель", 4899, 0.5), ("Triolan", 4899, 0.4), ("Datagroup", 4899, 0.3),
    ],
    "subscriptions": [
        ("Netflix.com", 4899, 1.0), ("Spotify", 5815, 1.0), ("YouTube Premium", 5815, 0.8),
        ("APPLE.COM/BILL", 5818, 1.2), ("Google One", 5818, 0.6), ("OpenAI *ChatGPT", 5734, 0.6),
        ("MEGOGO", 4899, 0.5), ("Sweet.tv", 4899, 0.4), ("Steam purchase", 5816, 0.3),
    ],
    "health": [
        ("Аптека АНЦ", 5912, 1.5), ("APTEKA ANC {n}", 5912, 0.8), ("Подорожник", 5912, 1.0),
        ("Аптека 911", 5912, 0.8), ("Аптека Доброго Дня", 5912, 0.6), ("Медичний центр Добробут", 8062, 0.4),
        ("Synevo", 8071, 0.4), ("Сінево аналізи", 8071, 0.2), ("Стоматологія Дент", 8021, 0.3),
        ("Prom.ua", 5399, 0.2), ("Нова Пошта", 4215, 0.2), ("WFP*Clinic Med", 7399, 0.2),
    ],
    "clothing": [
        ("ZARA {city}", 5651, 1.0), ("H&M", 5651, 1.0), ("Reserved", 5651, 0.8), ("Intertop", 5661, 0.8),
        ("Sinsay", 5651, 0.6), ("Answear", 5691, 0.4), ("Lc Waikiki", 5651, 0.4), ("Puma Store", 5661, 0.3),
        ("Rozetka", 5311, 0.4), ("Нова Пошта", 4215, 0.6), ("Prom.ua", 5399, 0.6), ("Епіцентр", 5200, 0.2),
    ],
    "entertainment": [
        ("Multiplex", 7832, 1.0), ("Планета Кіно", 7832, 0.8), ("Karabas.com", 7922, 0.6), ("Concert.ua", 7922, 0.5),
        ("Боулінг Олімпік", 7933, 0.3), ("Книгарня Є", 5942, 0.6), ("Yakaboo", 5942, 0.5), ("PlayStation Store", 5816, 0.3),
        ("Rozetka", 5311, 0.3), ("Нова Пошта", 4215, 0.2), ("WFP*Kvytky {n}", 7399, 0.3), ("Prom.ua", 5399, 0.2),
    ],
    "home": [
        ("Епіцентр", 5200, 1.5), ("EPICENTR K {n}", 5200, 0.8), ("JYSK", 5712, 0.8), ("IKEA", 5712, 0.3),
        ("Rozetka", 5311, 1.2), ("ROZETKA.UA", 5732, 0.8), ("Comfy", 5732, 0.6), ("Фокстрот", 5732, 0.5),
        ("Eva", 5977, 0.8), ("Watsons", 5977, 0.4), ("Prostor", 5977, 0.5), ("Нова Пошта", 4215, 0.6),
        ("Prom.ua", 5399, 0.6), ("АТБ", 5411, 0.3),
    ],
    "transfers": [
        ("Переказ на картку", 4829, 1.0), ("Переказ: {name}", 4829, 1.5), ("Олена К.", 4829, 0.4),
        ("З чорної картки", 4829, 0.3), ("На банку «На відпустку»", 4829, 0.5), ("Поповнення Payoneer", 6012, 0.1),
        ("Переказ через Приват24", 4829, 0.3),
    ],
    "cash": [
        ("Банкомат PRIVATBANK", 6011, 1.0), ("ATM OSCHADBANK {city}", 6011, 0.8), ("Банкомат Укргазбанк", 6011, 0.4),
        ("Готівка в касі", 6010, 0.2),
    ],
}

NEW_MERCHANTS: dict[str, list[tuple[str, int | None, float]]] = {
    "groceries": [("Маркет Свій {n}", 5499, 1.0), ("Lotok", 5411, 1.0), ("SPAR {city}", 5411, 1.0)],
    "cafe": [("Кав'ярня Зерно", 5814, 1.0), ("Pizza Celentano", 5812, 1.0), ("Chicken Hut", 5814, 1.0)],
    "transport": [("OnTaxi", 4121, 1.0), ("Електросамокат Jet", 4121, 1.0)],
    "health": [("Аптека Бажаємо здоров'я", 5912, 1.0), ("Медікавер", 8062, 1.0)],
    "clothing": [("Mango", 5651, 1.0), ("Massimo Dutti", 5651, 1.0)],
    "entertainment": [("Кінотеатр Жовтень", 7832, 1.0), ("Nebo VR park", 7999, 1.0)],
    "home": [("Нова Лінія", 5200, 1.0), ("Brabrabra", 5719, 1.0)],
    "telecom": [("Ланет", 4899, 1.0)],
}
NEW_MERCHANTS_FROM = "2025-06-01"
NEW_MERCHANTS_SHARE = 0.3

MANUAL_TEXTS: dict[str, list[str]] = {
    "groceries": ["ринок овочі", "продукти на тиждень", "хліб і молоко", "фрукти з базару", "кіоск біля дому"],
    "cafe": ["кава біля роботи", "обід з колегами", "шаурма", "кава", "піца з друзями"],
    "transport": ["маршрутка", "таксі додому", "квиток на автобус", "трамвай"],
    "health": ["аптека біля дому", "ліки", "лікар приватно"],
    "clothing": ["куртка на ринку", "шкарпетки", "взуття ремонт"],
    "entertainment": ["квитки в кіно готівкою", "подарунок другу", "книга з розкладки"],
    "home": ["господарські дрібниці", "лампочки", "засоби для прибирання"],
    "cash": [],
}

NAMES = ["Олена К.", "Андрій М.", "Ірина П.", "Mykola S.", "Тарас Б.", "Наталія В.", "Dmytro H."]

# (expected count per month, log-mean amount UAH, log-std)
CATEGORY_PROFILE: dict[str, tuple[float, float, float]] = {
    "groceries": (14.0, 5.7, 0.75),
    "cafe": (9.0, 5.0, 0.55),
    "transport": (8.0, 4.9, 0.6),
    "fuel": (3.0, 7.1, 0.35),
    "utilities": (3.0, 6.9, 0.55),
    "telecom": (1.3, 5.3, 0.3),
    "subscriptions": (0.0, 5.3, 0.4),
    "health": (1.8, 5.9, 0.8),
    "clothing": (0.8, 7.3, 0.6),
    "entertainment": (1.6, 6.0, 0.6),
    "home": (2.0, 6.6, 0.85),
    "transfers": (2.5, 6.8, 0.9),
    "cash": (0.7, 7.0, 0.5),
}

SUBSCRIPTION_PRICES = {
    "Netflix.com": 330.0, "Spotify": 170.0, "YouTube Premium": 140.0, "APPLE.COM/BILL": 99.0,
    "Google One": 85.0, "OpenAI *ChatGPT": 830.0, "MEGOGO": 249.0, "Sweet.tv": 129.0, "Steam purchase": 400.0,
}


def _season_count(category: str, month: int) -> float:
    if category == "clothing":
        return {3: 1.6, 4: 1.5, 9: 1.6, 10: 1.4, 11: 1.8, 12: 1.3}.get(month, 0.7)
    if category == "entertainment":
        return {12: 1.8, 1: 1.3, 7: 1.3, 8: 1.3}.get(month, 0.9)
    if category == "home":
        return {11: 1.6, 12: 1.8, 4: 1.2}.get(month, 0.9)
    if category in ("transport", "fuel"):
        return {7: 1.25, 8: 1.3, 12: 1.1}.get(month, 1.0)
    if category == "groceries":
        return {12: 1.2, 1: 1.05}.get(month, 1.0)
    return 1.0


def _season_amount(category: str, month: int) -> float:
    if category == "utilities":
        return {11: 1.6, 12: 2.0, 1: 2.2, 2: 1.9, 3: 1.4, 10: 1.2}.get(month, 0.85)
    if category == "groceries":
        return {12: 1.15}.get(month, 1.0)
    return 1.0


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    level: float
    has_car: bool
    activity: float
    subscriptions: tuple[str, ...]
    manual_share: float
    relabel: tuple[tuple[str, str, float], ...] = ()
    rent_to: str | None = None


USERS = [
    UserProfile("user-01", 0.0, True, 1.0, ("Netflix.com", "Spotify", "APPLE.COM/BILL"), 0.06),
    UserProfile("user-02", 0.25, False, 1.2, ("YouTube Premium", "OpenAI *ChatGPT", "Google One"), 0.04,
                (("Rozetka", "clothing", 0.5),), rent_to="Ірина П."),
    UserProfile("user-03", -0.3, False, 0.8, ("MEGOGO",), 0.15, (("Glovo", "groceries", 0.7),), rent_to="Тарас Б."),
    UserProfile("user-04", 0.4, True, 1.3, ("Netflix.com", "Spotify", "Sweet.tv", "Steam purchase"), 0.05),
    UserProfile("user-05", -0.1, True, 0.9, ("APPLE.COM/BILL", "YouTube Premium"), 0.10,
                (("Bolt Food", "groceries", 0.6),)),
]


def _fill(template: str, rng: np.random.Generator) -> str:
    return template.format(
        n=rng.integers(1, 999),
        city=rng.choice(CITIES),
        name=rng.choice(NAMES),
    ).strip()


def _noisy(text: str, rng: np.random.Generator) -> str:
    r = rng.random()
    if r < 0.15:
        return text.upper()
    if r < 0.22:
        return text.lower()
    if r < 0.27:
        return f"{text} {rng.integers(10000, 99999)}"
    return text


def _when(category: str, day: pd.Timestamp, rng: np.random.Generator) -> pd.Timestamp:
    if category == "cafe":
        hour = rng.choice([8, 9, 12, 13, 14, 18, 19, 20], p=[0.12, 0.12, 0.18, 0.18, 0.1, 0.1, 0.1, 0.1])
    elif category == "groceries":
        hour = rng.choice([10, 12, 17, 18, 19, 20, 21], p=[0.1, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1])
    elif category == "transport":
        hour = rng.choice([7, 8, 9, 17, 18, 19, 23], p=[0.15, 0.2, 0.1, 0.15, 0.2, 0.1, 0.1])
    elif category == "entertainment":
        hour = rng.choice([15, 18, 19, 20, 21], p=[0.15, 0.2, 0.25, 0.25, 0.15])
    else:
        hour = rng.integers(8, 22)
    return day + pd.Timedelta(hours=int(hour), minutes=int(rng.integers(0, 60)), seconds=int(rng.integers(0, 60)))


def _tx_id(rng: np.random.Generator) -> str:
    alphabet = np.array(list(string.ascii_letters + string.digits))
    return "".join(rng.choice(alphabet, 12))


def generate(
    start: str = "2023-01-01",
    end: str = "2025-12-31",
    users: list[UserProfile] | None = None,
    label_noise: float = 0.02,
    anomaly_rate: float = 0.006,
    dirty: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    users = users or USERS
    months = pd.period_range(start, end, freq="M")
    cats = list(CATEGORY_PROFILE)
    rows: list[dict] = []

    for u in users:
        for t, month in enumerate(months):
            first = month.start_time
            days = pd.date_range(first, month.end_time.normalize(), freq="D")
            inflation = 1.009**t
            m = month.month

            rows.append({
                "user_id": u.user_id, "datetime": first + pd.Timedelta(days=4, hours=10),
                "amount": round(float(np.exp(10.4 + u.level) * inflation * rng.uniform(0.97, 1.03)), 2),
                "description": "Зарплата ТОВ «Софт Девелопмент»", "mcc": 4829, "source": "monobank",
                "direction": "income", "category": None,
            })
            if rng.random() < 0.35:
                rows.append({
                    "user_id": u.user_id, "datetime": first + pd.Timedelta(days=int(rng.integers(0, 27)), hours=14),
                    "amount": round(float(rng.lognormal(7.5, 0.6)), 2), "description": f"Від: {rng.choice(NAMES)}",
                    "mcc": 4829, "source": "monobank", "direction": "income", "category": None,
                })

            if u.rent_to:
                rows.append({
                    "user_id": u.user_id, "datetime": first + pd.Timedelta(days=int(rng.integers(0, 3)), hours=11),
                    "amount": round(float(np.exp(9.2 + u.level) * (1.004**t) * rng.uniform(0.98, 1.02)), 2),
                    "description": f"Переказ: {u.rent_to}", "mcc": 4829, "source": "monobank",
                    "direction": "expense", "category": "utilities",
                })

            for sub in u.subscriptions:
                price = SUBSCRIPTION_PRICES[sub] * (1.0 + 0.004 * t)
                mcc = next(mc for d, mc, _ in MERCHANTS["subscriptions"] if d == sub)
                day = first + pd.Timedelta(days=sum(map(ord, sub)) % 25)
                rows.append({
                    "user_id": u.user_id, "datetime": day + pd.Timedelta(hours=3, minutes=int(rng.integers(0, 60))),
                    "amount": round(price, 2), "description": _noisy(sub, rng), "mcc": mcc, "source": "monobank",
                    "direction": "expense", "category": "subscriptions",
                })

            for cat in cats:
                count_mean, mu, sigma = CATEGORY_PROFILE[cat]
                if cat == "subscriptions" or (cat == "fuel" and not u.has_car):
                    continue
                if cat == "utilities":
                    n = 3
                else:
                    n = rng.poisson(count_mean * u.activity * _season_count(cat, m))
                merchants = MERCHANTS[cat]
                weights = np.array([w for _, _, w in merchants])
                if first >= pd.Timestamp(NEW_MERCHANTS_FROM) and cat in NEW_MERCHANTS:
                    new = NEW_MERCHANTS[cat]
                    new_w = np.array([w for _, _, w in new])
                    new_w = new_w / new_w.sum() * weights.sum() * NEW_MERCHANTS_SHARE / (1 - NEW_MERCHANTS_SHARE)
                    merchants, weights = merchants + new, np.concatenate([weights, new_w])
                for i in range(n):
                    manual = cat in MANUAL_TEXTS and MANUAL_TEXTS[cat] and rng.random() < u.manual_share
                    if cat == "utilities":
                        desc, mcc, _w = merchants[i % 3]
                        day = first + pd.Timedelta(days=int(10 + 4 * i + rng.integers(0, 3)))
                        amount = np.exp(mu + u.level * 0.3) * _season_amount(cat, m) * rng.uniform(0.75, 1.25)
                    else:
                        idx = rng.choice(len(merchants), p=weights / weights.sum())
                        desc, mcc, _w = merchants[idx]
                        desc = _fill(desc, rng)
                        day = days[rng.integers(0, len(days))]
                        amount = rng.lognormal(mu + u.level, sigma) * _season_amount(cat, m)
                    if manual:
                        desc, mcc = str(rng.choice(MANUAL_TEXTS[cat])), None
                    label = cat
                    for merchant, alt, p in u.relabel:
                        if desc.lower().startswith(merchant.lower()) and rng.random() < p:
                            label = alt
                    rows.append({
                        "user_id": u.user_id, "datetime": _when(cat, day, rng),
                        "amount": round(float(amount * inflation), 2),
                        "description": desc if manual else _noisy(desc, rng),
                        "mcc": mcc, "source": "manual" if manual else "monobank",
                        "direction": "expense", "category": label,
                    })

    df = pd.DataFrame(rows)
    df["currency"] = "UAH"
    df["is_anomaly"] = 0

    exp_idx = df.index[df["direction"] == "expense"]
    noisy = rng.choice(exp_idx, int(label_noise * len(exp_idx)), replace=False)
    df.loc[noisy, "category"] = rng.choice(cats, len(noisy))

    anom = rng.choice(exp_idx, int(anomaly_rate * len(exp_idx)), replace=False)
    df.loc[anom, "amount"] = (df.loc[anom, "amount"] * rng.uniform(5, 15, len(anom))).round(2)
    df.loc[anom, "is_anomaly"] = 1

    df["transaction_id"] = [_tx_id(rng) for _ in range(len(df))]
    df.loc[df["source"] == "manual", "transaction_id"] = [
        f"manual-{i:05d}" for i in range((df["source"] == "manual").sum())
    ]
    df["amount"] = np.where(df["direction"] == "expense", -df["amount"], df["amount"])

    if dirty:
        dup = df.sample(frac=0.003, random_state=seed)
        bad = df.sample(n=12, random_state=seed + 1).astype({"datetime": object, "mcc": object})
        bad.loc[bad.index[:4], "datetime"] = "31.02.2024"
        bad.loc[bad.index[4:8], "amount"] = np.nan
        bad.loc[bad.index[8:10], "mcc"] = 99999
        bad.loc[bad.index[10:], "description"] = None
        bad["transaction_id"] = [_tx_id(rng) for _ in range(len(bad))]
        df = pd.concat([df, dup, bad], ignore_index=True)

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["mcc"] = pd.to_numeric(df["mcc"], errors="coerce").astype("Int64")
    cols = ["transaction_id", "user_id", "datetime", "amount", "currency", "description", "mcc", "source",
            "direction", "category", "is_anomaly"]
    return df[cols]
