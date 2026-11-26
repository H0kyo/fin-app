from __future__ import annotations

CATEGORIES: dict[str, str] = {
    "groceries": "Продукти",
    "cafe": "Кафе та ресторани",
    "transport": "Транспорт і таксі",
    "fuel": "Пальне",
    "utilities": "Комунальні послуги",
    "telecom": "Мобільний зв'язок та інтернет",
    "subscriptions": "Цифрові підписки",
    "health": "Здоров'я та аптеки",
    "clothing": "Одяг і взуття",
    "entertainment": "Розваги",
    "home": "Дім і техніка",
    "transfers": "Перекази",
    "cash": "Зняття готівки",
    "donations": "Благодійність і донати",
    "other": "Інше",
}

CATEGORY_IDS: list[str] = list(CATEGORIES)


def label(category_id: str) -> str:
    return CATEGORIES.get(category_id, category_id)
