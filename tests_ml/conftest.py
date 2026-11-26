import warnings

import numpy as np
import pandas as pd
import pytest

from finapp.ml.data import validate_transactions
from finapp.ml.synthetic import generate

warnings.filterwarnings("ignore")
np.seterr(all="ignore")


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    df, _ = validate_transactions(generate(start="2023-01-01", end="2024-12-31", dirty=False, seed=7))
    return df


def tx(**overrides) -> dict:
    row = {"transaction_id": "t1", "user_id": "u1", "datetime": "2025-01-15 12:00:00", "amount": -100.0,
           "description": "Сільпо", "mcc": 5411, "source": "monobank", "category": "groceries"}
    return row | overrides
