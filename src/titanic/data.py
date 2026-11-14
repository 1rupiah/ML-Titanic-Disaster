"""Data loading and schema validation."""

from __future__ import annotations

import pandas as pd

from titanic import config

REQUIRED_COLUMNS = {
    "PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp",
    "Parch", "Ticket", "Fare", "Cabin", "Embarked",
}


def _validate(df: pd.DataFrame, name: str, require_target: bool) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if require_target and config.TARGET not in df.columns:
        missing.add(config.TARGET)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if df[config.ID_COL].duplicated().any():
        raise ValueError(f"{name} contains duplicated {config.ID_COL}")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw Kaggle files and validate their schema."""
    if not config.TRAIN_FILE.exists() or not config.TEST_FILE.exists():
        raise FileNotFoundError(
            "Raw data not found. Put train.csv and test.csv in data/raw/ "
            "(see data/README.md)."
        )
    train = pd.read_csv(config.TRAIN_FILE)
    test = pd.read_csv(config.TEST_FILE)
    _validate(train, "train.csv", require_target=True)
    _validate(test, "test.csv", require_target=False)
    return train, test
