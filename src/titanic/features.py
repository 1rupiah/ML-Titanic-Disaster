"""Feature engineering.

Two stages:
1. ``add_group_features``: label free group statistics (ticket sharing) that
   need the full passenger manifest. They never touch the target, so no label
   leakage occurs.
2. ``FeatureEngineer``: a scikit learn transformer with row level features
   plus an Age imputer learned on training folds only, so it runs inside
   cross validation without leaking statistics from validation folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TITLE_MAP = {
    "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Capt": "Rare", "Sir": "Rare", "Don": "Rare", "Dona": "Rare",
    "Lady": "Rare", "Countess": "Rare", "Jonkheer": "Rare", "the Countess": "Rare",
}


def extract_title(name: pd.Series) -> pd.Series:
    raw = name.str.extract(r",\s*([^.]+)\.", expand=False).str.strip()
    return raw.map(TITLE_MAP).fillna("Rare")


def add_group_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add TicketGroupSize computed over the full manifest (train + test)."""
    counts = pd.concat([train["Ticket"], test["Ticket"]]).value_counts()
    train, test = train.copy(), test.copy()
    train["TicketGroupSize"] = train["Ticket"].map(counts).astype(int)
    test["TicketGroupSize"] = test["Ticket"].map(counts).astype(int)
    return train, test


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create model ready features from raw Titanic columns."""

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        X["Title"] = extract_title(X["Name"])
        self.age_by_title_ = X.groupby("Title")["Age"].median().to_dict()
        self.age_global_ = float(X["Age"].median())
        self.fare_by_class_ = X.groupby("Pclass")["Fare"].median().to_dict()
        self.embarked_mode_ = X["Embarked"].mode().iloc[0]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["Title"] = extract_title(X["Name"])

        missing_age = X["Age"].isna()
        X.loc[missing_age, "Age"] = (
            X.loc[missing_age, "Title"].map(self.age_by_title_).fillna(self.age_global_)
        )
        missing_fare = X["Fare"].isna()
        X.loc[missing_fare, "Fare"] = X.loc[missing_fare, "Pclass"].map(self.fare_by_class_)
        X["Embarked"] = X["Embarked"].fillna(self.embarked_mode_)

        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
        X["FamilyType"] = pd.cut(
            X["FamilySize"], bins=[0, 1, 4, np.inf], labels=["Solo", "Small", "Large"]
        ).astype(str)
        if "TicketGroupSize" not in X:
            X["TicketGroupSize"] = 1
        X["FarePerPerson"] = X["Fare"] / X["TicketGroupSize"].clip(lower=1)
        X["Deck"] = X["Cabin"].str[0].fillna("U").replace({"T": "U"})
        X["HasCabin"] = X["Cabin"].notna().astype(int)
        X["IsChild"] = (X["Age"] < 14).astype(int)
        X["Pclass"] = X["Pclass"].astype(str)
        return X
