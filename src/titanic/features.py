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


def group_keys(X: pd.DataFrame) -> pd.DataFrame:
    """Two keys that identify people travelling together.

    FamilyKey: surname + class + ticket stem + fare + port (separates unrelated
    passengers who share a common surname). TicketKey: identical ticket number.
    """
    surname = X["Name"].str.split(",").str[0].str.strip()
    stem = X["Ticket"].astype(str).str[:-1]
    fare = X["Fare"].round(2).astype(str)
    family = surname + "|" + X["Pclass"].astype(str) + "|" + stem + "|" + fare + "|" + X[
        "Embarked"
    ].astype(str)
    return pd.DataFrame({"FamilyKey": family, "TicketKey": X["Ticket"].astype(str)},
                        index=X.index)


def is_woman_or_child(X: pd.DataFrame) -> pd.Series:
    """Women and boys (title Master): the group protected by 'women and children first'."""
    return (X["Sex"] == "female") | (extract_title(X["Name"]) == "Master")


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

        # Woman/child group survival memory (target based, learned from fit data only).
        self.wc_members_ = {}
        if y is not None:
            keys = group_keys(X)
            wc = is_woman_or_child(X).to_numpy()
            ids = X["PassengerId"].to_numpy()
            ys = np.asarray(y)
            for col in ("FamilyKey", "TicketKey"):
                for key, pid, w, lab in zip(keys[col], ids, wc, ys, strict=True):
                    if w:
                        self.wc_members_.setdefault((col, key), {})[pid] = int(lab)
        return self

    def _wc_group_rate(self, X: pd.DataFrame) -> pd.Series:
        """Survival rate of the OTHER women/children in a passenger's group.

        Excluding the passenger's own id makes this leave one out on training rows,
        so a row never sees its own label. NaN when no labelled member is known.
        """
        keys = group_keys(X)
        rates = []
        for i, pid in enumerate(X["PassengerId"].to_numpy()):
            seen = {}
            for col in ("FamilyKey", "TicketKey"):
                seen.update(self.wc_members_.get((col, keys[col].iat[i]), {}))
            seen.pop(pid, None)
            rates.append(np.mean(list(seen.values())) if seen else np.nan)
        return pd.Series(rates, index=X.index, dtype=float)

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
        deck = X["Cabin"].astype("string").str[0].fillna("U")
        X["Deck"] = deck.replace({"T": "U"}).astype(str)
        X["HasCabin"] = X["Cabin"].notna().astype(int)
        X["IsChild"] = (X["Age"] < 14).astype(int)
        rate = self._wc_group_rate(X)
        X["WCGroupKnown"] = rate.notna().astype(int)
        X["WCGroupRate"] = rate.fillna(0.5)
        X["IsWomanOrChild"] = is_woman_or_child(X).astype(int)
        X["Pclass"] = X["Pclass"].astype(str)
        return X
