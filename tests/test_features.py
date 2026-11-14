import numpy as np
import pandas as pd
import pytest

from titanic import config
from titanic.features import FeatureEngineer, add_group_features, extract_title


@pytest.fixture
def raw():
    return pd.DataFrame({
        "PassengerId": [1, 2, 3, 4],
        "Pclass": [3, 1, 3, 2],
        "Name": [
            "Braund, Mr. Owen Harris",
            "Cumings, Mrs. John Bradley",
            "Palsson, Master. Gosta Leonard",
            "Rothes, the Countess. of",
        ],
        "Sex": ["male", "female", "male", "female"],
        "Age": [22.0, np.nan, 2.0, np.nan],
        "SibSp": [1, 1, 3, 0],
        "Parch": [0, 0, 1, 0],
        "Ticket": ["A", "B", "A", "C"],
        "Fare": [7.25, 71.28, np.nan, 26.0],
        "Cabin": [np.nan, "C85", np.nan, "T"],
        "Embarked": ["S", np.nan, "S", "C"],
    })


def test_extract_title_groups_rare_titles(raw):
    assert extract_title(raw["Name"]).tolist() == ["Mr", "Mrs", "Master", "Rare"]


def test_transform_leaves_no_missing_model_inputs(raw):
    out = FeatureEngineer().fit(raw).transform(raw)
    cols = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES + config.BINARY_FEATURES
    assert out[cols].isna().sum().sum() == 0


def test_engineered_values(raw):
    out = FeatureEngineer().fit(raw).transform(raw)
    assert out["FamilySize"].tolist() == [2, 2, 5, 1]
    assert out["IsAlone"].tolist() == [0, 0, 0, 1]
    assert out["FamilyType"].tolist() == ["Small", "Small", "Large", "Solo"]
    assert out["Deck"].tolist() == ["U", "C", "U", "U"]
    assert out.loc[[0, 2], "IsChild"].tolist() == [0, 1]


def test_imputer_uses_only_fit_data(raw):
    fe = FeatureEngineer().fit(raw.iloc[:2])
    unseen = raw.iloc[[3]].copy()
    out = fe.transform(unseen)
    assert out["Age"].iloc[0] == pytest.approx(fe.age_global_)


def test_group_features_do_not_need_target(raw):
    train, test = add_group_features(raw.iloc[:2], raw.iloc[2:])
    assert "Survived" not in train
    assert train["TicketGroupSize"].tolist() == [2, 1]
    assert test["TicketGroupSize"].tolist() == [2, 1]


def test_wc_group_rate_is_leave_one_out():
    # Two sisters on one ticket: each row must only see the OTHER sister's label.
    df = pd.DataFrame({
        "PassengerId": [10, 11, 12],
        "Pclass": [3, 3, 1],
        "Name": ["Smith, Miss. Ann", "Smith, Miss. Beth", "Jones, Mr. Carl"],
        "Sex": ["female", "female", "male"],
        "Age": [20.0, 22.0, 40.0],
        "SibSp": [1, 1, 0], "Parch": [0, 0, 0],
        "Ticket": ["X1", "X1", "Y9"],
        "Fare": [10.0, 10.0, 50.0],
        "Cabin": [np.nan, np.nan, np.nan],
        "Embarked": ["S", "S", "S"],
    })
    y = pd.Series([1, 0, 0])
    out = FeatureEngineer().fit(df, y).transform(df)
    assert out["WCGroupRate"].tolist() == [0.0, 1.0, 0.5]
    assert out["WCGroupKnown"].tolist() == [1, 1, 0]


def test_unlabelled_fit_has_no_group_memory(raw):
    out = FeatureEngineer().fit(raw).transform(raw)
    assert out["WCGroupKnown"].sum() == 0
