"""Model zoo: every candidate is a full Pipeline (features, preprocessing, estimator)."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from titanic import config
from titanic.features import FeatureEngineer


class WomanChildGroupRule(ClassifierMixin, BaseEstimator):
    """Gender model corrected by family outcomes (no parameters to overfit).

    * Women survive, unless every other woman/child in their group died.
    * Men die, except boys (Master) whose group's women/children all survived.

    Operates directly on the FeatureEngineer output.
    """

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        female = (X["Sex"] == "female").to_numpy()
        boy = (X["Title"] == "Master").to_numpy()
        known = X["WCGroupKnown"].to_numpy() == 1
        rate = X["WCGroupRate"].to_numpy()
        pred = female.astype(int)
        pred[female & known & (rate == 0)] = 0
        pred[boy & known & (rate == 1)] = 1
        return pred

    def predict_proba(self, X):
        # Soften hard rules with the group rate so ROC AUC is informative.
        p = np.where(X["WCGroupKnown"] == 1, X["WCGroupRate"], 0.5) * 0.2
        p = np.clip(0.8 * self.predict(X) + p, 0, 1)
        return np.column_stack([1 - p, p])


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             config.CATEGORICAL_FEATURES),
            ("bin", "passthrough", config.BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _estimators() -> dict:
    rs = config.RANDOM_STATE
    return {
        "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(C=0.5, max_iter=2000),
        "SVM (RBF)": CalibratedClassifierCV(SVC(C=1.0, gamma="scale"), ensemble=False),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=7, min_samples_leaf=2,
            max_features="sqrt", random_state=rs, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=rs,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=4,
            l2_regularization=1.0, random_state=rs,
        ),
    }


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("features", FeatureEngineer()),
        ("preprocess", build_preprocessor()),
        ("model", estimator),
    ])


# Lower number = fewer learned parameters = more robust to distribution shift.
COMPLEXITY = {
    "Baseline (majority class)": 0,
    "Woman/Child Group Rule": 1,
    "Logistic Regression": 2,
    "SVM (RBF)": 3,
    "Random Forest": 4,
    "HistGradientBoosting": 4,
    "Gradient Boosting": 4,
    "Soft Voting Ensemble": 5,
}


def get_candidates() -> dict[str, Pipeline]:
    cands = {name: make_pipeline(est) for name, est in _estimators().items()}
    est = _estimators()
    cands["Soft Voting Ensemble"] = make_pipeline(
        VotingClassifier(
            estimators=[
                ("lr", est["Logistic Regression"]),
                ("svm", est["SVM (RBF)"]),
                ("rf", est["Random Forest"]),
                ("gb", est["Gradient Boosting"]),
            ],
            voting="soft",
        )
    )
    cands["Woman/Child Group Rule"] = Pipeline([
        ("features", FeatureEngineer()),
        ("model", WomanChildGroupRule()),
    ])
    return cands
