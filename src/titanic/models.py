"""Model zoo: every candidate is a full Pipeline (features, preprocessing, estimator)."""

from __future__ import annotations

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
    return cands
