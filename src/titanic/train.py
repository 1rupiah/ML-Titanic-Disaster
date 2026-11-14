"""Train, compare and select a model; persist the winner and all metrics.

Usage:
    python -m titanic.train
"""

from __future__ import annotations

import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
)

from titanic import __version__, config, plots
from titanic.data import load_data
from titanic.features import add_group_features
from titanic.models import COMPLEXITY, get_candidates

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("train")

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}
RAW_COLUMNS = ["Sex", "Pclass", "Name", "Age", "Fare", "SibSp", "Parch",
               "Ticket", "Cabin", "Embarked", "TicketGroupSize"]
DISPLAY_NAMES = {
    "Name": "Name (title, surname)",
    "Ticket": "Ticket (travel group)",
    "Cabin": "Cabin (deck)",
    "TicketGroupSize": "Ticket group size",
}


def gender_rule_accuracy(train: pd.DataFrame) -> float:
    """The classic 'all women survive' heuristic, a strong domain baseline."""
    return float(((train["Sex"] == "female").astype(int) == train[config.TARGET]).mean())


def compare_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = RepeatedStratifiedKFold(
        n_splits=config.CV_SPLITS, n_repeats=config.CV_REPEATS,
        random_state=config.RANDOM_STATE,
    )
    rows = {}
    for name, pipe in get_candidates().items():
        t0 = time.time()
        scores = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
        row = {}
        for m in SCORING:
            row[f"{m}_mean"] = float(np.mean(scores[f"test_{m}"]))
            row[f"{m}_std"] = float(np.std(scores[f"test_{m}"]))
        row["fit_seconds"] = float(np.mean(scores["fit_time"]))
        rows[name] = row
        log.info("%-28s acc=%.4f +/- %.4f  auc=%.4f  (%.1fs)", name,
                 row["accuracy_mean"], row["accuracy_std"], row["roc_auc_mean"],
                 time.time() - t0)
    return pd.DataFrame(rows).T


def out_of_fold(pipe, X, y):
    """Out of fold probabilities and per fold permutation importance."""
    skf = StratifiedKFold(config.CV_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    proba = np.zeros(len(y))
    imps = []
    for tr, va in skf.split(X, y):
        model = pipe.fit(X.iloc[tr], y.iloc[tr])
        proba[va] = model.predict_proba(X.iloc[va])[:, 1]
        Xva = X.iloc[va]
        pi = permutation_importance(
            model, Xva, y.iloc[va], scoring="accuracy",
            n_repeats=10, random_state=config.RANDOM_STATE, n_jobs=-1,
        )
        cols = list(Xva.columns)
        imps.append([pi.importances_mean[cols.index(c)] for c in RAW_COLUMNS])
    imps = np.array(imps)
    imp_df = pd.DataFrame({"mean": imps.mean(0), "std": imps.std(0)}, index=RAW_COLUMNS)
    imp_df = imp_df.rename(index=DISPLAY_NAMES)
    return proba, imp_df


def select_one_std_rule(results: pd.DataFrame) -> str:
    """Pick the simplest model whose CV accuracy is within one std of the best.

    Lesson from v1: a 0.840 CV Gradient Boosting scored 0.744 on Kaggle. On small
    data, CV differences of 1 point are noise, and simpler models transfer better.
    """
    best = results["accuracy_mean"].idxmax()
    floor = results.loc[best, "accuracy_mean"] - results.loc[best, "accuracy_std"]
    eligible = results[results["accuracy_mean"] >= floor].copy()
    eligible["complexity"] = [COMPLEXITY[n] for n in eligible.index]
    return eligible.sort_values(["complexity", "accuracy_mean"],
                                ascending=[True, False]).index[0]


def main() -> None:
    train, test = load_data()
    train, test = add_group_features(train, test)
    X, y = train.drop(columns=[config.TARGET]), train[config.TARGET]
    log.info("Train shape %s, positive rate %.3f", X.shape, y.mean())

    plots.eda_figures(train)

    results = compare_models(X, y)
    gender_acc = gender_rule_accuracy(train)
    log.info("%-28s acc=%.4f", "Gender rule baseline", gender_acc)

    selectable = results.drop(index="Baseline (majority class)")
    best_cv_name = selectable["accuracy_mean"].idxmax()
    best_name = select_one_std_rule(selectable)
    log.info("Highest CV model: %s | Selected (one standard error rule): %s",
             best_cv_name, best_name)
    best_pipe = get_candidates()[best_name]

    proba, imp = out_of_fold(best_pipe, X, y)
    oof_pred = (proba >= 0.5).astype(int)

    plots.model_comparison(results, best_name)
    plots.oof_diagnostics(y.values, proba, best_name)
    plots.feature_importance(imp)
    plots.threshold_curve(y.values, proba)
    plots.cv_vs_leaderboard()

    final = get_candidates()[best_name].fit(X, y)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, config.MODEL_FILE)
    joblib.dump(get_candidates()[best_cv_name].fit(X, y), config.MODEL_BEST_CV_FILE)

    metrics = {
        "project_version": __version__,
        "selected_model": best_name,
        "selection_rule": "simplest model within one std of the best CV accuracy",
        "highest_cv_model": best_cv_name,
        "cv_scheme": (
            f"RepeatedStratifiedKFold({config.CV_SPLITS} folds x {config.CV_REPEATS} repeats)"
        ),
        "baselines": {
            "majority_class_accuracy": float(
                results.loc["Baseline (majority class)", "accuracy_mean"]
            ),
            "gender_rule_accuracy": gender_acc,
        },
        "selected_model_cv": {k: float(v) for k, v in results.loc[best_name].items()},
        "out_of_fold": {
            "accuracy": float(accuracy_score(y, oof_pred)),
            "f1": float(f1_score(y, oof_pred)),
            "roc_auc": float(roc_auc_score(y, proba)),
        },
        "all_models_cv": results.round(4).to_dict(orient="index"),
        "permutation_importance": imp.round(4).to_dict(orient="index"),
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    results.round(4).to_csv(config.REPORTS_DIR / "model_comparison.csv")
    log.info("Saved model to %s and metrics to %s", config.MODEL_FILE, config.METRICS_FILE)


if __name__ == "__main__":
    main()
