"""Generate the Kaggle submission file from the trained model.

Usage:
    python -m titanic.predict
"""

from __future__ import annotations

import logging

import joblib
import pandas as pd

from titanic import config
from titanic.data import load_data
from titanic.features import add_group_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("predict")

EXPECTED_ROWS = 418


def validate_submission(sub: pd.DataFrame, test_ids: pd.Series) -> None:
    """Enforce the exact Kaggle contract before anything is written to disk."""
    if list(sub.columns) != [config.ID_COL, config.TARGET]:
        raise ValueError(f"Columns must be exactly {[config.ID_COL, config.TARGET]}")
    if len(sub) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, got {len(sub)}")
    if not set(sub[config.TARGET].unique()) <= {0, 1}:
        raise ValueError("Survived must contain only 0 and 1")
    if set(sub[config.ID_COL]) != set(test_ids):
        raise ValueError("PassengerId values do not match test.csv")


def main() -> None:
    if not config.MODEL_FILE.exists():
        raise FileNotFoundError("No trained model. Run `python -m titanic.train` first.")
    train, test = load_data()
    _, test = add_group_features(train, test)
    config.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(config.MODEL_FILE, config.SUBMISSION_FILE),
            (config.MODEL_BEST_CV_FILE, config.SUBMISSION_BEST_CV_FILE)]
    for model_path, out_path in jobs:
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        preds = model.predict(test).astype(int)
        sub = pd.DataFrame({config.ID_COL: test[config.ID_COL], config.TARGET: preds})
        validate_submission(sub, test[config.ID_COL])
        sub.to_csv(out_path, index=False)
        log.info("Wrote %d predictions to %s (predicted survival rate %.3f)",
                 len(sub), out_path, sub[config.TARGET].mean())


if __name__ == "__main__":
    main()
