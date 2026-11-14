import pandas as pd
import pytest

from titanic import config
from titanic.predict import EXPECTED_ROWS, validate_submission

IDS = pd.Series(range(892, 892 + EXPECTED_ROWS))


def _sub(**overrides):
    df = pd.DataFrame({"PassengerId": IDS, "Survived": 0})
    for k, v in overrides.items():
        df[k] = v
    return df


def test_valid_submission_passes():
    validate_submission(_sub(), IDS)


def test_extra_column_fails():
    with pytest.raises(ValueError):
        validate_submission(_sub(Extra=1), IDS)


def test_wrong_row_count_fails():
    with pytest.raises(ValueError):
        validate_submission(_sub().iloc[:-1], IDS)


def test_non_binary_fails():
    with pytest.raises(ValueError):
        validate_submission(_sub(Survived=2), IDS)


@pytest.mark.skipif(not config.SUBMISSION_FILE.exists(), reason="submission not generated")
def test_generated_submission_file_is_valid():
    sub = pd.read_csv(config.SUBMISSION_FILE)
    test = pd.read_csv(config.TEST_FILE) if config.TEST_FILE.exists() else None
    ids = test["PassengerId"] if test is not None else sub["PassengerId"]
    validate_submission(sub, ids)
