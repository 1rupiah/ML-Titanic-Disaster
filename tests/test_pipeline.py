import pytest
from sklearn.model_selection import cross_val_score

from titanic import config
from titanic.data import load_data
from titanic.features import add_group_features
from titanic.models import get_candidates

pytestmark = pytest.mark.skipif(not config.TRAIN_FILE.exists(), reason="raw data not available")


def test_pipeline_beats_gender_baseline():
    train, test = load_data()
    train, _ = add_group_features(train, test)
    X, y = train.drop(columns=[config.TARGET]), train[config.TARGET]
    pipe = get_candidates()["Logistic Regression"]
    score = cross_val_score(pipe, X, y, cv=5, scoring="accuracy").mean()
    gender_rule = ((X["Sex"] == "female").astype(int) == y).mean()
    assert score > gender_rule
