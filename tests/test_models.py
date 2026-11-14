import pandas as pd

from titanic.models import COMPLEXITY, WomanChildGroupRule, get_candidates
from titanic.train import select_one_std_rule


def test_every_candidate_has_a_complexity_rank():
    assert set(get_candidates()) <= set(COMPLEXITY)


def test_group_rule_overrides_gender_model():
    X = pd.DataFrame({
        "Sex": ["female", "female", "male", "male"],
        "Title": ["Mrs", "Miss", "Master", "Mr"],
        "WCGroupKnown": [1, 0, 1, 1],
        "WCGroupRate": [0.0, 0.5, 1.0, 1.0],
    })
    assert WomanChildGroupRule().fit(X, [0, 1, 1, 0]).predict(X).tolist() == [0, 1, 1, 0]


def test_one_std_rule_prefers_simpler_model_within_noise():
    res = pd.DataFrame(
        {"accuracy_mean": [0.85, 0.84], "accuracy_std": [0.02, 0.02]},
        index=["Soft Voting Ensemble", "Logistic Regression"],
    )
    assert select_one_std_rule(res) == "Logistic Regression"
