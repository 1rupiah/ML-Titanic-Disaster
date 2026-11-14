# Model Card: Titanic Survival Classifier v1.0

## Model details
* **Type:** Gradient Boosting classifier (scikit learn) inside a full preprocessing pipeline
* **Hyperparameters:** 300 trees, learning rate 0.05, max depth 3, subsample 0.8
* **Inputs:** raw Kaggle columns (`Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`,
  `Ticket`, `Fare`, `Cabin`, `Embarked`)
* **Output:** survival probability, thresholded at 0.5

## Intended use
Educational and portfolio demonstration of an end to end classification workflow on the
Kaggle Titanic competition. Not intended for decisions about real people.

## Training data
891 passengers from `train.csv`, 38.4% positive class.

## Evaluation
| Metric | Value |
|---|---|
| CV accuracy (5 fold x 3 repeats) | 0.840 ± 0.019 |
| CV ROC AUC | 0.881 |
| Out of fold accuracy | 0.846 |
| Out of fold precision / recall (survived) | 0.82 / 0.77 |
| Majority class baseline | 0.616 |
| "All women survive" baseline | 0.787 |

## Ethical considerations
The model relies heavily on sex, age and ticket class because those determined survival
in 1912. In any modern application, features like these would require a fairness review,
since a model can reproduce historical bias very accurately.

## Limitations
Small sample size, historical data, and a test distribution that differs slightly from
training. Expect the Kaggle leaderboard score to be somewhat below the CV estimate.
