# Model Card: Titanic Survival Classifier v2.0

## Model details
* **Type:** Woman/Child Group Rule inside a scikit learn pipeline (no learned weights;
  the only fitted state is imputation values and the group survival memory)
* **Logic:** women survive unless all other women/children in their travel group died;
  men die except boys whose group's women/children all survived
* **Selection:** simplest model within one standard deviation of the best CV accuracy
* **Alternative:** Soft Voting Ensemble (LR, SVM, RF, GB), exported separately

## Intended use
Educational and portfolio demonstration of an end to end classification workflow.
Not intended for decisions about real people.

## Evaluation
| Metric | Value |
|---|---|
| CV accuracy (5 fold x 3 repeats) | 0.841 ± 0.026 |
| CV precision / recall (survived) | 0.843 / 0.721 |
| Majority class baseline | 0.616 |
| "All women survive" baseline | 0.787 (Kaggle LB 0.76555) |
| **Kaggle leaderboard (418 passengers)** | **0.80143** (335 correct) |
| Soft Voting Ensemble, Kaggle LB | 0.78947 |
| v1.0 Gradient Boosting, Kaggle LB | 0.74401 |

## Ethical considerations
The model relies on sex, age and family outcomes because those determined survival in
1912. Comparable features in a modern system would require a fairness review, since a
model can reproduce historical bias very accurately.

## Limitations
Small sample, historical data, and a test set that differs from training. Passengers
travelling alone fall back to the gender rule.
