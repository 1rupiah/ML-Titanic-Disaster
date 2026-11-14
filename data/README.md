# Data

The raw files come from the Kaggle competition
[Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)
and are not committed to this repository, in line with the competition rules.

Place these files in `data/raw/`:

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 891 | Passengers with the `Survived` label |
| `test.csv` | 418 | Passengers to predict |
| `gender_submission.csv` | 418 | Kaggle example submission |

Download with the Kaggle CLI:

```bash
kaggle competitions download -c titanic -p data/raw
cd data/raw && unzip titanic.zip
```
