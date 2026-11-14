"""All figures used in the README. Consistent style, saved as PNG."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from titanic import config

PALETTE = ["#C0392B", "#2E86C1"]
sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)


def _save(fig, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def eda_figures(train: pd.DataFrame) -> None:
    df = train.copy()
    df["Outcome"] = df["Survived"].map({0: "Died", 1: "Survived"})
    df["Title"] = df["Name"].str.extract(r",\s*([^.]+)\.", expand=False)
    df["Title"] = df["Title"].where(df["Title"].isin(["Mr", "Mrs", "Miss", "Master"]), "Other")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, col, title in zip(
        axes, ["Sex", "Pclass", "Title"], ["by sex", "by ticket class", "by title"],
        strict=True,
    ):
        rate = df.groupby(col)["Survived"].mean().sort_values(ascending=False)
        sns.barplot(x=rate.index.astype(str), y=rate.values, ax=ax, color="#2E86C1")
        ax.set_title(f"Survival rate {title}")
        ax.set_ylabel("Survival rate")
        ax.set_xlabel("")
        ax.set_ylim(0, 1)
        for i, v in enumerate(rate.values):
            ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=10)
    _save(fig, "01_survival_by_group.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    sns.histplot(data=df, x="Age", hue="Outcome", bins=30, kde=True,
                 palette=PALETTE, ax=axes[0], element="step")
    axes[0].set_title("Age distribution by outcome")
    sns.boxplot(data=df, x="Pclass", y="Fare", hue="Outcome", palette=PALETTE,
                ax=axes[1], showfliers=False)
    axes[1].set_title("Fare by class and outcome")
    _save(fig, "02_age_fare.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    miss = train.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    sns.barplot(x=miss.values, y=miss.index, ax=ax, color="#7F8C8D")
    ax.set_title("Missing values in train.csv")
    ax.set_xlabel("Share missing")
    for i, v in enumerate(miss.values):
        ax.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=10)
    _save(fig, "03_missing_values.png")


def model_comparison(results: pd.DataFrame) -> None:
    res = results.sort_values("accuracy_mean")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2E86C1" if i == len(res) - 1 else "#AEB6BF" for i in range(len(res))]
    ax.barh(res.index, res["accuracy_mean"], xerr=res["accuracy_std"],
            color=colors, capsize=4)
    for i, (m, s) in enumerate(zip(res["accuracy_mean"], res["accuracy_std"], strict=True)):
        ax.text(m + s + 0.005, i, f"{m:.3f}", va="center", fontsize=10)
    ax.set_xlim(0.5, 0.92)
    ax.set_xlabel("Cross validated accuracy (5 fold x 3 repeats)")
    ax.set_title("Model comparison")
    _save(fig, "04_model_comparison.png")


def oof_diagnostics(y: np.ndarray, proba: np.ndarray, model_name: str) -> None:
    pred = (proba >= 0.5).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, display_labels=["Died", "Survived"], cmap="Blues",
        ax=axes[0], colorbar=False,
    )
    axes[0].set_title("Out of fold confusion matrix")
    axes[0].grid(False)
    RocCurveDisplay.from_predictions(y, proba, ax=axes[1], name=model_name)
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1].set_title("Out of fold ROC curve")
    _save(fig, "05_confusion_roc.png")


def feature_importance(imp: pd.DataFrame) -> None:
    imp = imp.sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp.index, imp["mean"], xerr=imp["std"], color="#2E86C1", capsize=3)
    ax.set_xlabel("Drop in accuracy when the column is shuffled")
    ax.set_title("Permutation importance (raw input columns)")
    _save(fig, "06_feature_importance.png")


def threshold_curve(y: np.ndarray, proba: np.ndarray) -> None:
    ts = np.linspace(0.1, 0.9, 81)
    prec, rec, acc = [], [], []
    for t in ts:
        p = proba >= t
        tp = (p & (y == 1)).sum()
        prec.append(tp / max(p.sum(), 1))
        rec.append(tp / (y == 1).sum())
        acc.append((p == y).mean())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ts, acc, label="Accuracy", lw=2.5)
    ax.plot(ts, prec, label="Precision (survived)", lw=2)
    ax.plot(ts, rec, label="Recall (survived)", lw=2)
    ax.axvline(0.5, color="k", ls="--", alpha=0.5)
    ax.set_xlabel("Decision threshold")
    ax.set_title("Trade off between precision and recall")
    ax.legend()
    _save(fig, "07_threshold_tradeoff.png")
