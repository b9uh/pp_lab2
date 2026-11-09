from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
TARGET = "MRI_Count"


def load_brain(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, sep="\t", na_values=["NA"])


def plot_correlation_heatmaps(df: pd.DataFrame) -> None:
    women = df[df["Gender"] == "Female"]
    men = df[df["Gender"] == "Male"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(
        women[FEATURES + [TARGET]].corr(),
        annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
        ax=axes[0],
    )
    axes[0].set_title("Корреляции — женщины")

    sns.heatmap(
        men[FEATURES + [TARGET]].corr(),
        annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
        ax=axes[1],
    )
    axes[1].set_title("Корреляции — мужчины")

    plt.tight_layout()
    plt.show()


def plot_mri_scatter(df: pd.DataFrame) -> None:
    women = df[df["Gender"] == "Female"]
    men = df[df["Gender"] == "Male"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax, feature in zip(axes, FEATURES):
        ax.scatter(women[feature], women[TARGET],
                   alpha=0.7, label="Women", color="crimson")
        ax.scatter(men[feature], men[TARGET],
                   alpha=0.7, label="Men", color="steelblue")
        ax.set_xlabel(feature)
        ax.set_ylabel(TARGET)
        ax.set_title(f"{feature} vs {TARGET}")
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_correlation_bars(df: pd.DataFrame) -> None:
    women = df[df["Gender"] == "Female"]
    men = df[df["Gender"] == "Male"]

    women_corr = women[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET)
    men_corr = men[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET)

    x = range(len(FEATURES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], women_corr.values,
           width, label="Women", color="crimson")
    ax.bar([i + width / 2 for i in x], men_corr.values,
           width, label="Men", color="steelblue")

    ax.set_xticks(list(x))
    ax.set_xticklabels(FEATURES)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(f"Корреляция с {TARGET}")
    ax.set_title(f"Корреляция признаков с {TARGET} по полу")
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    csv_path = "brainsize.txt"
    df = load_brain(csv_path)

    plot_correlation_heatmaps(df)
    plot_mri_scatter(df)
    plot_correlation_bars(df)