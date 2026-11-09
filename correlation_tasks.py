"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

import pandas as pd


FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
TARGET = "MRI_Count"


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """

    df = pd.read_csv(data.csv_path, sep="\t", na_values=["NA"])
    women = df[df["Gender"] == "Female"]
    men = df[df["Gender"] == "Male"]
    women_corr = women[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).to_dict()
    men_corr = men[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).to_dict()

    strongest_mri_feature = max(
        FEATURES,
        key=lambda f: max(abs(women_corr[f]), abs(men_corr[f])),
    )

    return BrainCorrelationSummary(
        men_count=len(men),
        women_count=len(women),
        women_mri_correlation=women_corr,
        men_mri_correlation=men_corr,
        strongest_mri_feature=strongest_mri_feature,
    )
