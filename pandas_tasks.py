"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    df = pd.read_csv(data.csv_path)

    # 1. Пропуски в каждом столбце
    missing_by_column = df.isna().sum().to_dict()
    # 2. Пассажиры старше 30 лет
    adults_over_30_count = int((df["Age"] > 30).sum())
    # 3. Средний возраст по классам
    mean_age_by_pclass = (df.groupby("Pclass")["Age"].mean().to_dict())
    # 4. Доля выживших по классам
    survival_rate_by_pclass = (df.groupby("Pclass")["Survived"].mean().to_dict())
    # 5. Пять наибольших тарифов по убыванию
    highest_fares = (df["Fare"].sort_values(ascending=False).head(5).tolist())

    return TitanicSummary(
        row_count=len(df),
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares,
    )