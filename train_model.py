#!/usr/bin/env python3
"""Treina e avalia modelos para previsao de resultados de partidas de Counter-Strike 2.

Uso:
    python train_model.py
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.cs2_pipeline import BASE_FEATURE_COLUMNS, HOLDOUT_SEASON_USAGE, TRAIN_SEASON_USAGE

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message=".*'penalty' was deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*penalty=l1 with l1_ratio=0.0.*", category=UserWarning)

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = ROOT / "data" / "processed" / "match_feature_differences.csv"
DEFAULT_SNAPSHOT_PATH = ROOT / "data" / "processed" / "team_snapshot.csv"
DEFAULT_MODELS_DIR = ROOT / "models"
DEFAULT_REPORTS_DIR = ROOT / "reports"
DEFAULT_LOGREG_CONFIG_PATH = ROOT / "reports" / "modelo_principal" / "config_modelo_principal.json"
DEFAULT_TRAINING_CONFIG_PATH = DEFAULT_LOGREG_CONFIG_PATH

LOGISTIC_FAMILY_BY_PENALTY = {
    "l1": "logreg_l1",
    "l2": "logreg_l2",
    "elasticnet": "logreg_elasticnet",
}
DEFAULT_SOLVER_BY_FAMILY = {
    "logreg_l1": "liblinear",
    "logreg_l2": "lbfgs",
    "logreg_elasticnet": "saga",
}


def normalize_class_weight(raw_value: object) -> str | None:
    if raw_value in {None, "", "none", "None"}:
        return None
    return str(raw_value)


def normalize_logreg_family(raw_family: object, raw_penalty: object) -> str:
    family_text = str(raw_family or "").strip().lower()
    if family_text in {"logreg_l1", "logreg_l2", "logreg_elasticnet"}:
        return family_text

    penalty_text = str(raw_penalty or "l2").strip().lower()
    return LOGISTIC_FAMILY_BY_PENALTY.get(penalty_text, "logreg_l2")


def load_logreg_tuning_config(config_path: Path = DEFAULT_LOGREG_CONFIG_PATH) -> dict[str, object] | None:
    if not config_path.exists():
        return None

    config = json.loads(config_path.read_text(encoding="utf-8"))
    feature_spec = config.get("features")
    if isinstance(feature_spec, str):
        feature_columns = [item.strip() for item in feature_spec.split(",") if item.strip()]
    elif isinstance(feature_spec, list):
        feature_columns = [str(item).strip() for item in feature_spec if str(item).strip()]
    else:
        feature_columns = []

    if not feature_columns:
        return None

    penalty = str(config.get("penalty", "l2")).strip().lower()
    model_family = normalize_logreg_family(config.get("model_family"), penalty)
    solver = str(config.get("solver") or DEFAULT_SOLVER_BY_FAMILY[model_family]).strip()
    class_weight = normalize_class_weight(config.get("class_weight"))

    resolved: dict[str, object] = {
        "feature_columns": feature_columns,
        "model_family": model_family,
        "penalty": penalty,
        "C": float(config.get("C", 1.0)),
        "class_weight": class_weight,
        "solver": solver,
    }
    if model_family == "logreg_elasticnet":
        resolved["l1_ratio"] = float(config.get("l1_ratio", 0.5))
    for key in [
        "description",
        "source_experiment_dir",
        "source_candidate_name",
        "selection_rule",
        "best_overall_candidate_name",
        "best_overall_model_family",
    ]:
        if key in config:
            resolved[key] = config[key]
    return resolved


def apply_logreg_tuning(
    feature_columns: list[str],
    tuning_config: dict[str, object] | None,
) -> tuple[list[str], dict[str, object] | None]:
    if tuning_config is None:
        return feature_columns, None

    requested = list(tuning_config.get("feature_columns", []))
    selected = [column for column in requested if column in feature_columns]
    missing = [column for column in requested if column not in feature_columns]
    if missing:
        raise ValueError(f"Configuracao de features da regressao logistica contem colunas ausentes: {missing}")
    if not selected:
        raise ValueError("Configuracao de features da regressao logistica nao possui colunas validas.")
    return selected, tuning_config


def build_logistic_regression_pipeline(logreg_overrides: dict[str, object] | None = None) -> Pipeline:
    logreg_kwargs: dict[str, object] = {
        "max_iter": 6000,
        "random_state": 42,
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
    }
    if logreg_overrides:
        logreg_kwargs.update(logreg_overrides)

    model_family = normalize_logreg_family(logreg_kwargs.get("model_family"), logreg_kwargs.get("penalty"))
    logreg_kwargs["model_family"] = model_family

    if "penalty" not in logreg_kwargs or not str(logreg_kwargs["penalty"]).strip():
        logreg_kwargs["penalty"] = {
            "logreg_l1": "l1",
            "logreg_l2": "l2",
            "logreg_elasticnet": "elasticnet",
        }[model_family]
    if "solver" not in logreg_kwargs or not str(logreg_kwargs["solver"]).strip():
        logreg_kwargs["solver"] = DEFAULT_SOLVER_BY_FAMILY[model_family]
    if normalize_class_weight(logreg_kwargs.get("class_weight")) is None:
        logreg_kwargs.pop("class_weight", None)
    else:
        logreg_kwargs["class_weight"] = normalize_class_weight(logreg_kwargs.get("class_weight"))
    if model_family != "logreg_elasticnet":
        logreg_kwargs.pop("l1_ratio", None)
    elif "l1_ratio" in logreg_kwargs:
        logreg_kwargs["l1_ratio"] = float(logreg_kwargs["l1_ratio"])
    logreg_kwargs.pop("model_family", None)

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**logreg_kwargs)),
        ]
    )


def build_models(logreg_overrides: dict[str, object] | None = None) -> dict[str, Pipeline]:
    return {
        "logistic_regression": build_logistic_regression_pipeline(logreg_overrides),
    }


def save_latex_table(
    df: pd.DataFrame,
    output_path: Path,
    caption: str,
    label: str,
    column_format: str | None = None,
) -> None:
    tabular = df.to_latex(
        index=False,
        escape=True,
        na_rep="-",
        float_format=lambda value: f"{value:.3f}",
        column_format=column_format,
    )
    output_path.write_text(
        "\n".join(
            [
                r"\begin{table}[H]",
                r"\centering",
                rf"\caption{{{caption}}}",
                rf"\label{{{label}}}",
                tabular,
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def save_feature_descriptive_stats(features: pd.DataFrame, tables_dir: Path) -> None:
    stats = (
        features.describe()
        .T[["mean", "std", "min", "max"]]
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    stats.to_csv(tables_dir / "feature_descriptive_stats.csv", index=False)

    latex_df = stats.rename(
        columns={
            "feature": "Atributo",
            "mean": "Media",
            "std": "Desvio-padrao",
            "min": "Minimo",
            "max": "Maximo",
        }
    )
    save_latex_table(
        latex_df,
        tables_dir / "feature_descriptive_stats.tex",
        "Estatisticas descritivas dos atributos do dataset supervisionado.",
        "tab:feature_stats",
        column_format="lrrrr",
    )


def save_model_metrics_tables(metrics: pd.DataFrame, tables_dir: Path) -> None:
    metrics.to_csv(tables_dir / "model_metrics.csv", index=False)

    latex_df = metrics[
        ["model", "cv_roc_auc_mean", "test_accuracy", "test_roc_auc", "test_log_loss", "test_brier"]
    ].rename(
        columns={
            "model": "Modelo",
            "cv_roc_auc_mean": "CV ROC-AUC",
            "test_accuracy": "Accuracy",
            "test_roc_auc": "ROC-AUC",
            "test_log_loss": "Log-loss",
            "test_brier": "Brier",
        }
    )
    save_latex_table(
        latex_df,
        tables_dir / "model_metrics.tex",
        "Resultados da avaliacao dos modelos, incluindo ROC-AUC medio de validacao cruzada no treino.",
        "tab:prelim_metrics",
        column_format="lccccc",
    )


def save_logistic_coefficients_table(model: Pipeline, feature_cols: list[str], tables_dir: Path) -> pd.DataFrame:
    coefficients = pd.DataFrame(
        {
            "feature": feature_cols,
            "coefficient": model.named_steps["clf"].coef_.ravel(),
        }
    ).sort_values("coefficient", ascending=False, kind="stable")
    coefficients.to_csv(tables_dir / "logistic_coefficients.csv", index=False)

    latex_df = coefficients.rename(columns={"feature": "Atributo", "coefficient": "Coeficiente"})
    save_latex_table(
        latex_df,
        tables_dir / "logistic_coefficients.tex",
        "Coeficientes da regressao logistica.",
        "tab:coef",
        column_format="lc",
    )
    return coefficients


def base_figure(width: float = 8.0, height: float = 5.0):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def save_class_balance_figure(target: pd.Series, figures_dir: Path) -> None:
    counts = target.value_counts().sort_index()
    fig, ax = base_figure(6.5, 4.5)
    ax.bar(["Derrota (0)", "Vitoria (1)"], counts.values, color=["#b0bec5", "#00897b"])
    ax.set_ylabel("Numero de confrontos")
    ax.set_title("Distribuicao da variavel alvo")
    for index, value in enumerate(counts.values):
        ax.text(index, value + 15, str(int(value)), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(figures_dir / "class_balance.pdf")
    plt.close(fig)


def save_top_recent_winrate_figure(snapshot: pd.DataFrame, figures_dir: Path) -> None:
    top10 = snapshot.sort_values("recent_win_rate", ascending=False).head(10).iloc[::-1]
    fig, ax = base_figure(8.0, 5.0)
    ax.barh(top10["team_display_name"], top10["recent_win_rate"], color="#1565c0")
    ax.set_xlabel("Taxa de vitoria recente")
    ax.set_title("Top 10 equipes por win rate recente")
    ax.set_xlim(0, max(1.0, top10["recent_win_rate"].max() + 0.05))
    fig.tight_layout()
    fig.savefig(figures_dir / "top10_recent_winrate.pdf")
    plt.close(fig)


def save_model_comparison_auc_figure(metrics: pd.DataFrame, figures_dir: Path) -> None:
    ordered = metrics.sort_values("test_roc_auc", ascending=True)
    fig, ax = base_figure(8.0, 4.8)
    ax.barh(ordered["model"], ordered["test_roc_auc"], color="#6d4c41")
    ax.set_xlabel("ROC-AUC de teste")
    if len(ordered) == 1:
        ax.set_title("ROC-AUC do modelo principal")
    else:
        ax.set_title("Comparacao dos modelos por ROC-AUC")
    ax.set_xlim(0.45, max(0.75, ordered["test_roc_auc"].max() + 0.05))
    fig.tight_layout()
    fig.savefig(figures_dir / "model_comparison_auc.pdf")
    plt.close(fig)


def save_roc_curves_figure(y_test: pd.Series, test_probabilities: dict[str, pd.Series], figures_dir: Path) -> None:
    fig, ax = base_figure(7.0, 5.5)
    for name, proba in test_probabilities.items():
        RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black", alpha=0.7)
    if len(test_probabilities) == 1:
        ax.set_title("Curva ROC do modelo principal")
    else:
        ax.set_title("Curvas ROC no conjunto de teste")
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_curves.pdf")
    plt.close(fig)


def save_logistic_confusion_matrix_figure(y_test: pd.Series, logistic_proba: pd.Series, figures_dir: Path) -> None:
    predictions = (logistic_proba >= 0.5).astype(int)
    matrix = confusion_matrix(y_test, predictions)
    fig, ax = base_figure(5.5, 4.8)
    display = ConfusionMatrixDisplay(matrix, display_labels=["Derrota", "Vitoria"])
    display.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Matriz de confusao da regressao logistica")
    fig.tight_layout()
    fig.savefig(figures_dir / "logistic_confusion_matrix.pdf")
    plt.close(fig)


def save_logistic_pr_curve_figure(y_test: pd.Series, logistic_proba: pd.Series, figures_dir: Path) -> None:
    fig, ax = base_figure(7.0, 5.5)
    PrecisionRecallDisplay.from_predictions(y_test, logistic_proba, ax=ax, name="logistic_regression")
    ax.set_title("Curva precisao-revocacao da regressao logistica")
    fig.tight_layout()
    fig.savefig(figures_dir / "logistic_pr_curve.pdf")
    plt.close(fig)


def save_logistic_coefficients_figure(coefficients: pd.DataFrame, figures_dir: Path) -> None:
    ordered = coefficients.sort_values("coefficient", ascending=True)
    colors = ["#c62828" if value < 0 else "#2e7d32" for value in ordered["coefficient"]]
    fig, ax = base_figure(8.2, 4.8)
    ax.barh(ordered["feature"], ordered["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Coeficiente padronizado")
    ax.set_title("Coeficientes da regressao logistica")
    fig.tight_layout()
    fig.savefig(figures_dir / "logistic_coefficients.pdf")
    plt.close(fig)


def resolve_feature_columns(data_path: Path, dataset: pd.DataFrame) -> list[str]:
    metadata_path = data_path.parent / "dataset_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        feature_columns = list(metadata.get("feature_columns", []))
        if feature_columns:
            return feature_columns

    inferred = [column for column in dataset.columns if column.startswith("diff_")]
    ordered = [column for column in BASE_FEATURE_COLUMNS if column in inferred]
    extras = [column for column in inferred if column not in ordered]
    return [*ordered, *extras]


def prepare_temporal_holdout(dataset: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if "match_date" not in dataset.columns:
        raise ValueError("Split temporal exige a coluna match_date no dataset supervisionado.")

    working = dataset.copy()
    working["match_date"] = pd.to_datetime(working["match_date"], errors="coerce")
    working = working.dropna(subset=["match_date"]).sort_values(
        ["match_date", "team_name", "opponent_name"],
        kind="stable",
    ).reset_index(drop=True)
    unique_dates = list(pd.Series(working["match_date"].unique()).sort_values())
    if len(unique_dates) < 2:
        raise ValueError("Nao ha datas suficientes para gerar holdout temporal.")

    target_index = max(1, int(len(unique_dates) * (1.0 - test_size)))
    if target_index >= len(unique_dates):
        target_index = len(unique_dates) - 1

    candidate_indices = list(range(target_index, len(unique_dates) - 0))
    for index in candidate_indices:
        if index >= len(unique_dates):
            continue
        cutoff = unique_dates[index]
        train_df = working.loc[working["match_date"] < cutoff].copy()
        test_df = working.loc[working["match_date"] >= cutoff].copy()
        if train_df.empty or test_df.empty:
            continue
        if train_df["win_target"].nunique() < 2 or test_df["win_target"].nunique() < 2:
            continue
        cutoff_label = pd.Timestamp(cutoff).strftime("%Y-%m-%d")
        return train_df, test_df, cutoff_label

    raise ValueError("Nao foi possivel criar um holdout temporal com as duas classes em treino e teste.")


def prepare_season_holdout(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "season_usage" not in dataset.columns:
        raise ValueError("Split por season exige a coluna season_usage no dataset supervisionado.")

    working = dataset.copy()
    working["season_usage"] = working["season_usage"].astype(str).str.strip().str.lower()
    train_df = working.loc[working["season_usage"] == TRAIN_SEASON_USAGE].copy()
    test_df = working.loc[working["season_usage"] == HOLDOUT_SEASON_USAGE].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Nao foi possivel criar o holdout por season. Verifique season_usage no dataset supervisionado.")
    if train_df["win_target"].nunique() < 2 or test_df["win_target"].nunique() < 2:
        raise ValueError("Treino ou holdout por season nao contem as duas classes necessarias.")
    return train_df, test_df


def split_dataset(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    split_strategy: str = "season_holdout",
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, str]]:
    if split_strategy == "stratified":
        X = dataset[feature_columns]
        y = dataset["win_target"].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y,
        )
        metadata = {
            "evaluation_protocol": "train_test_split_stratified_80_20_plus_5fold_cv_no_train",
        }
        return X_train, X_test, y_train, y_test, metadata

    if split_strategy == "temporal":
        train_df, test_df, cutoff_label = prepare_temporal_holdout(dataset, test_size=test_size)
        metadata = {
            "evaluation_protocol": f"temporal_holdout_pre_{cutoff_label}_plus_5fold_stratified_cv_on_train",
            "test_period_start": str(pd.to_datetime(test_df["match_date"]).min().date()),
            "test_period_end": str(pd.to_datetime(test_df["match_date"]).max().date()),
        }
        return (
            train_df[feature_columns],
            test_df[feature_columns],
            train_df["win_target"].astype(int),
            test_df["win_target"].astype(int),
            metadata,
        )

    if split_strategy == "season_holdout":
        train_df, test_df = prepare_season_holdout(dataset)
        metadata = {
            "evaluation_protocol": "season_holdout_train_vs_2026_s1_plus_5fold_stratified_cv_on_train",
            "holdout_seasons": ", ".join(sorted(test_df["season_label"].astype(str).unique().tolist())),
            "test_period_start": str(pd.to_datetime(test_df["match_date"]).min().date()) if "match_date" in test_df.columns else "",
            "test_period_end": str(pd.to_datetime(test_df["match_date"]).max().date()) if "match_date" in test_df.columns else "",
        }
        return (
            train_df[feature_columns],
            test_df[feature_columns],
            train_df["win_target"].astype(int),
            test_df["win_target"].astype(int),
            metadata,
        )

    raise ValueError(f"Estrategia de split invalida: {split_strategy}")


def run_training(
    data_path: Path,
    snapshot_path: Path,
    models_dir: Path,
    reports_dir: Path,
    logreg_config_path: Path = DEFAULT_LOGREG_CONFIG_PATH,
    split_strategy: str = "season_holdout",
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, dict[str, object]]:
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"

    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            "Dataset processado nao encontrado. Execute python build_datasets.py antes do treino."
        )
    if not snapshot_path.exists():
        raise FileNotFoundError(
            "Snapshot processado nao encontrado. Execute python build_datasets.py antes do treino."
        )

    dataset = pd.read_csv(data_path)
    snapshot = pd.read_csv(snapshot_path)
    feature_columns = resolve_feature_columns(data_path, dataset)
    tuning_config = load_logreg_tuning_config(logreg_config_path)
    feature_columns, tuning_config = apply_logreg_tuning(feature_columns, tuning_config)

    missing_features = [column for column in feature_columns if column not in dataset.columns]
    if missing_features:
        raise ValueError(f"Colunas obrigatorias ausentes em match_feature_differences.csv: {missing_features}")

    X = dataset[feature_columns]
    y = dataset["win_target"].astype(int)
    save_feature_descriptive_stats(X, tables_dir)
    save_class_balance_figure(y, figures_dir)
    save_top_recent_winrate_figure(snapshot, figures_dir)

    X_train, X_test, y_train, y_test, split_metadata = split_dataset(
        dataset,
        feature_columns=feature_columns,
        split_strategy=split_strategy,
        test_size=test_size,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows: list[dict[str, float | str]] = []
    fitted_models: dict[str, Pipeline] = {}
    test_probabilities: dict[str, pd.Series] = {}

    logreg_overrides = None
    if tuning_config is not None:
        logreg_overrides = {
            "model_family": tuning_config.get("model_family", "logreg_l2"),
            "penalty": tuning_config["penalty"],
            "C": tuning_config["C"],
            "solver": tuning_config["solver"],
        }
        if tuning_config["class_weight"] is not None:
            logreg_overrides["class_weight"] = tuning_config["class_weight"]
        if "l1_ratio" in tuning_config:
            logreg_overrides["l1_ratio"] = tuning_config["l1_ratio"]

    for name, model in build_models(logreg_overrides).items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "roc_auc": "roc_auc",
                "neg_log_loss": "neg_log_loss",
                "neg_brier": "neg_brier_score",
            },
            n_jobs=1,
        )

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        rows.append(
            {
                "model": name,
                "cv_accuracy_mean": scores["test_accuracy"].mean(),
                "cv_accuracy_std": scores["test_accuracy"].std(),
                "cv_roc_auc_mean": scores["test_roc_auc"].mean(),
                "cv_roc_auc_std": scores["test_roc_auc"].std(),
                "cv_log_loss_mean": -scores["test_neg_log_loss"].mean(),
                "cv_brier_mean": -scores["test_neg_brier"].mean(),
                "test_accuracy": accuracy_score(y_test, pred),
                "test_precision": precision_score(y_test, pred, zero_division=0),
                "test_recall": recall_score(y_test, pred, zero_division=0),
                "test_f1": f1_score(y_test, pred, zero_division=0),
                "test_roc_auc": roc_auc_score(y_test, proba),
                "test_log_loss": log_loss(y_test, proba, labels=[0, 1]),
                "test_brier": brier_score_loss(y_test, proba),
            }
        )
        fitted_models[name] = model
        test_probabilities[name] = proba

    metrics = pd.DataFrame(rows).sort_values("test_roc_auc", ascending=False).reset_index(drop=True)
    save_model_metrics_tables(metrics, tables_dir)

    primary_name = "logistic_regression"
    best_test_name = str(metrics.iloc[0]["model"])
    best_cv_name = str(metrics.sort_values("cv_roc_auc_mean", ascending=False).iloc[0]["model"])

    for name, model in fitted_models.items():
        joblib.dump(model, models_dir / f"{name}.joblib")

    with (models_dir / "feature_columns.json").open("w", encoding="utf-8") as file:
        json.dump(feature_columns, file, ensure_ascii=False, indent=2)

    registry: dict[str, object] = {
        "primary_model": primary_name,
        "best_model_by_test_roc_auc": best_test_name,
        "best_model_by_cv_roc_auc": best_cv_name,
        "available_models": sorted(fitted_models),
        "feature_columns": feature_columns,
        "training_config_source": str(logreg_config_path) if tuning_config is not None else "metadata",
        "feature_selection_source": str(logreg_config_path) if tuning_config is not None else "metadata",
        "official_training_config": tuning_config,
        "logreg_tuning_config": tuning_config,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "split_strategy": split_strategy,
        "data_path": str(data_path),
        "snapshot_path": str(snapshot_path),
        **split_metadata,
    }
    with (models_dir / "model_registry.json").open("w", encoding="utf-8") as file:
        json.dump(registry, file, ensure_ascii=False, indent=2)

    coefficients = save_logistic_coefficients_table(fitted_models[primary_name], feature_columns, tables_dir)
    save_model_comparison_auc_figure(metrics, figures_dir)
    save_roc_curves_figure(y_test, test_probabilities, figures_dir)
    save_logistic_confusion_matrix_figure(y_test, test_probabilities[primary_name], figures_dir)
    save_logistic_pr_curve_figure(y_test, test_probabilities[primary_name], figures_dir)
    save_logistic_coefficients_figure(coefficients, figures_dir)

    return metrics, registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--snapshot-path", default=str(DEFAULT_SNAPSHOT_PATH))
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument(
        "--training-config",
        "--logreg-config",
        dest="training_config",
        default=str(DEFAULT_TRAINING_CONFIG_PATH),
        help="JSON de configuracao do modelo logistico oficial.",
    )
    parser.add_argument("--split-strategy", choices=["stratified", "temporal", "season_holdout"], default="season_holdout")
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path).resolve()
    snapshot_path = Path(args.snapshot_path).resolve()
    models_dir = Path(args.models_dir).resolve()
    reports_dir = Path(args.reports_dir).resolve()
    logreg_config_path = Path(args.training_config).resolve()

    metrics, registry = run_training(
        data_path=data_path,
        snapshot_path=snapshot_path,
        models_dir=models_dir,
        reports_dir=reports_dir,
        logreg_config_path=logreg_config_path,
        split_strategy=args.split_strategy,
        test_size=args.test_size,
    )

    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"

    print(metrics.to_string(index=False))
    print()
    print(f"Modelo principal salvo em: {models_dir / (registry['primary_model'] + '.joblib')}")
    print(f"Melhor ROC-AUC de teste: {registry['best_model_by_test_roc_auc']}")
    print(f"Melhor ROC-AUC medio na validacao cruzada: {registry['best_model_by_cv_roc_auc']}")
    print(f"Tabelas atualizadas em: {tables_dir}")
    print(f"Figuras atualizadas em: {figures_dir}")
    if registry.get("logreg_tuning_config"):
        print(f"Configuracao de treino aplicada a partir de: {logreg_config_path}")
        print(f"Features selecionadas: {len(registry['feature_columns'])}")
    if "test_period_start" in registry and "test_period_end" in registry:
        print(
            "Periodo do holdout final: "
            f"{registry['test_period_start']} ate {registry['test_period_end']}"
        )


if __name__ == "__main__":
    main()
