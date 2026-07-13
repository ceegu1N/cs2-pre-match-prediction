#!/usr/bin/env python3
"""Nucleo compartilhado das baterias experimentais do projeto de CS2.

Este modulo concentra a infraestrutura reutilizavel da busca experimental:
- definicao de candidatos, modelos e familias de features;
- avaliacao temporal e confirmatoria;
- utilitarios de IO, manifests, ranking e metricas;
- suporte a checkpoint/resume e rastreabilidade experimental.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations, islice
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", message=".*'penalty' was deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Inconsistent values: penalty=l1 with l1_ratio=0.0.*", category=UserWarning)

if importlib.util.find_spec("lightgbm") is not None:
    from lightgbm import LGBMClassifier
else:
    LGBMClassifier = None

if importlib.util.find_spec("xgboost") is not None:
    from xgboost import XGBClassifier
else:
    XGBClassifier = None

if importlib.util.find_spec("catboost") is not None:
    from catboost import CatBoostClassifier
else:
    CatBoostClassifier = None

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT / "data" / "processed" / "match_feature_differences.csv"
DEFAULT_REPORTS_ROOT = ROOT / "reports" / "experiments"
DEFAULT_OFFICIAL_CONFIG_PATH = ROOT / "reports" / "modelo_principal" / "config_modelo_principal.json"
DEFAULT_FROZEN_MODEL_PATH = ROOT / "reports" / "experiments" / "frozen_final_model_20260406.json"
DEFAULT_RANDOM_STATE = 42
DEFAULT_WORKERS = max(1, min(12, (os.cpu_count() or 1) - 1))
TRAIN_USAGE = "train"
HOLDOUT_USAGE = "holdout"
META_COLUMNS = {
    "match_date",
    "actual_match_id",
    "season_label",
    "season_usage",
    "team_name",
    "opponent_name",
    "win_target",
}
PRIMARY_SORT_COLUMNS = [
    "temporal_cv_roc_auc_mean",
    "temporal_cv_log_loss_mean",
    "temporal_cv_brier_mean",
    "temporal_cv_accuracy_mean",
]
PRIMARY_SORT_ASCENDING = [False, True, True, False]
THRESHOLDS = np.round(np.arange(0.30, 0.701, 0.005), 3)
DEFAULT_STOCHASTIC_SEEDS = [42, 52, 62, 72, 82, 92, 102, 112]
DEFAULT_SCREENING_SEEDS = [42, 52]
DEFAULT_THRESHOLD_OBJECTIVES = ["f1", "accuracy", "precision", "recall"]
DEFAULT_STAGE1_TOP_GLOBAL = 320
DEFAULT_STAGE1_TOP_PER_FEATURE_FAMILY = 6
DEFAULT_STAGE1_TOP_PER_MODEL_FAMILY = 8
DEFAULT_STAGE3_POOL_TOP_GLOBAL = 72
DEFAULT_STAGE3_POOL_TOP_PER_FEATURE_FAMILY = 4
DEFAULT_STAGE3_POOL_TOP_PER_MODEL_FAMILY = 6
DEFAULT_STAGE3_REPEATED_CV_REPEATS = 4
DEFAULT_ERROR_ANALYSIS_TOP_N = 5
STOCHASTIC_FAMILIES = {
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "lightgbm",
    "xgboost",
    "catboost",
}
CONTEXT_COLUMNS = ["context_is_lan", "context_is_bo1", "context_is_bo3", "context_is_bo5"]
FORCED_CANDIDATE_NAMES = {
    "official11_frozen_recreation__logreg_l2_c3.0_cwbalanced_frozen",
    "official11_baseline__logreg_l2_c3.0_cwbalanced",
    "official11_baseline__logreg_l1_c0.1_cwbalanced",
    "official11_baseline__logreg_l1_c0.1_cwnone",
}


@dataclass(frozen=True)
class TemporalFold:
    fold_name: str
    train_seasons: tuple[str, ...]
    valid_season: str
    train_size: int
    valid_size: int


@dataclass(frozen=True)
class FeatureSetSpec:
    name: str
    family: str
    params: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    params: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    feature_set_name: str
    model_name: str


def kv_pairs(**kwargs: object) -> tuple[tuple[str, object], ...]:
    return tuple(sorted(kwargs.items(), key=lambda item: item[0]))


def params_to_dict(params: tuple[tuple[str, object], ...]) -> dict[str, object]:
    return {key: value for key, value in params}


def parse_csv_strings(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_csv_ints(value: str | None) -> list[int]:
    return [int(item) for item in parse_csv_strings(value)]


def parse_csv_floats(value: str | None) -> list[float]:
    return [float(item) for item in parse_csv_strings(value)]


def available_optional_families() -> dict[str, bool]:
    return {
        "lightgbm": LGBMClassifier is not None,
        "xgboost": XGBClassifier is not None,
        "catboost": CatBoostClassifier is not None,
    }


def is_stochastic_family(model_family: str) -> bool:
    return model_family in STOCHASTIC_FAMILIES


def resolve_seed_list(model_family: str, stochastic_seeds: Sequence[int], base_random_state: int) -> list[int]:
    if is_stochastic_family(model_family):
        deduped = [int(seed) for seed in stochastic_seeds]
        return deduped if deduped else [int(base_random_state)]
    return [int(base_random_state)]


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_output_dir(base_dir: Path, experiment_name: str | None) -> Path:
    name = experiment_name.strip() if experiment_name else f"dissertation_battery_{timestamp_now()}"
    return (base_dir / name).resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dump_json(payload: dict[str, object], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(frame: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    frame.to_csv(output_path, index=False)


def append_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    ensure_dir(output_path.parent)
    frame.to_csv(output_path, mode="a", header=not output_path.exists(), index=False)


def batched(items: Sequence[object], batch_size: int) -> Iterable[list[object]]:
    iterator = iter(items)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            return
        yield batch


def log(message: str, log_path: Path) -> None:
    print(message, flush=True)
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def season_sort_key(label: str) -> tuple[int, int]:
    text = str(label).strip().lower()
    try:
        year, split = text.split("_s", maxsplit=1)
        return int(year), int(split)
    except Exception:
        return (0, 0)


def normalize_series(values: pd.Series) -> pd.Series:
    data = values.astype(float)
    min_value = float(data.min())
    max_value = float(data.max())
    if np.isclose(min_value, max_value):
        return pd.Series(np.ones(len(data)), index=data.index, dtype=float)
    return (data - min_value) / (max_value - min_value)


def dedup_features(features: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for feature in features:
        if feature not in seen:
            ordered.append(feature)
            seen.add(feature)
    return ordered


def infer_feature_blocks(feature_columns: Sequence[str]) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {
        "recent_form": [],
        "temporal_form": [],
        "combat_core": [],
        "utility_vision": [],
        "entry_trade": [],
        "clutch_survival": [],
        "win_pattern": [],
        "roster_context": [],
        "match_context": [],
        "elo_context": [],
        "derived_interactions": [],
        "derived_stability": [],
    }
    for feature in feature_columns:
        if feature.startswith("interaction_"):
            blocks["derived_interactions"].append(feature)
            continue
        elif feature.startswith("delta_") or feature.startswith("ratio_") or feature.startswith("abs_"):
            blocks["derived_stability"].append(feature)
            continue
        if feature in {"diff_recent_win_rate", "diff_maps_played_mean_5", "diff_rounds_played_mean_5"}:
            blocks["recent_form"].append(feature)
        elif feature.startswith("diff_temporal_"):
            blocks["temporal_form"].append(feature)
        elif feature == "diff_elo_pre_match":
            blocks["elo_context"].append(feature)
        elif feature.startswith("context_"):
            blocks["match_context"].append(feature)
        elif feature in {"diff_players_count", "diff_missing_players"}:
            blocks["roster_context"].append(feature)
        elif any(token in feature for token in ["utility", "flash", "flashed"]):
            blocks["utility_vision"].append(feature)
        elif any(token in feature for token in ["opening", "trade_kills"]):
            blocks["entry_trade"].append(feature)
        elif any(token in feature for token in ["saved_", "last_alive", "one_on_one", "time_alive"]):
            blocks["clutch_survival"].append(feature)
        elif any(token in feature for token in ["rounds_with_a_kill", "kills_per_round_win", "damage_per_round_win", "pistol_round"]):
            blocks["win_pattern"].append(feature)
        else:
            blocks["combat_core"].append(feature)
    return {name: values for name, values in blocks.items() if values}


def clip_probabilities(probabilities: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), eps, 1.0 - eps)


def derive_v5_feature_columns(
    dataset: pd.DataFrame,
    base_feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    frame = dataset.copy()
    manifest_rows: list[dict[str, object]] = []
    derived_columns: list[str] = []

    def add_feature(
        name: str,
        values: pd.Series,
        *,
        family: str,
        source_columns: Sequence[str],
        description: str,
    ) -> None:
        cleaned = pd.to_numeric(values, errors="coerce").astype(float)
        cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
        frame[name] = cleaned
        derived_columns.append(name)
        manifest_rows.append(
            {
                "feature_name": name,
                "family": family,
                "source_columns": ",".join(source_columns),
                "description": description,
            }
        )

    def has_all(*columns: str) -> bool:
        return all(column in frame.columns for column in columns)

    def col(name: str) -> pd.Series:
        return frame[name].astype(float)

    if has_all("diff_elo_pre_match", "diff_recent_win_rate"):
        add_feature(
            "interaction_elo_x_recent_form",
            col("diff_elo_pre_match") * col("diff_recent_win_rate"),
            family="interaction",
            source_columns=["diff_elo_pre_match", "diff_recent_win_rate"],
            description="Interacao entre elo pre-match e forma recente.",
        )
    if has_all("diff_elo_pre_match", "diff_rating_mean_5"):
        add_feature(
            "interaction_elo_x_rating",
            col("diff_elo_pre_match") * col("diff_rating_mean_5"),
            family="interaction",
            source_columns=["diff_elo_pre_match", "diff_rating_mean_5"],
            description="Interacao entre elo pre-match e rating recente.",
        )
    if has_all("diff_elo_pre_match", "context_is_lan"):
        add_feature(
            "interaction_elo_x_lan",
            col("diff_elo_pre_match") * col("context_is_lan"),
            family="interaction",
            source_columns=["diff_elo_pre_match", "context_is_lan"],
            description="Vantagem de elo modulada por contexto LAN.",
        )
    if has_all("diff_elo_pre_match", "context_is_bo3"):
        add_feature(
            "interaction_elo_x_bo3",
            col("diff_elo_pre_match") * col("context_is_bo3"),
            family="interaction",
            source_columns=["diff_elo_pre_match", "context_is_bo3"],
            description="Vantagem de elo modulada por series BO3.",
        )
    if has_all("diff_elo_pre_match", "context_is_bo1"):
        add_feature(
            "interaction_elo_x_bo1",
            col("diff_elo_pre_match") * col("context_is_bo1"),
            family="interaction",
            source_columns=["diff_elo_pre_match", "context_is_bo1"],
            description="Vantagem de elo modulada por series BO1.",
        )
    if has_all("diff_elo_pre_match", "context_is_bo5"):
        add_feature(
            "interaction_elo_x_bo5",
            col("diff_elo_pre_match") * col("context_is_bo5"),
            family="interaction",
            source_columns=["diff_elo_pre_match", "context_is_bo5"],
            description="Vantagem de elo modulada por series BO5.",
        )
    if has_all("diff_recent_win_rate", "context_is_lan"):
        add_feature(
            "interaction_form_x_lan",
            col("diff_recent_win_rate") * col("context_is_lan"),
            family="interaction",
            source_columns=["diff_recent_win_rate", "context_is_lan"],
            description="Forma recente modulada por contexto LAN.",
        )
    if has_all("diff_recent_win_rate", "context_is_bo3"):
        add_feature(
            "interaction_form_x_bo3",
            col("diff_recent_win_rate") * col("context_is_bo3"),
            family="interaction",
            source_columns=["diff_recent_win_rate", "context_is_bo3"],
            description="Forma recente modulada por series BO3.",
        )
    if has_all("diff_recent_win_rate", "context_is_bo1"):
        add_feature(
            "interaction_form_x_bo1",
            col("diff_recent_win_rate") * col("context_is_bo1"),
            family="interaction",
            source_columns=["diff_recent_win_rate", "context_is_bo1"],
            description="Forma recente modulada por series BO1.",
        )
    if has_all("diff_opening_success_mean_5", "diff_win_after_opening_kill_pct_mean_5"):
        add_feature(
            "interaction_opening_success_x_conversion",
            col("diff_opening_success_mean_5") * col("diff_win_after_opening_kill_pct_mean_5"),
            family="interaction",
            source_columns=["diff_opening_success_mean_5", "diff_win_after_opening_kill_pct_mean_5"],
            description="Interacao entre sucesso de abertura e conversao apos opening kill.",
        )
        add_feature(
            "delta_opening_success_vs_conversion",
            col("diff_opening_success_mean_5") - col("diff_win_after_opening_kill_pct_mean_5"),
            family="delta",
            source_columns=["diff_opening_success_mean_5", "diff_win_after_opening_kill_pct_mean_5"],
            description="Diferenca entre sucesso de abertura e conversao da vantagem.",
        )
    if has_all("diff_opening_kills_per_round_mean_5", "diff_win_after_opening_kill_pct_mean_5"):
        add_feature(
            "interaction_opening_kills_x_conversion",
            col("diff_opening_kills_per_round_mean_5") * col("diff_win_after_opening_kill_pct_mean_5"),
            family="interaction",
            source_columns=["diff_opening_kills_per_round_mean_5", "diff_win_after_opening_kill_pct_mean_5"],
            description="Interacao entre volume de opening kills e conversao da vantagem.",
        )
    if has_all("diff_utility_damage_per_round_mean_5", "diff_flash_assists_per_round_mean_5"):
        add_feature(
            "interaction_utility_x_flash",
            col("diff_utility_damage_per_round_mean_5") * col("diff_flash_assists_per_round_mean_5"),
            family="interaction",
            source_columns=["diff_utility_damage_per_round_mean_5", "diff_flash_assists_per_round_mean_5"],
            description="Interacao entre dano de utilitario e flash assists.",
        )
        add_feature(
            "ratio_utility_to_flash",
            col("diff_utility_damage_per_round_mean_5") / (1.0 + col("diff_flash_assists_per_round_mean_5").abs()),
            family="ratio",
            source_columns=["diff_utility_damage_per_round_mean_5", "diff_flash_assists_per_round_mean_5"],
            description="Razao estabilizada entre dano de utilitario e flash assists.",
        )
    if has_all("diff_utility_damage_per_round_mean_5", "context_is_lan"):
        add_feature(
            "interaction_utility_x_lan",
            col("diff_utility_damage_per_round_mean_5") * col("context_is_lan"),
            family="interaction",
            source_columns=["diff_utility_damage_per_round_mean_5", "context_is_lan"],
            description="Impacto de utilitario modulado por contexto LAN.",
        )
    if has_all("diff_one_on_one_win_pct_mean_5", "context_is_lan"):
        add_feature(
            "interaction_clutch_x_lan",
            col("diff_one_on_one_win_pct_mean_5") * col("context_is_lan"),
            family="interaction",
            source_columns=["diff_one_on_one_win_pct_mean_5", "context_is_lan"],
            description="Capacidade clutch modulada por contexto LAN.",
        )
    if has_all("diff_time_alive_per_round_mean_5", "diff_rating_mean_5"):
        add_feature(
            "interaction_timealive_x_rating",
            col("diff_time_alive_per_round_mean_5") * col("diff_rating_mean_5"),
            family="interaction",
            source_columns=["diff_time_alive_per_round_mean_5", "diff_rating_mean_5"],
            description="Interacao entre sobrevivencia e rating recente.",
        )
    if has_all("diff_time_alive_per_round_mean_5", "diff_one_on_one_win_pct_mean_5"):
        add_feature(
            "interaction_survival_x_clutch",
            col("diff_time_alive_per_round_mean_5") * col("diff_one_on_one_win_pct_mean_5"),
            family="interaction",
            source_columns=["diff_time_alive_per_round_mean_5", "diff_one_on_one_win_pct_mean_5"],
            description="Interacao entre sobrevivencia e desempenho clutch.",
        )
    if has_all("diff_recent_win_rate", "diff_impact_mean_5"):
        add_feature(
            "interaction_form_x_impact",
            col("diff_recent_win_rate") * col("diff_impact_mean_5"),
            family="interaction",
            source_columns=["diff_recent_win_rate", "diff_impact_mean_5"],
            description="Interacao entre forma recente e impacto.",
        )
    if has_all("diff_recent_win_rate", "diff_rating_mean_5"):
        add_feature(
            "interaction_form_x_rating",
            col("diff_recent_win_rate") * col("diff_rating_mean_5"),
            family="interaction",
            source_columns=["diff_recent_win_rate", "diff_rating_mean_5"],
            description="Interacao entre forma recente e rating.",
        )
    if has_all("diff_kast_mean_5", "diff_rating_mean_5"):
        add_feature(
            "interaction_kast_x_rating",
            col("diff_kast_mean_5") * col("diff_rating_mean_5"),
            family="interaction",
            source_columns=["diff_kast_mean_5", "diff_rating_mean_5"],
            description="Interacao entre KAST e rating recente.",
        )
        add_feature(
            "delta_rating_vs_kast",
            col("diff_rating_mean_5") - col("diff_kast_mean_5"),
            family="delta",
            source_columns=["diff_rating_mean_5", "diff_kast_mean_5"],
            description="Diferenca entre rating e KAST recentes.",
        )
    if has_all("diff_rating_mean_5", "diff_adr_mean_5"):
        add_feature(
            "interaction_rating_x_adr",
            col("diff_rating_mean_5") * col("diff_adr_mean_5"),
            family="interaction",
            source_columns=["diff_rating_mean_5", "diff_adr_mean_5"],
            description="Interacao entre rating e ADR.",
        )
    if has_all("diff_impact_mean_5", "diff_kd_mean_5"):
        add_feature(
            "interaction_impact_x_kd",
            col("diff_impact_mean_5") * col("diff_kd_mean_5"),
            family="interaction",
            source_columns=["diff_impact_mean_5", "diff_kd_mean_5"],
            description="Interacao entre impacto e K/D.",
        )
    if has_all("diff_rating_mean_5", "diff_impact_mean_5"):
        add_feature(
            "delta_rating_vs_impact",
            col("diff_rating_mean_5") - col("diff_impact_mean_5"),
            family="delta",
            source_columns=["diff_rating_mean_5", "diff_impact_mean_5"],
            description="Diferenca entre rating e impacto recente.",
        )
    if has_all("diff_adr_mean_5", "diff_impact_mean_5"):
        add_feature(
            "delta_adr_vs_impact",
            col("diff_adr_mean_5") - col("diff_impact_mean_5"),
            family="delta",
            source_columns=["diff_adr_mean_5", "diff_impact_mean_5"],
            description="Diferenca entre ADR e impacto recentes.",
        )
    if has_all("diff_kills_per_round_mean_5", "diff_deaths_per_round_mean_5"):
        add_feature(
            "delta_kills_vs_deaths",
            col("diff_kills_per_round_mean_5") - col("diff_deaths_per_round_mean_5"),
            family="delta",
            source_columns=["diff_kills_per_round_mean_5", "diff_deaths_per_round_mean_5"],
            description="Margem entre kills por round e deaths por round.",
        )
        add_feature(
            "ratio_kills_to_deaths",
            col("diff_kills_per_round_mean_5") / (1.0 + col("diff_deaths_per_round_mean_5").abs()),
            family="ratio",
            source_columns=["diff_kills_per_round_mean_5", "diff_deaths_per_round_mean_5"],
            description="Razao estabilizada entre kills por round e deaths por round.",
        )
    if has_all("diff_opening_kills_per_round_mean_5", "diff_opening_deaths_per_round_mean_5"):
        add_feature(
            "delta_opening_kills_vs_deaths",
            col("diff_opening_kills_per_round_mean_5") - col("diff_opening_deaths_per_round_mean_5"),
            family="delta",
            source_columns=["diff_opening_kills_per_round_mean_5", "diff_opening_deaths_per_round_mean_5"],
            description="Margem entre opening kills e opening deaths.",
        )
        add_feature(
            "ratio_opening_kills_to_deaths",
            col("diff_opening_kills_per_round_mean_5") / (1.0 + col("diff_opening_deaths_per_round_mean_5").abs()),
            family="ratio",
            source_columns=["diff_opening_kills_per_round_mean_5", "diff_opening_deaths_per_round_mean_5"],
            description="Razao estabilizada entre opening kills e opening deaths.",
        )
    if has_all("diff_trade_kills_per_round_mean_5", "diff_opening_success_mean_5"):
        add_feature(
            "interaction_trade_x_opening_success",
            col("diff_trade_kills_per_round_mean_5") * col("diff_opening_success_mean_5"),
            family="interaction",
            source_columns=["diff_trade_kills_per_round_mean_5", "diff_opening_success_mean_5"],
            description="Interacao entre trade kills e sucesso de abertura.",
        )
    if has_all("diff_trade_kills_per_round_mean_5", "context_is_lan"):
        add_feature(
            "interaction_trade_x_lan",
            col("diff_trade_kills_per_round_mean_5") * col("context_is_lan"),
            family="interaction",
            source_columns=["diff_trade_kills_per_round_mean_5", "context_is_lan"],
            description="Trade kills modulados por contexto LAN.",
        )
    if has_all("diff_players_count", "context_is_lan"):
        add_feature(
            "interaction_players_count_x_lan",
            col("diff_players_count") * col("context_is_lan"),
            family="interaction",
            source_columns=["diff_players_count", "context_is_lan"],
            description="Diferenca de players count modulada por LAN.",
        )
    if has_all("diff_missing_players", "diff_elo_pre_match"):
        add_feature(
            "interaction_missing_players_x_elo",
            col("diff_missing_players") * col("diff_elo_pre_match"),
            family="interaction",
            source_columns=["diff_missing_players", "diff_elo_pre_match"],
            description="Ausencias no roster moduladas pela diferenca de elo.",
        )
    if has_all("diff_missing_players", "context_is_lan"):
        add_feature(
            "interaction_missing_players_x_lan",
            col("diff_missing_players") * col("context_is_lan"),
            family="interaction",
            source_columns=["diff_missing_players", "context_is_lan"],
            description="Ausencias no roster moduladas por LAN.",
        )
    if has_all("diff_pistol_round_rating_mean_5", "context_is_bo1"):
        add_feature(
            "interaction_pistol_x_bo1",
            col("diff_pistol_round_rating_mean_5") * col("context_is_bo1"),
            family="interaction",
            source_columns=["diff_pistol_round_rating_mean_5", "context_is_bo1"],
            description="Pistol round rating modulada por series BO1.",
        )
    if has_all("diff_rounds_played_mean_5", "diff_maps_played_mean_5"):
        add_feature(
            "delta_rounds_vs_maps",
            col("diff_rounds_played_mean_5") - col("diff_maps_played_mean_5"),
            family="delta",
            source_columns=["diff_rounds_played_mean_5", "diff_maps_played_mean_5"],
            description="Diferenca entre rounds jogados e mapas jogados.",
        )
    if has_all("diff_elo_pre_match"):
        add_feature(
            "abs_elo_gap",
            col("diff_elo_pre_match").abs(),
            family="absolute",
            source_columns=["diff_elo_pre_match"],
            description="Magnitude absoluta da diferenca de elo.",
        )
    if has_all("diff_recent_win_rate"):
        add_feature(
            "abs_recent_form_gap",
            col("diff_recent_win_rate").abs(),
            family="absolute",
            source_columns=["diff_recent_win_rate"],
            description="Magnitude absoluta da diferenca de forma recente.",
        )
    if has_all("diff_opening_kills_per_round_mean_5", "diff_opening_deaths_per_round_mean_5"):
        add_feature(
            "abs_opening_gap",
            (col("diff_opening_kills_per_round_mean_5") - col("diff_opening_deaths_per_round_mean_5")).abs(),
            family="absolute",
            source_columns=["diff_opening_kills_per_round_mean_5", "diff_opening_deaths_per_round_mean_5"],
            description="Magnitude absoluta do saldo de opening duels.",
        )
    if has_all("diff_kd_mean_5"):
        add_feature(
            "abs_kd_gap",
            col("diff_kd_mean_5").abs(),
            family="absolute",
            source_columns=["diff_kd_mean_5"],
            description="Magnitude absoluta da diferenca de K/D.",
        )

    ordered_columns = dedup_features(list(base_feature_columns) + derived_columns)
    return frame, ordered_columns, pd.DataFrame(manifest_rows)


def load_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset supervisionado nao encontrado: {data_path}")
    dataset = pd.read_csv(data_path)
    dataset["season_label"] = dataset["season_label"].astype(str).str.strip().str.lower()
    dataset["season_usage"] = dataset["season_usage"].astype(str).str.strip().str.lower()
    dataset["match_date"] = pd.to_datetime(dataset["match_date"], errors="coerce")
    return dataset.sort_values(["match_date", "season_label", "team_name", "opponent_name"], kind="stable").reset_index(drop=True)


def read_official_feature_list(config_path: Path) -> list[str]:
    if not config_path.exists():
        return []
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    features = [str(item).strip() for item in payload.get("features", []) if str(item).strip()]
    return dedup_features(features)


def resolve_feature_columns(dataset: pd.DataFrame, data_path: Path) -> list[str]:
    metadata_candidates = [
        data_path.with_name(f"{data_path.stem}_metadata.json"),
        data_path.parent / "dataset_metadata.json",
    ]
    for metadata_path in metadata_candidates:
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            feature_columns = [str(item).strip() for item in metadata.get("feature_columns", []) if str(item).strip()]
            if feature_columns:
                return [column for column in feature_columns if column in dataset.columns]
    inferred = [column for column in dataset.columns if column not in META_COLUMNS]
    return dedup_features(inferred)


def split_train_holdout(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = dataset.loc[dataset["season_usage"] == TRAIN_USAGE].copy().reset_index(drop=True)
    holdout_df = dataset.loc[dataset["season_usage"] == HOLDOUT_USAGE].copy().reset_index(drop=True)
    if train_df.empty or holdout_df.empty:
        raise ValueError("Nao foi possivel separar treino e holdout a partir de season_usage.")
    if train_df["win_target"].nunique() < 2 or holdout_df["win_target"].nunique() < 2:
        raise ValueError("Treino ou holdout nao contem as duas classes necessarias.")
    return train_df, holdout_df


def build_temporal_folds(train_df: pd.DataFrame) -> list[TemporalFold]:
    seasons = sorted(train_df["season_label"].astype(str).unique().tolist(), key=season_sort_key)
    folds: list[TemporalFold] = []
    for index, valid_season in enumerate(seasons[1:], start=1):
        train_seasons = tuple(seasons[:index])
        fold_train = train_df.loc[train_df["season_label"].isin(train_seasons)]
        fold_valid = train_df.loc[train_df["season_label"] == valid_season]
        if fold_train.empty or fold_valid.empty:
            continue
        if fold_train["win_target"].nunique() < 2 or fold_valid["win_target"].nunique() < 2:
            continue
        folds.append(
            TemporalFold(
                fold_name=f"temporal_fold_{index}",
                train_seasons=train_seasons,
                valid_season=valid_season,
                train_size=int(len(fold_train)),
                valid_size=int(len(fold_valid)),
            )
        )
    if not folds:
        raise ValueError("Nao foi possivel montar folds temporais internos a partir das seasons de treino.")
    return folds


def build_pipeline_audit(
    *,
    data_path: Path,
    official_config_path: Path,
    feature_columns: list[str],
    official_features: list[str],
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
) -> dict[str, object]:
    return {
        "generated_at": datetime.now().isoformat(),
        "dataset_path": str(data_path),
        "dataset_feature_count": int(len(feature_columns)),
        "dataset_features": feature_columns,
        "official_feature_config_path": str(official_config_path),
        "official_feature_count": int(len(official_features)),
        "official_features": official_features,
        "official_pipeline_constraints": {
            "run_pipeline_uses_train_model_defaults": str(ROOT / "run_pipeline.py"),
            "train_model_default_logreg_config": str(ROOT / "train_model.py"),
            "official_training_config": str(official_config_path),
            "official_outputs_models_dir": str(ROOT / "models"),
            "official_outputs_reports_dir": str(ROOT / "reports"),
            "official_processed_dir": str(ROOT / "data" / "processed"),
            "prediction_entrypoint": str(ROOT / "predict_match.py"),
        },
        "isolation_strategy": {
            "runner": str(ROOT / "run_dissertation_battery.py"),
            "output_root": str(DEFAULT_REPORTS_ROOT),
            "no_overwrite_targets": [
                str(ROOT / "models"),
                str(ROOT / "reports"),
                str(ROOT / "data" / "processed"),
            ],
        },
        "split_summary": {
            "train_rows": int(len(train_df)),
            "holdout_rows": int(len(holdout_df)),
            "train_seasons": sorted(train_df["season_label"].astype(str).unique().tolist(), key=season_sort_key),
            "holdout_seasons": sorted(holdout_df["season_label"].astype(str).unique().tolist(), key=season_sort_key),
            "holdout_date_min": str(holdout_df["match_date"].min().date()) if holdout_df["match_date"].notna().any() else "",
            "holdout_date_max": str(holdout_df["match_date"].max().date()) if holdout_df["match_date"].notna().any() else "",
        },
        "methodological_gaps_in_previous_runner": [
            "avaliava holdout para todos os candidatos na etapa exploratoria",
            "nao incluia random_forest, extra_trees, histgb, dummy ou elastic_net",
            "nao tinha shortlist congelada antes da fase confirmatoria",
            "nao tratava calibracao, bootstrap e threshold tuning apenas para finalistas",
        ],
    }


def fold_to_frames(train_df: pd.DataFrame, fold: TemporalFold) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_train = train_df.loc[train_df["season_label"].isin(fold.train_seasons)].copy()
    fold_valid = train_df.loc[train_df["season_label"] == fold.valid_season].copy()
    return fold_train, fold_valid


def load_completed_candidates(output_path: Path, column: str = "candidate_name") -> set[str]:
    if not output_path.exists():
        return set()
    completed: set[str] = set()
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row.get(column)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                completed.add(text)
    return completed


def build_pipeline(model_family: str, params: dict[str, object], random_state: int) -> Pipeline:
    if model_family == "dummy_prior":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", DummyClassifier(strategy="prior")),
            ]
        )

    if model_family == "logreg_l2":
        kwargs: dict[str, object] = {
            "C": float(params["C"]),
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 6000,
            "random_state": random_state,
        }
        if params.get("class_weight") is not None:
            kwargs["class_weight"] = params["class_weight"]
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(**kwargs)),
            ]
        )

    if model_family == "logreg_l1":
        kwargs = {
            "C": float(params["C"]),
            "penalty": "l1",
            "solver": "liblinear",
            "max_iter": 6000,
            "random_state": random_state,
        }
        if params.get("class_weight") is not None:
            kwargs["class_weight"] = params["class_weight"]
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(**kwargs)),
            ]
        )

    if model_family == "logreg_elasticnet":
        kwargs = {
            "C": float(params["C"]),
            "penalty": "elasticnet",
            "solver": "saga",
            "l1_ratio": float(params["l1_ratio"]),
            "max_iter": 8000,
            "random_state": random_state,
        }
        if params.get("class_weight") is not None:
            kwargs["class_weight"] = params["class_weight"]
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(**kwargs)),
            ]
        )

    if model_family == "gaussian_nb":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", GaussianNB(var_smoothing=float(params["var_smoothing"]))),
            ]
        )

    if model_family == "knn":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    KNeighborsClassifier(
                        n_neighbors=int(params["n_neighbors"]),
                        weights=str(params["weights"]),
                        p=2,
                    ),
                ),
            ]
        )

    if model_family == "svm_rbf":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        C=float(params["C"]),
                        kernel="rbf",
                        gamma=str(params["gamma"]),
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if model_family == "svm_linear":
        kwargs: dict[str, object] = {
            "C": float(params["C"]),
            "kernel": "linear",
            "probability": True,
            "random_state": random_state,
        }
        if params.get("class_weight") is not None:
            kwargs["class_weight"] = params["class_weight"]
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", SVC(**kwargs)),
            ]
        )

    if model_family == "random_forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=int(params["n_estimators"]),
                        max_depth=params["max_depth"],
                        min_samples_leaf=int(params["min_samples_leaf"]),
                        max_features=params["max_features"],
                        class_weight=params["class_weight"],
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    if model_family == "extra_trees":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=int(params["n_estimators"]),
                        max_depth=params["max_depth"],
                        min_samples_leaf=int(params["min_samples_leaf"]),
                        max_features=params["max_features"],
                        class_weight=params["class_weight"],
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    if model_family == "hist_gradient_boosting":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        learning_rate=float(params["learning_rate"]),
                        max_depth=params["max_depth"],
                        max_leaf_nodes=int(params["max_leaf_nodes"]),
                        min_samples_leaf=int(params["min_samples_leaf"]),
                        max_iter=int(params["max_iter"]),
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if model_family == "lightgbm":
        if LGBMClassifier is None:
            raise ValueError("LightGBM nao esta disponivel neste ambiente.")
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    LGBMClassifier(
                        n_estimators=int(params["n_estimators"]),
                        learning_rate=float(params["learning_rate"]),
                        max_depth=int(params["max_depth"]),
                        num_leaves=int(params["num_leaves"]),
                        subsample=float(params["subsample"]),
                        colsample_bytree=float(params["colsample_bytree"]),
                        objective="binary",
                        random_state=random_state,
                        n_jobs=1,
                        verbose=-1,
                    ),
                ),
            ]
        )

    if model_family == "xgboost":
        if XGBClassifier is None:
            raise ValueError("XGBoost nao esta disponivel neste ambiente.")
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=int(params["n_estimators"]),
                        learning_rate=float(params["learning_rate"]),
                        max_depth=int(params["max_depth"]),
                        subsample=float(params["subsample"]),
                        colsample_bytree=float(params["colsample_bytree"]),
                        reg_lambda=float(params["reg_lambda"]),
                        random_state=random_state,
                        n_jobs=1,
                        eval_metric="logloss",
                    ),
                ),
            ]
        )

    if model_family == "catboost":
        if CatBoostClassifier is None:
            raise ValueError("CatBoost nao esta disponivel neste ambiente.")
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    CatBoostClassifier(
                        iterations=int(params["iterations"]),
                        learning_rate=float(params["learning_rate"]),
                        depth=int(params["depth"]),
                        loss_function="Logloss",
                        verbose=False,
                        allow_writing_files=False,
                        random_seed=random_state,
                    ),
                ),
            ]
        )

    raise ValueError(f"Familia de modelo invalida: {model_family}")


def predict_positive_probability(model: Pipeline, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    return model.predict_proba(frame[feature_columns])[:, 1]


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    predictions = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def probability_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    clipped = clip_probabilities(y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, object]] = []
    ece = 0.0
    total = max(len(clipped), 1)
    for index in range(n_bins):
        left = float(edges[index])
        right = float(edges[index + 1])
        mask = (clipped >= left) & (clipped <= right) if index == n_bins - 1 else (clipped >= left) & (clipped < right)
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin": index + 1,
                    "lower": left,
                    "upper": right,
                    "count": 0,
                    "mean_confidence": np.nan,
                    "mean_accuracy": np.nan,
                    "abs_gap": np.nan,
                }
            )
            continue
        mean_confidence = float(clipped[mask].mean())
        mean_accuracy = float(y_true[mask].mean())
        gap = abs(mean_confidence - mean_accuracy)
        ece += (count / total) * gap
        rows.append(
            {
                "bin": index + 1,
                "lower": left,
                "upper": right,
                "count": count,
                "mean_confidence": mean_confidence,
                "mean_accuracy": mean_accuracy,
                "abs_gap": gap,
            }
        )
    return float(ece), pd.DataFrame(rows)


def evaluate_probabilities(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    metrics = {}
    metrics.update(probability_metrics(y_true, y_prob))
    metrics.update(binary_metrics(y_true, y_prob, threshold=threshold))
    ece, _ = compute_ece(y_true, y_prob, n_bins=10)
    metrics["ece"] = float(ece)
    return metrics


def build_inner_split_specs(
    frame: pd.DataFrame,
    *,
    random_state: int,
    fallback_splits: int,
    fallback_repeats: int,
) -> list[dict[str, object]]:
    seasons = sorted(frame["season_label"].astype(str).unique().tolist(), key=season_sort_key)
    split_specs: list[dict[str, object]] = []

    if len(seasons) >= 2:
        for index, valid_season in enumerate(seasons[1:], start=1):
            train_seasons = seasons[:index]
            train_idx = np.flatnonzero(frame["season_label"].isin(train_seasons).to_numpy())
            valid_idx = np.flatnonzero((frame["season_label"] == valid_season).to_numpy())
            if len(train_idx) == 0 or len(valid_idx) == 0:
                continue
            y_train = frame.iloc[train_idx]["win_target"].astype(int)
            y_valid = frame.iloc[valid_idx]["win_target"].astype(int)
            if y_train.nunique() < 2 or y_valid.nunique() < 2:
                continue
            split_specs.append(
                {
                    "split_name": f"inner_temporal_{index}",
                    "split_type": "temporal",
                    "train_idx": train_idx,
                    "valid_idx": valid_idx,
                }
            )
        if split_specs:
            return split_specs

    y = frame["win_target"].astype(int).to_numpy()
    class_counts = pd.Series(y).value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
    usable_splits = min(max(2, fallback_splits), min_class_count) if min_class_count >= 2 else 0
    if usable_splits >= 2:
        cv = RepeatedStratifiedKFold(
            n_splits=usable_splits,
            n_repeats=max(1, fallback_repeats),
            random_state=random_state,
        )
        for split_index, (train_idx, valid_idx) in enumerate(cv.split(np.zeros(len(frame)), y), start=1):
            split_specs.append(
                {
                    "split_name": f"inner_repeated_{split_index}",
                    "split_type": "repeated_stratified",
                    "train_idx": train_idx,
                    "valid_idx": valid_idx,
                }
            )
        return split_specs

    if len(frame) >= 20:
        ordered = frame.sort_values(["match_date", "season_label", "team_name", "opponent_name"], kind="stable").reset_index(drop=True)
        split_at = max(1, int(len(ordered) * 0.75))
        head = ordered.iloc[:split_at]
        tail = ordered.iloc[split_at:]
        if not head.empty and not tail.empty and head["win_target"].nunique() >= 2 and tail["win_target"].nunique() >= 2:
            split_specs.append(
                {
                    "split_name": "inner_tail_split",
                    "split_type": "time_tail",
                    "train_idx": head.index.to_numpy(dtype=int),
                    "valid_idx": tail.index.to_numpy(dtype=int),
                }
            )
    if not split_specs:
        raise ValueError("Nao foi possivel montar splits internos para ranking/seleção de features.")
    return split_specs


def compute_univariate_feature_ranking(
    frame: pd.DataFrame,
    feature_columns: list[str],
    split_specs: list[dict[str, object]],
    *,
    random_state: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in feature_columns:
        fold_metrics: list[dict[str, float]] = []
        for split in split_specs:
            split_train = frame.iloc[np.asarray(split["train_idx"], dtype=int)]
            split_valid = frame.iloc[np.asarray(split["valid_idx"], dtype=int)]
            model = build_pipeline("logreg_l2", {"C": 1.0, "class_weight": "balanced"}, random_state)
            model.fit(split_train[[feature]], split_train["win_target"].astype(int))
            probabilities = predict_positive_probability(model, split_valid, [feature])
            fold_metrics.append(
                evaluate_probabilities(
                    split_valid["win_target"].astype(int).to_numpy(),
                    probabilities,
                    threshold=0.5,
                )
            )
        metric_frame = pd.DataFrame(fold_metrics)
        rows.append(
            {
                "feature": feature,
                "inner_split_count": int(len(metric_frame)),
                "roc_auc_mean": float(metric_frame["roc_auc"].mean()),
                "roc_auc_std": float(metric_frame["roc_auc"].std(ddof=0)),
                "log_loss_mean": float(metric_frame["log_loss"].mean()),
                "brier_mean": float(metric_frame["brier"].mean()),
                "accuracy_mean": float(metric_frame["accuracy"].mean()),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["roc_auc_mean", "log_loss_mean", "brier_mean", "accuracy_mean"],
        ascending=[False, True, True, False],
        kind="stable",
    )
    return ranking.reset_index(drop=True)


def compute_l1_stability_ranking(
    frame: pd.DataFrame,
    feature_columns: list[str],
    split_specs: list[dict[str, object]],
    *,
    random_state: int,
) -> pd.DataFrame:
    frequency = {feature: 0 for feature in feature_columns}
    coefficient_sum = {feature: 0.0 for feature in feature_columns}
    total_rounds = 0
    l1_grid = [
        {"C": 0.03, "class_weight": None},
        {"C": 0.03, "class_weight": "balanced"},
        {"C": 0.10, "class_weight": None},
        {"C": 0.10, "class_weight": "balanced"},
        {"C": 0.30, "class_weight": None},
        {"C": 0.30, "class_weight": "balanced"},
        {"C": 1.00, "class_weight": None},
        {"C": 1.00, "class_weight": "balanced"},
    ]

    for split in split_specs:
        split_train = frame.iloc[np.asarray(split["train_idx"], dtype=int)]
        y_train = split_train["win_target"].astype(int)
        for params in l1_grid:
            model = build_pipeline("logreg_l1", params, random_state)
            model.fit(split_train[feature_columns], y_train)
            coefficients = np.abs(model.named_steps["clf"].coef_[0])
            total_rounds += 1
            for feature, coef in zip(feature_columns, coefficients):
                coefficient_sum[feature] += float(coef)
                if coef > 1e-8:
                    frequency[feature] += 1

    rows = []
    total_rounds = max(total_rounds, 1)
    for feature in feature_columns:
        rows.append(
            {
                "feature": feature,
                "l1_frequency": int(frequency[feature]),
                "l1_frequency_rate": float(frequency[feature] / total_rounds),
                "l1_mean_abs_coef": float(coefficient_sum[feature] / total_rounds),
                "l1_total_rounds": int(total_rounds),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["l1_frequency_rate", "l1_mean_abs_coef", "feature"],
        ascending=[False, False, True],
        kind="stable",
    )
    return ranking.reset_index(drop=True)


def compute_tree_importance_ranking(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    random_state: int,
) -> pd.DataFrame:
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(frame[feature_columns])
    y = frame["win_target"].astype(int).to_numpy()

    extra_trees = ExtraTreesClassifier(
        n_estimators=700,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )
    extra_trees.fit(X, y)

    random_forest = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=1,
    )
    random_forest.fit(X, y)

    ranking = pd.DataFrame(
        {
            "feature": feature_columns,
            "extra_trees_importance": extra_trees.feature_importances_,
            "random_forest_importance": random_forest.feature_importances_,
        }
    )
    ranking["tree_importance_mean"] = ranking[["extra_trees_importance", "random_forest_importance"]].mean(axis=1)
    ranking = ranking.sort_values(
        ["tree_importance_mean", "extra_trees_importance", "random_forest_importance"],
        ascending=[False, False, False],
        kind="stable",
    )
    return ranking.reset_index(drop=True)


def combine_feature_rankings(
    univariate_df: pd.DataFrame,
    l1_df: pd.DataFrame,
    tree_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = univariate_df.merge(l1_df, on="feature", how="left").merge(tree_df, on="feature", how="left")
    merged = merged.fillna(0.0)
    top_k_vote = max(5, min(15, len(merged)))
    merged["univariate_rank"] = merged["roc_auc_mean"].rank(method="dense", ascending=False)
    merged["l1_rank"] = merged["l1_frequency_rate"].rank(method="dense", ascending=False)
    merged["tree_rank"] = merged["tree_importance_mean"].rank(method="dense", ascending=False)
    merged["single_auc_signal"] = (merged["roc_auc_mean"] - 0.5).clip(lower=0.0)
    merged["single_auc_norm"] = normalize_series(merged["single_auc_signal"])
    merged["single_logloss_inv_norm"] = 1.0 - normalize_series(merged["log_loss_mean"])
    merged["l1_frequency_norm"] = normalize_series(merged["l1_frequency_rate"])
    merged["l1_coef_norm"] = normalize_series(merged["l1_mean_abs_coef"])
    merged["extra_trees_norm"] = normalize_series(merged["extra_trees_importance"])
    merged["random_forest_norm"] = normalize_series(merged["random_forest_importance"])
    merged["stable_minimum_flag"] = (merged["l1_frequency_rate"] >= 0.50).astype(float)
    merged["top_vote_count"] = (
        (merged["univariate_rank"] <= top_k_vote).astype(int)
        + (merged["l1_rank"] <= top_k_vote).astype(int)
        + (merged["tree_rank"] <= top_k_vote).astype(int)
    ).astype(float)
    merged["top_vote_norm"] = normalize_series(merged["top_vote_count"])
    merged["combined_rank_score"] = (
        0.22 * merged["single_auc_norm"]
        + 0.08 * merged["single_logloss_inv_norm"]
        + 0.18 * merged["l1_frequency_norm"]
        + 0.07 * merged["l1_coef_norm"]
        + 0.15 * merged["extra_trees_norm"]
        + 0.15 * merged["random_forest_norm"]
        + 0.10 * merged["stable_minimum_flag"]
        + 0.05 * merged["top_vote_norm"]
    )
    merged["tree_tilt_rank_score"] = (
        0.18 * merged["single_auc_norm"]
        + 0.06 * merged["single_logloss_inv_norm"]
        + 0.12 * merged["l1_frequency_norm"]
        + 0.04 * merged["l1_coef_norm"]
        + 0.26 * merged["extra_trees_norm"]
        + 0.26 * merged["random_forest_norm"]
        + 0.05 * merged["stable_minimum_flag"]
        + 0.03 * merged["top_vote_norm"]
    )
    merged["l1_tilt_rank_score"] = (
        0.18 * merged["single_auc_norm"]
        + 0.08 * merged["single_logloss_inv_norm"]
        + 0.28 * merged["l1_frequency_norm"]
        + 0.12 * merged["l1_coef_norm"]
        + 0.12 * merged["extra_trees_norm"]
        + 0.12 * merged["random_forest_norm"]
        + 0.06 * merged["stable_minimum_flag"]
        + 0.04 * merged["top_vote_norm"]
    )
    merged["univariate_tilt_rank_score"] = (
        0.34 * merged["single_auc_norm"]
        + 0.14 * merged["single_logloss_inv_norm"]
        + 0.15 * merged["l1_frequency_norm"]
        + 0.05 * merged["l1_coef_norm"]
        + 0.12 * merged["extra_trees_norm"]
        + 0.12 * merged["random_forest_norm"]
        + 0.04 * merged["stable_minimum_flag"]
        + 0.04 * merged["top_vote_norm"]
    )
    merged["conservative_rank_score"] = (
        0.18 * merged["single_auc_norm"]
        + 0.12 * merged["single_logloss_inv_norm"]
        + 0.18 * merged["l1_frequency_norm"]
        + 0.08 * merged["l1_coef_norm"]
        + 0.14 * merged["extra_trees_norm"]
        + 0.14 * merged["random_forest_norm"]
        + 0.10 * merged["stable_minimum_flag"]
        + 0.06 * merged["top_vote_norm"]
    )
    merged = merged.sort_values(
        ["combined_rank_score", "roc_auc_mean", "log_loss_mean"],
        ascending=[False, False, True],
        kind="stable",
    )
    return merged.reset_index(drop=True)


def build_feature_set_specs(official_features: list[str], feature_columns: list[str], preset: str) -> list[FeatureSetSpec]:
    feature_count = len(feature_columns)
    limited_official = tuple(feature for feature in official_features[:11] if feature)
    feature_blocks = infer_feature_blocks(feature_columns)
    derived_interactions = [feature for feature in feature_columns if feature.startswith("interaction_")]
    derived_stability = [
        feature for feature in feature_columns if feature.startswith("delta_") or feature.startswith("ratio_") or feature.startswith("abs_")
    ]
    derived_all = dedup_features(list(derived_interactions) + list(derived_stability))
    ranking_schemes = [
        "combined_rank_score",
        "tree_tilt_rank_score",
        "l1_tilt_rank_score",
        "univariate_tilt_rank_score",
        "conservative_rank_score",
    ]
    if preset == "smoke":
        combined_ks = [5, 11, min(feature_count, 17)]
        family_ks = [5, 11]
        derived_ks = [5, 8]
    else:
        combined_ks = [4, 5, 6, 8, 10, 11, 12, 14, 17, 20, 23, 26, 29, 32, 35, 40, 45, 50, feature_count]
        family_ks = [4, 5, 8, 11, 14, 17, 23, 29, 35, 45]
        derived_ks = [4, 5, 8, 11, 14, 17, 23]

    specs: list[FeatureSetSpec] = [FeatureSetSpec(name="all_features", family="all_features", params=kv_pairs())]
    if limited_official:
        specs.append(FeatureSetSpec(name="official11_baseline", family="explicit", params=(("features", limited_official),)))
        specs.append(FeatureSetSpec(name="official11_frozen_recreation", family="explicit", params=(("features", limited_official),)))
    if derived_all:
        specs.append(FeatureSetSpec(name="derived_only", family="explicit", params=(("features", tuple(derived_all)),)))
    if derived_interactions:
        specs.append(FeatureSetSpec(name="derived_interactions_only", family="explicit", params=(("features", tuple(derived_interactions)),)))
    if derived_stability:
        specs.append(FeatureSetSpec(name="derived_stability_only", family="explicit", params=(("features", tuple(derived_stability)),)))
    if limited_official and derived_interactions:
        specs.append(
            FeatureSetSpec(
                name="official11_plus_derived_interactions",
                family="explicit",
                params=(("features", tuple(dedup_features(list(limited_official) + list(derived_interactions)))),),
            )
        )
    if limited_official and derived_stability:
        specs.append(
            FeatureSetSpec(
                name="official11_plus_derived_stability",
                family="explicit",
                params=(("features", tuple(dedup_features(list(limited_official) + list(derived_stability)))),),
            )
        )
    if limited_official and derived_all:
        specs.append(
            FeatureSetSpec(
                name="official11_plus_all_derived",
                family="explicit",
                params=(("features", tuple(dedup_features(list(limited_official) + list(derived_all)))),),
            )
        )
    for k in derived_ks:
        specs.append(FeatureSetSpec(name=f"derived_top_{k}", family="derived_top_k", params=kv_pairs(k=k)))
        specs.append(
            FeatureSetSpec(
                name=f"official11_plus_derived_top_{k}",
                family="official_plus_derived_top_k",
                params=kv_pairs(k=k),
            )
        )
        if preset != "smoke":
            specs.append(
                FeatureSetSpec(
                    name=f"derived_combined_top_{k}",
                    family="derived_rank_scheme_top_k",
                    params=kv_pairs(k=k, scheme="combined_rank_score"),
                )
            )
            specs.append(
                FeatureSetSpec(
                    name=f"derived_tree_tilt_top_{k}",
                    family="derived_rank_scheme_top_k",
                    params=kv_pairs(k=k, scheme="tree_tilt_rank_score"),
                )
            )
            specs.append(
                FeatureSetSpec(
                    name=f"derived_l1_tilt_top_{k}",
                    family="derived_rank_scheme_top_k",
                    params=kv_pairs(k=k, scheme="l1_tilt_rank_score"),
                )
            )

    for k in dedup_features([str(item) for item in combined_ks]):
        count = min(feature_count, int(k))
        specs.append(FeatureSetSpec(name=f"combined_top_{count}", family="combined_top_k", params=kv_pairs(k=count)))
    for k in family_ks:
        count = min(feature_count, int(k))
        specs.append(FeatureSetSpec(name=f"univariate_top_{count}", family="univariate_top_k", params=kv_pairs(k=count)))
        specs.append(FeatureSetSpec(name=f"tree_top_{count}", family="tree_top_k", params=kv_pairs(k=count)))
        specs.append(FeatureSetSpec(name=f"l1_top_{count}", family="l1_top_k", params=kv_pairs(k=count)))
        if preset != "smoke":
            specs.append(FeatureSetSpec(name=f"intersection_vote_top_{count}_min2", family="intersection_vote_top_k", params=kv_pairs(k=count, min_votes=2)))
            specs.append(FeatureSetSpec(name=f"intersection_vote_top_{count}_min3", family="intersection_vote_top_k", params=kv_pairs(k=count, min_votes=3)))
            specs.append(FeatureSetSpec(name=f"union_vote_top_{count}", family="union_vote_top_k", params=kv_pairs(k=count)))
            specs.append(FeatureSetSpec(name=f"stable_combined_top_{count}", family="stable_combined_top_k", params=kv_pairs(k=count, min_rate=0.50)))
            specs.append(FeatureSetSpec(name=f"stable_combined_top_{count}_strict", family="stable_combined_top_k", params=kv_pairs(k=count, min_rate=0.67)))
            for scheme in ranking_schemes:
                specs.append(
                    FeatureSetSpec(
                        name=f"{scheme.replace('_rank_score', '')}_top_{count}",
                        family="rank_scheme_top_k",
                        params=kv_pairs(k=count, scheme=scheme),
                    )
                )

    if preset != "smoke":
        specs.append(FeatureSetSpec(name="stable_l1_rate_050", family="stable_l1_min_rate", params=kv_pairs(min_rate=0.50, fallback_k=11)))
        specs.append(FeatureSetSpec(name="stable_l1_rate_067", family="stable_l1_min_rate", params=kv_pairs(min_rate=0.67, fallback_k=8)))
        specs.append(FeatureSetSpec(name="stable_l1_rate_080", family="stable_l1_min_rate", params=kv_pairs(min_rate=0.80, fallback_k=8)))
        specs.append(FeatureSetSpec(name="official_plus_combined_5", family="official_plus_combined", params=kv_pairs(k=5)))
        specs.append(FeatureSetSpec(name="official_plus_combined_11", family="official_plus_combined", params=kv_pairs(k=11)))
        specs.append(FeatureSetSpec(name="official_plus_combined_17", family="official_plus_combined", params=kv_pairs(k=17)))

        for block_name, features in sorted(feature_blocks.items()):
            specs.append(FeatureSetSpec(name=f"block_{block_name}", family="explicit", params=(("features", tuple(features)),)))
        block_names = sorted(feature_blocks)
        for combo_size in [2, 3]:
            for combo in combinations(block_names, combo_size):
                combo_features = dedup_features(
                    feature for block_name in combo for feature in feature_blocks.get(block_name, [])
                )
                if combo_features:
                    specs.append(
                        FeatureSetSpec(
                            name=f"block_combo_{'_'.join(combo)}",
                            family="explicit",
                            params=(("features", tuple(combo_features)),),
                        )
                    )
        for block_name, features in sorted(feature_blocks.items()):
            combo = dedup_features(list(features) + [feature for feature in ["diff_elo_pre_match", "diff_recent_win_rate"] if feature in feature_columns])
            if combo:
                specs.append(
                    FeatureSetSpec(
                        name=f"block_{block_name}_plus_elo_form",
                        family="explicit",
                        params=(("features", tuple(combo)),),
                    )
                )

    deduped_specs: list[FeatureSetSpec] = []
    seen_names: set[str] = set()
    for spec in specs:
        if spec.name not in seen_names:
            deduped_specs.append(spec)
            seen_names.add(spec.name)
    return deduped_specs


def build_model_specs(preset: str) -> list[ModelSpec]:
    specs: list[ModelSpec] = [ModelSpec(name="dummy_prior", family="dummy_prior", params=kv_pairs())]

    if preset == "smoke":
        configs = [
            ("logreg_l2_c1_balanced", "logreg_l2", kv_pairs(C=1.0, class_weight="balanced")),
            ("logreg_l2_c3.0_cwbalanced_frozen", "logreg_l2", kv_pairs(C=3.0, class_weight="balanced")),
            ("logreg_l1_c0.3_balanced", "logreg_l1", kv_pairs(C=0.3, class_weight="balanced")),
            ("logreg_l1_c0.1_balanced", "logreg_l1", kv_pairs(C=0.1, class_weight="balanced")),
            ("logreg_elastic_c0.3_r05", "logreg_elasticnet", kv_pairs(C=0.3, class_weight="balanced", l1_ratio=0.5)),
            ("gaussian_nb_1e-8", "gaussian_nb", kv_pairs(var_smoothing=1e-8)),
            ("knn_k5_distance", "knn", kv_pairs(n_neighbors=5, weights="distance")),
            ("svm_rbf_c1_scale", "svm_rbf", kv_pairs(C=1.0, gamma="scale")),
            ("svm_linear_c1_balanced", "svm_linear", kv_pairs(C=1.0, class_weight="balanced")),
            ("rf_n400_dnone_l2", "random_forest", kv_pairs(n_estimators=400, max_depth=None, min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample")),
            ("extratrees_n400_dnone_l2", "extra_trees", kv_pairs(n_estimators=400, max_depth=None, min_samples_leaf=2, max_features="sqrt", class_weight="balanced")),
            ("histgb_lr003_d6_l31", "hist_gradient_boosting", kv_pairs(learning_rate=0.03, max_depth=6, max_leaf_nodes=31, min_samples_leaf=10, max_iter=400)),
        ]
        return [ModelSpec(name=name, family=family, params=params) for name, family, params in configs]

    for c in [0.003, 0.01, 0.03, 0.10, 0.30, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0]:
        for class_weight in [None, "balanced"]:
            cw_name = "balanced" if class_weight else "none"
            specs.append(ModelSpec(name=f"logreg_l2_c{c}_cw{cw_name}", family="logreg_l2", params=kv_pairs(C=c, class_weight=class_weight)))
    specs.append(
        ModelSpec(
            name="logreg_l2_c3.0_cwbalanced_frozen",
            family="logreg_l2",
            params=kv_pairs(C=3.0, class_weight="balanced"),
        )
    )
    for c in [0.003, 0.01, 0.03, 0.10, 0.30, 1.0, 2.0, 3.0, 5.0, 10.0]:
        for class_weight in [None, "balanced"]:
            cw_name = "balanced" if class_weight else "none"
            specs.append(ModelSpec(name=f"logreg_l1_c{c}_cw{cw_name}", family="logreg_l1", params=kv_pairs(C=c, class_weight=class_weight)))
    for c in [0.03, 0.10, 0.30, 1.0, 3.0, 5.0]:
        for l1_ratio in [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80]:
            specs.append(
                ModelSpec(
                    name=f"logreg_elastic_c{c}_r{str(l1_ratio).replace('.', '')}",
                    family="logreg_elasticnet",
                    params=kv_pairs(C=c, class_weight="balanced", l1_ratio=l1_ratio),
                )
            )
    for smoothing in [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
        specs.append(ModelSpec(name=f"gaussian_nb_{smoothing:.0e}", family="gaussian_nb", params=kv_pairs(var_smoothing=smoothing)))
    for n_neighbors in [1, 3, 5, 7, 10, 15, 21]:
        for weights in ["uniform", "distance"]:
            specs.append(ModelSpec(name=f"knn_k{n_neighbors}_{weights}", family="knn", params=kv_pairs(n_neighbors=n_neighbors, weights=weights)))
    for c in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        for gamma in ["scale", "auto"]:
            specs.append(ModelSpec(name=f"svm_rbf_c{c}_{gamma}", family="svm_rbf", params=kv_pairs(C=c, gamma=gamma)))
    for c in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        for class_weight in [None, "balanced"]:
            cw_name = "balanced" if class_weight else "none"
            specs.append(ModelSpec(name=f"svm_linear_c{c}_cw{cw_name}", family="svm_linear", params=kv_pairs(C=c, class_weight=class_weight)))
    for n_estimators, max_depth, min_samples_leaf in [
        (400, None, 2), (700, None, 2), (700, 12, 2), (700, 20, 1), (400, 12, 1), (400, 20, 2),
        (1000, None, 1), (1000, 16, 1), (1400, None, 2), (1400, 24, 1)
    ]:
        specs.append(
            ModelSpec(
                name=f"rf_n{n_estimators}_d{str(max_depth).lower()}_l{min_samples_leaf}",
                family="random_forest",
                params=kv_pairs(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                ),
            )
        )
    for n_estimators, max_depth, min_samples_leaf, max_features in [
        (900, None, 1, "sqrt"), (900, 16, 1, "sqrt"), (1200, None, 2, 0.5),
        (1600, None, 1, "sqrt"), (1600, 24, 1, 0.5)
    ]:
        specs.append(
            ModelSpec(
                name=f"rf_plus_n{n_estimators}_d{str(max_depth).lower()}_l{min_samples_leaf}_f{str(max_features).replace('.', 'p')}",
                family="random_forest",
                params=kv_pairs(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    class_weight="balanced_subsample",
                ),
            )
        )
    for n_estimators, max_depth, min_samples_leaf in [
        (400, None, 2), (700, None, 2), (700, 12, 2), (700, 20, 1), (400, 12, 1), (400, 20, 2),
        (1000, None, 1), (1000, 16, 1), (1400, None, 2), (1400, 24, 1)
    ]:
        specs.append(
            ModelSpec(
                name=f"extratrees_n{n_estimators}_d{str(max_depth).lower()}_l{min_samples_leaf}",
                family="extra_trees",
                params=kv_pairs(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features="sqrt",
                    class_weight="balanced",
                ),
            )
        )
    for n_estimators, max_depth, min_samples_leaf, max_features in [
        (900, None, 1, "sqrt"), (900, 16, 1, "sqrt"), (1200, None, 2, 0.5),
        (1600, None, 1, "sqrt"), (1600, 24, 1, 0.5)
    ]:
        specs.append(
            ModelSpec(
                name=f"extratrees_plus_n{n_estimators}_d{str(max_depth).lower()}_l{min_samples_leaf}_f{str(max_features).replace('.', 'p')}",
                family="extra_trees",
                params=kv_pairs(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    class_weight="balanced",
                ),
            )
        )
    for learning_rate, max_depth, max_leaf_nodes in [
        (0.03, None, 31), (0.03, 6, 31), (0.05, None, 63), (0.05, 6, 63), (0.03, 8, 63), (0.07, 8, 127)
    ]:
        specs.append(
            ModelSpec(
                name=f"histgb_lr{str(learning_rate).replace('.', '')}_d{str(max_depth).lower()}_l{max_leaf_nodes}",
                family="hist_gradient_boosting",
                params=kv_pairs(
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    max_leaf_nodes=max_leaf_nodes,
                    min_samples_leaf=10,
                    max_iter=400,
                ),
            )
        )
    for learning_rate, max_depth, max_leaf_nodes, min_samples_leaf, max_iter in [
        (0.03, 8, 63, 6, 500), (0.05, 8, 63, 6, 600), (0.07, 6, 31, 10, 500),
        (0.03, 10, 127, 4, 800), (0.05, 10, 127, 4, 1000), (0.07, 8, 63, 6, 800)
    ]:
        specs.append(
            ModelSpec(
                name=f"histgb_plus_lr{str(learning_rate).replace('.', '')}_d{str(max_depth).lower()}_l{max_leaf_nodes}_m{max_iter}",
                family="hist_gradient_boosting",
                params=kv_pairs(
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    max_leaf_nodes=max_leaf_nodes,
                    min_samples_leaf=min_samples_leaf,
                    max_iter=max_iter,
                ),
                )
            )
    if LGBMClassifier is not None:
        for n_estimators, learning_rate, max_depth, num_leaves, subsample, colsample in [
            (400, 0.03, 6, 31, 0.9, 0.8),
            (700, 0.03, 8, 63, 0.9, 0.8),
            (900, 0.05, 8, 63, 0.9, 0.8),
            (1200, 0.03, 10, 127, 0.85, 0.75),
            (1600, 0.05, 10, 127, 0.85, 0.75),
        ]:
            specs.append(
                ModelSpec(
                    name=f"lightgbm_n{n_estimators}_lr{str(learning_rate).replace('.', '')}_d{max_depth}_l{num_leaves}",
                    family="lightgbm",
                    params=kv_pairs(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        num_leaves=num_leaves,
                        subsample=subsample,
                        colsample_bytree=colsample,
                    ),
                )
            )
    if XGBClassifier is not None:
        for n_estimators, learning_rate, max_depth, subsample, colsample, reg_lambda in [
            (400, 0.03, 6, 0.9, 0.8, 1.0),
            (700, 0.03, 8, 0.9, 0.8, 1.0),
            (900, 0.05, 8, 0.9, 0.8, 1.0),
            (1200, 0.03, 10, 0.85, 0.75, 1.5),
            (1600, 0.05, 10, 0.85, 0.75, 2.0),
        ]:
            specs.append(
                ModelSpec(
                    name=f"xgboost_n{n_estimators}_lr{str(learning_rate).replace('.', '')}_d{max_depth}",
                    family="xgboost",
                    params=kv_pairs(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        subsample=subsample,
                        colsample_bytree=colsample,
                        reg_lambda=reg_lambda,
                    ),
                )
            )
    if CatBoostClassifier is not None:
        for iterations, learning_rate, depth in [(400, 0.03, 6), (700, 0.03, 8), (900, 0.05, 8), (1200, 0.03, 10), (1600, 0.05, 10)]:
            specs.append(
                ModelSpec(
                    name=f"catboost_i{iterations}_lr{str(learning_rate).replace('.', '')}_d{depth}",
                    family="catboost",
                    params=kv_pairs(
                        iterations=iterations,
                        learning_rate=learning_rate,
                        depth=depth,
                    ),
                )
            )
    return specs


def build_candidate_specs(feature_specs: list[FeatureSetSpec], model_specs: list[ModelSpec]) -> list[CandidateSpec]:
    return [
        CandidateSpec(
            name=f"{feature_spec.name}__{model_spec.name}",
            feature_set_name=feature_spec.name,
            model_name=model_spec.name,
        )
        for feature_spec in feature_specs
        for model_spec in model_specs
    ]


def resolve_feature_set(
    spec: FeatureSetSpec,
    *,
    available_features: list[str],
    official_features: list[str],
    combined_df: pd.DataFrame,
    univariate_df: pd.DataFrame,
    l1_df: pd.DataFrame,
    tree_df: pd.DataFrame,
) -> list[str]:
    params = params_to_dict(spec.params)
    available = dedup_features(available_features)
    official_present = [feature for feature in official_features if feature in available]

    def top_from(frame: pd.DataFrame, column: str, k: int) -> list[str]:
        ordered = [str(item) for item in frame[column].tolist() if str(item) in available]
        return dedup_features(ordered[:k])

    def top_derived_from(frame: pd.DataFrame, *, k: int, scheme: str | None = None) -> list[str]:
        derived_features = [
            feature
            for feature in available
            if feature.startswith("interaction_") or feature.startswith("delta_") or feature.startswith("ratio_") or feature.startswith("abs_")
        ]
        if not derived_features:
            return []
        if scheme is None:
            ordered = [str(item) for item in frame["feature"].tolist() if str(item) in derived_features]
        else:
            if scheme not in frame.columns:
                raise ValueError(f"Esquema de ranking ausente no combined_df: {scheme}")
            ordered = (
                frame.sort_values([scheme, "roc_auc_mean", "log_loss_mean"], ascending=[False, False, True], kind="stable")["feature"]
                .astype(str)
                .tolist()
            )
            ordered = [feature for feature in ordered if feature in derived_features]
        return dedup_features(ordered[:k])

    if spec.family == "all_features":
        return available
    if spec.family == "explicit":
        features = list(params.get("features", []))
        return dedup_features([feature for feature in features if feature in available])
    if spec.family == "derived_top_k":
        selected = top_derived_from(combined_df, k=int(params["k"]))
        return selected if selected else top_from(combined_df, "feature", int(params["k"]))
    if spec.family == "combined_top_k":
        return top_from(combined_df, "feature", int(params["k"]))
    if spec.family == "univariate_top_k":
        return top_from(univariate_df, "feature", int(params["k"]))
    if spec.family == "tree_top_k":
        return top_from(tree_df, "feature", int(params["k"]))
    if spec.family == "l1_top_k":
        return top_from(l1_df, "feature", int(params["k"]))
    if spec.family == "derived_rank_scheme_top_k":
        selected = top_derived_from(combined_df, k=int(params["k"]), scheme=str(params["scheme"]))
        return selected if selected else top_from(combined_df, "feature", int(params["k"]))
    if spec.family == "rank_scheme_top_k":
        scheme = str(params["scheme"])
        k = int(params["k"])
        if scheme not in combined_df.columns:
            raise ValueError(f"Esquema de ranking ausente no combined_df: {scheme}")
        ordered = (
            combined_df.sort_values([scheme, "roc_auc_mean", "log_loss_mean"], ascending=[False, False, True], kind="stable")["feature"]
            .astype(str)
            .tolist()
        )
        ordered = [feature for feature in ordered if feature in available]
        return dedup_features(ordered[:k])
    if spec.family == "block_budget_top_k":
        scheme = str(params.get("scheme", "combined_rank_score"))
        k = int(params["k"])
        if scheme not in combined_df.columns:
            raise ValueError(f"Esquema de ranking ausente no combined_df: {scheme}")

        feature_blocks = infer_feature_blocks(available)
        feature_to_block = {
            str(feature): str(block_name)
            for block_name, features in feature_blocks.items()
            for feature in features
        }
        block_budgets = {
            str(key).replace("budget_", "", 1): int(value)
            for key, value in params.items()
            if str(key).startswith("budget_")
        }
        if not block_budgets:
            raise ValueError("block_budget_top_k exige pelo menos um parametro budget_<bloco>.")

        ordered = (
            combined_df.sort_values([scheme, "roc_auc_mean", "log_loss_mean"], ascending=[False, False, True], kind="stable")["feature"]
            .astype(str)
            .tolist()
        )
        ordered = [feature for feature in ordered if feature in available]

        selected: list[str] = []
        usage_by_block: dict[str, int] = {}
        for feature in ordered:
            block_name = feature_to_block.get(str(feature))
            if block_name is None:
                continue
            budget = block_budgets.get(block_name)
            if budget is None:
                continue
            if usage_by_block.get(block_name, 0) >= int(budget):
                continue
            selected.append(str(feature))
            usage_by_block[block_name] = usage_by_block.get(block_name, 0) + 1
            if len(selected) >= k:
                break

        if len(selected) < k:
            filler = [feature for feature in ordered if feature not in selected]
            selected.extend(filler[: max(0, k - len(selected))])
        return dedup_features(selected[:k])
    if spec.family == "stable_l1_min_rate":
        min_rate = float(params["min_rate"])
        fallback_k = int(params.get("fallback_k", 11))
        stable = [str(item) for item in l1_df.loc[l1_df["l1_frequency_rate"] >= min_rate, "feature"].tolist() if str(item) in available]
        if stable:
            return dedup_features(stable)
        return top_from(combined_df, "feature", fallback_k)
    if spec.family == "stable_combined_top_k":
        min_rate = float(params["min_rate"])
        k = int(params["k"])
        stable_features = set(l1_df.loc[l1_df["l1_frequency_rate"] >= min_rate, "feature"].astype(str).tolist())
        ordered = [str(item) for item in combined_df["feature"].tolist() if str(item) in available and str(item) in stable_features]
        if ordered:
            return dedup_features(ordered[:k])
        return top_from(combined_df, "feature", k)
    if spec.family == "intersection_vote_top_k":
        k = int(params["k"])
        min_votes = int(params["min_votes"])
        sources = {
            "combined": set(top_from(combined_df, "feature", k)),
            "univariate": set(top_from(univariate_df, "feature", k)),
            "tree": set(top_from(tree_df, "feature", k)),
            "l1": set(top_from(l1_df, "feature", k)),
        }
        votes: dict[str, int] = {}
        for source_features in sources.values():
            for feature in source_features:
                votes[feature] = votes.get(feature, 0) + 1
        selected = [feature for feature in combined_df["feature"].astype(str).tolist() if votes.get(feature, 0) >= min_votes and feature in available]
        if selected:
            return dedup_features(selected)
        return top_from(combined_df, "feature", k)
    if spec.family == "union_vote_top_k":
        k = int(params["k"])
        ordered = (
            top_from(combined_df, "feature", k)
            + top_from(univariate_df, "feature", k)
            + top_from(tree_df, "feature", k)
            + top_from(l1_df, "feature", k)
        )
        return dedup_features(ordered)
    if spec.family == "official_plus_combined":
        features = official_present + top_from(combined_df, "feature", int(params["k"]))
        return dedup_features(features)
    if spec.family == "official_plus_derived_top_k":
        features = official_present + top_derived_from(combined_df, k=int(params["k"]))
        if not features:
            features = official_present + top_from(combined_df, "feature", int(params["k"]))
        return dedup_features(features)
    raise ValueError(f"Familia de subset invalida: {spec.family}")


def freeze_feature_space(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    official_features: list[str],
    feature_specs: list[FeatureSetSpec],
    random_state: int,
    preset: str,
    output_dir: Path | None,
) -> dict[str, object]:
    split_specs = build_inner_split_specs(
        frame,
        random_state=random_state,
        fallback_splits=3 if preset == "smoke" else 5,
        fallback_repeats=1 if preset == "smoke" else 2,
    )
    univariate_df = compute_univariate_feature_ranking(frame, feature_columns, split_specs, random_state=random_state)
    l1_df = compute_l1_stability_ranking(frame, feature_columns, split_specs, random_state=random_state)
    tree_df = compute_tree_importance_ranking(frame, feature_columns, random_state=random_state)
    combined_df = combine_feature_rankings(univariate_df, l1_df, tree_df)

    resolved_sets: dict[str, list[str]] = {}
    manifest_rows: list[dict[str, object]] = []
    for spec in feature_specs:
        resolved = resolve_feature_set(
            spec,
            available_features=feature_columns,
            official_features=official_features,
            combined_df=combined_df,
            univariate_df=univariate_df,
            l1_df=l1_df,
            tree_df=tree_df,
        )
        resolved_sets[spec.name] = resolved
        manifest_rows.append(
            {
                "feature_set_name": spec.name,
                "family": spec.family,
                "params_json": json.dumps(params_to_dict(spec.params), ensure_ascii=False),
                "feature_count": int(len(resolved)),
                "features": ",".join(resolved),
            }
        )

    if output_dir is not None:
        ensure_dir(output_dir)
        write_csv(univariate_df, output_dir / "univariate_feature_ranking.csv")
        write_csv(l1_df, output_dir / "l1_stability_ranking.csv")
        write_csv(tree_df, output_dir / "tree_feature_importance_ranking.csv")
        write_csv(combined_df, output_dir / "combined_feature_ranking.csv")
        write_csv(pd.DataFrame(manifest_rows), output_dir / "feature_set_resolution.csv")
        dump_json(
            {
                "inner_split_count": int(len(split_specs)),
                "inner_split_types": sorted({str(item["split_type"]) for item in split_specs}),
                "feature_count": int(len(feature_columns)),
                "official_feature_count_present": int(sum(1 for feature in official_features if feature in feature_columns)),
            },
            output_dir / "feature_space_summary.json",
        )

    return {
        "split_specs": split_specs,
        "univariate_df": univariate_df,
        "l1_df": l1_df,
        "tree_df": tree_df,
        "combined_df": combined_df,
        "feature_sets": resolved_sets,
        "manifest_rows": manifest_rows,
    }


def aggregate_metric_frame(metric_frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    metrics = ["roc_auc", "log_loss", "brier", "accuracy", "precision", "recall", "f1", "ece"]
    summary: dict[str, float] = {}
    for metric in metrics:
        summary[f"{prefix}_{metric}_mean"] = float(metric_frame[metric].mean())
        summary[f"{prefix}_{metric}_std"] = float(metric_frame[metric].std(ddof=0))
    return summary


def aggregate_seed_variance(metric_frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    metrics = ["roc_auc", "log_loss", "brier", "accuracy", "precision", "recall", "f1", "ece"]
    empty_summary: dict[str, float] = {}
    for metric in metrics:
        empty_summary[f"{prefix}_seed_{metric}_std_mean"] = 0.0
        empty_summary[f"{prefix}_seed_{metric}_std_max"] = 0.0
    if "seed" not in metric_frame.columns or metric_frame["seed"].nunique() <= 1:
        return empty_summary
    group_keys = [column for column in ["outer_fold", "split_index"] if column in metric_frame.columns]
    if not group_keys:
        group_keys = ["seed"]
    grouped = metric_frame.groupby(group_keys, dropna=False)[metrics].std(ddof=0).fillna(0.0)
    summary = empty_summary.copy()
    for metric in metrics:
        summary[f"{prefix}_seed_{metric}_std_mean"] = float(grouped[metric].mean())
        summary[f"{prefix}_seed_{metric}_std_max"] = float(grouped[metric].max())
    return summary


def fold_order_key(fold_name: object) -> tuple[int, str]:
    text = str(fold_name)
    try:
        return int(text.rsplit("_", maxsplit=1)[-1]), text
    except Exception:
        return 10**9, text


def rebuild_exploratory_summary_from_fold_metrics(
    *,
    output_dir: Path,
    stage_dirname: str,
    log_path: Path,
) -> pd.DataFrame:
    stage_dir = output_dir / stage_dirname
    fold_path = stage_dir / "exploratory_fold_metrics.csv"
    summary_path = stage_dir / "exploratory_summary.csv"
    candidate_manifest_path = output_dir / "candidate_manifest.csv"
    if not fold_path.exists():
        log(f"[exploratory:{stage_dirname}] fold metrics ausentes; nao foi possivel reconstruir o resumo.", log_path)
        return pd.DataFrame()

    log(f"[exploratory:{stage_dirname}] reconstruindo exploratory_summary.csv a partir de exploratory_fold_metrics.csv", log_path)
    fold_df = pd.read_csv(fold_path)
    manifest_lookup: dict[str, dict[str, object]] = {}
    if candidate_manifest_path.exists():
        manifest_df = pd.read_csv(candidate_manifest_path).drop_duplicates(subset=["candidate_name"], keep="last")
        manifest_lookup = {
            str(row["candidate_name"]): row
            for row in manifest_df.to_dict(orient="records")
        }

    rows: list[dict[str, object]] = []
    for candidate_name, candidate_frame in fold_df.groupby("candidate_name", sort=False):
        metric_frame = candidate_frame.copy()
        metric_frame["_fold_order"] = metric_frame["outer_fold"].map(fold_order_key)
        metric_frame = metric_frame.sort_values(["_fold_order", "seed"], kind="stable").reset_index(drop=True)
        fold_resolution = metric_frame.drop_duplicates(subset=["outer_fold"], keep="last").reset_index(drop=True)
        manifest_row = manifest_lookup.get(str(candidate_name), {})
        first_row = metric_frame.iloc[0]
        feature_counts = fold_resolution["feature_count"].astype(int).tolist()
        summary = {
            "candidate_name": str(candidate_name),
            "feature_set_name": manifest_row.get("feature_set_name", first_row["feature_set_name"]),
            "feature_family": manifest_row.get("feature_family", first_row["feature_family"]),
            "feature_params_json": manifest_row.get("feature_params_json", "{}"),
            "model_name": manifest_row.get("model_name", first_row["model_name"]),
            "model_family": manifest_row.get("model_family", first_row["model_family"]),
            "model_params_json": manifest_row.get("model_params_json", "{}"),
            "temporal_cv_fold_count": int(metric_frame["outer_fold"].nunique()),
            "temporal_cv_eval_count": int(len(metric_frame)),
            "stochastic_seed_count": int(metric_frame["seed"].nunique()),
            "resolved_feature_count_min": int(min(feature_counts)),
            "resolved_feature_count_mean": float(np.mean(feature_counts)),
            "resolved_feature_count_max": int(max(feature_counts)),
            "resolved_feature_counts": ";".join(
                f"{row['outer_fold']}:{int(row['feature_count'])}"
                for row in fold_resolution.to_dict(orient="records")
            ),
            "resolved_features_last_fold": str(fold_resolution.iloc[-1]["features"]),
        }
        summary.update(aggregate_metric_frame(metric_frame, "temporal_cv"))
        summary.update(aggregate_seed_variance(metric_frame, "temporal_cv"))
        rows.append(summary)

    summary_df = pd.DataFrame(rows)
    write_csv(summary_df, summary_path)
    return summary_df


def load_exploratory_summary_with_repair(*, output_dir: Path, stage_dirname: str, log_path: Path) -> pd.DataFrame:
    summary_path = output_dir / stage_dirname / "exploratory_summary.csv"
    if not summary_path.exists():
        return rebuild_exploratory_summary_from_fold_metrics(output_dir=output_dir, stage_dirname=stage_dirname, log_path=log_path)
    try:
        return pd.read_csv(summary_path)
    except pd.errors.ParserError as exc:
        log(f"[exploratory:{stage_dirname}] resumo exploratorio corrompido ({exc}); iniciando reconstrucao.", log_path)
        return rebuild_exploratory_summary_from_fold_metrics(output_dir=output_dir, stage_dirname=stage_dirname, log_path=log_path)


def prepare_exploratory_folds(
    *,
    train_df: pd.DataFrame,
    temporal_folds: list[TemporalFold],
    feature_columns: list[str],
    official_features: list[str],
    feature_specs: list[FeatureSetSpec],
    random_state: int,
    preset: str,
    output_dir: Path,
    log_path: Path,
) -> list[dict[str, object]]:
    fold_payloads: list[dict[str, object]] = []
    feature_manifest_rows: list[dict[str, object]] = []

    for fold in temporal_folds:
        fold_train, fold_valid = fold_to_frames(train_df, fold)
        fold_dir = output_dir / fold.fold_name
        log(
            f"[feature-space] {fold.fold_name}: train={len(fold_train)} valid={len(fold_valid)} seasons={','.join(fold.train_seasons)}->{fold.valid_season}",
            log_path,
        )
        feature_space = freeze_feature_space(
            fold_train,
            feature_columns=feature_columns,
            official_features=official_features,
            feature_specs=feature_specs,
            random_state=random_state,
            preset=preset,
            output_dir=fold_dir,
        )
        for row in feature_space["manifest_rows"]:
            feature_manifest_rows.append({"outer_fold": fold.fold_name, **row})
        fold_payloads.append(
            {
                "fold": fold,
                "fold_train": fold_train,
                "fold_valid": fold_valid,
                "feature_space": feature_space,
            }
        )

    write_csv(pd.DataFrame(feature_manifest_rows), output_dir / "exploratory_feature_set_manifest.csv")
    return fold_payloads


def candidate_manifest_frame(
    *,
    candidates: list[CandidateSpec],
    feature_lookup: dict[str, FeatureSetSpec],
    model_lookup: dict[str, ModelSpec],
    stochastic_seeds: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        feature_spec = feature_lookup[candidate.feature_set_name]
        model_spec = model_lookup[candidate.model_name]
        rows.append(
            {
                "candidate_name": candidate.name,
                "feature_set_name": feature_spec.name,
                "feature_family": feature_spec.family,
                "feature_params_json": json.dumps(params_to_dict(feature_spec.params), ensure_ascii=False),
                "model_name": model_spec.name,
                "model_family": model_spec.family,
                "model_params_json": json.dumps(params_to_dict(model_spec.params), ensure_ascii=False),
                "stochastic_seed_count": int(len(resolve_seed_list(model_spec.family, stochastic_seeds, DEFAULT_RANDOM_STATE))),
            }
        )
    return pd.DataFrame(rows)


def evaluate_candidate_exploratory(
    candidate: CandidateSpec,
    *,
    fold_payloads: list[dict[str, object]],
    feature_lookup: dict[str, FeatureSetSpec],
    model_lookup: dict[str, ModelSpec],
    random_state: int,
    stochastic_seeds: Sequence[int],
) -> dict[str, object]:
    feature_spec = feature_lookup[candidate.feature_set_name]
    model_spec = model_lookup[candidate.model_name]
    model_params = params_to_dict(model_spec.params)
    seed_list = resolve_seed_list(model_spec.family, stochastic_seeds, random_state)

    fold_rows: list[dict[str, object]] = []
    feature_count_rows: list[int] = []
    feature_count_trace: list[str] = []

    for payload in fold_payloads:
        fold = payload["fold"]
        fold_train = payload["fold_train"]
        fold_valid = payload["fold_valid"]
        feature_space = payload["feature_space"]
        features = list(feature_space["feature_sets"][candidate.feature_set_name])
        if not features:
            raise ValueError(f"Subset vazio para {candidate.name} em {fold.fold_name}.")
        feature_count_rows.append(len(features))
        feature_count_trace.append(f"{fold.fold_name}:{len(features)}")
        for seed in seed_list:
            estimator = build_pipeline(model_spec.family, model_params, int(seed))
            estimator.fit(fold_train[features], fold_train["win_target"].astype(int))
            probabilities = predict_positive_probability(estimator, fold_valid, features)
            metrics = evaluate_probabilities(fold_valid["win_target"].astype(int).to_numpy(), probabilities, threshold=0.5)
            fold_rows.append(
                {
                    "candidate_name": candidate.name,
                    "outer_fold": fold.fold_name,
                    "seed": int(seed),
                    "train_seasons": ",".join(fold.train_seasons),
                    "valid_season": fold.valid_season,
                    "feature_set_name": feature_spec.name,
                    "feature_family": feature_spec.family,
                    "model_name": model_spec.name,
                    "model_family": model_spec.family,
                    "feature_count": int(len(features)),
                    "features": ",".join(features),
                    **metrics,
                }
            )

    metric_frame = pd.DataFrame(fold_rows)
    summary = {
        "candidate_name": candidate.name,
        "feature_set_name": feature_spec.name,
        "feature_family": feature_spec.family,
        "feature_params_json": json.dumps(params_to_dict(feature_spec.params), ensure_ascii=False),
        "model_name": model_spec.name,
        "model_family": model_spec.family,
        "model_params_json": json.dumps(model_params, ensure_ascii=False),
        "temporal_cv_fold_count": int(metric_frame["outer_fold"].nunique()),
        "temporal_cv_eval_count": int(len(fold_rows)),
        "stochastic_seed_count": int(len(seed_list)),
        "resolved_feature_count_min": int(min(feature_count_rows)),
        "resolved_feature_count_mean": float(np.mean(feature_count_rows)),
        "resolved_feature_count_max": int(max(feature_count_rows)),
        "resolved_feature_counts": ";".join(feature_count_trace),
        "resolved_features_last_fold": str(metric_frame.iloc[-1]["features"]),
    }
    summary.update(aggregate_metric_frame(metric_frame, "temporal_cv"))
    summary.update(aggregate_seed_variance(metric_frame, "temporal_cv"))
    return {"summary": summary, "fold_rows": fold_rows}


def sort_leaderboard(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    sorted_frame = frame.copy()
    if "rank" in sorted_frame.columns:
        sorted_frame = sorted_frame.drop(columns=["rank"])
    sorted_frame = sorted_frame.sort_values(
        PRIMARY_SORT_COLUMNS + ["candidate_name"],
        ascending=PRIMARY_SORT_ASCENDING + [True],
        kind="stable",
    ).reset_index(drop=True)
    sorted_frame.insert(0, "rank", np.arange(1, len(sorted_frame) + 1))
    return sorted_frame


def run_exploratory_search(
    *,
    output_dir: Path,
    candidates: list[CandidateSpec],
    feature_lookup: dict[str, FeatureSetSpec],
    model_lookup: dict[str, ModelSpec],
    fold_payloads: list[dict[str, object]],
    workers: int,
    batch_size: int,
    random_state: int,
    stochastic_seeds: Sequence[int],
    resume: bool,
    log_path: Path,
    stage_dirname: str = "exploratory",
    stage_label: str = "exploratory",
) -> pd.DataFrame:
    exploratory_dir = output_dir / stage_dirname
    ensure_dir(exploratory_dir)
    summary_path = exploratory_dir / "exploratory_summary.csv"
    fold_path = exploratory_dir / "exploratory_fold_metrics.csv"

    completed = set()
    if resume:
        completed = load_completed_candidates(summary_path)
        fold_completed = load_completed_candidates(fold_path)
        if len(fold_completed) > len(completed):
            completed = fold_completed
    pending = [candidate for candidate in candidates if candidate.name not in completed]
    log(
        f"[{stage_label}] candidatos totais={len(candidates)} completos={len(completed)} pendentes={len(pending)} workers={workers}",
        log_path,
    )

    for batch_index, batch in enumerate(batched(pending, max(1, batch_size)), start=1):
        results = Parallel(n_jobs=workers, backend="threading")(
            delayed(evaluate_candidate_exploratory)(
                candidate,
                fold_payloads=fold_payloads,
                feature_lookup=feature_lookup,
                model_lookup=model_lookup,
                random_state=random_state,
                stochastic_seeds=stochastic_seeds,
            )
            for candidate in batch
        )
        summary_rows = [item["summary"] for item in results]
        fold_rows = [row for item in results for row in item["fold_rows"]]
        append_csv(summary_rows, summary_path)
        append_csv(fold_rows, fold_path)
        log(f"[{stage_label}] lote={batch_index} candidatos_processados={len(summary_rows)}", log_path)

    summary_df = load_exploratory_summary_with_repair(output_dir=output_dir, stage_dirname=stage_dirname, log_path=log_path)
    leaderboard = sort_leaderboard(summary_df)
    write_csv(leaderboard, exploratory_dir / "exploratory_leaderboard.csv")
    return leaderboard


def select_stage2_candidate_names(
    stage1_df: pd.DataFrame,
    *,
    top_global: int,
    top_per_feature_family: int,
    top_per_model_family: int,
    forced_candidate_names: Sequence[str],
) -> list[str]:
    leaderboard = sort_leaderboard(stage1_df)
    selected_names: list[str] = []
    seen: set[str] = set()

    def add_names(values: Iterable[str]) -> None:
        for value in values:
            candidate_name = str(value)
            if candidate_name and candidate_name not in seen:
                selected_names.append(candidate_name)
                seen.add(candidate_name)

    add_names(leaderboard.head(top_global)["candidate_name"].astype(str).tolist())
    if "feature_family" in leaderboard.columns:
        for _, group in leaderboard.groupby("feature_family", sort=False):
            add_names(group.head(top_per_feature_family)["candidate_name"].astype(str).tolist())
    if "model_family" in leaderboard.columns:
        for _, group in leaderboard.groupby("model_family", sort=False):
            add_names(group.head(top_per_model_family)["candidate_name"].astype(str).tolist())
    add_names([name for name in forced_candidate_names if name in set(leaderboard["candidate_name"].astype(str))])
    return selected_names


def run_staged_search(
    *,
    output_dir: Path,
    candidates: list[CandidateSpec],
    feature_lookup: dict[str, FeatureSetSpec],
    model_lookup: dict[str, ModelSpec],
    fold_payloads: list[dict[str, object]],
    workers: int,
    batch_size: int,
    random_state: int,
    stochastic_seeds: Sequence[int],
    screening_seeds: Sequence[int],
    stage1_top_global: int,
    stage1_top_per_feature_family: int,
    stage1_top_per_model_family: int,
    forced_candidate_names: Sequence[str],
    resume: bool,
    log_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    stage1_leaderboard = run_exploratory_search(
        output_dir=output_dir,
        candidates=candidates,
        feature_lookup=feature_lookup,
        model_lookup=model_lookup,
        fold_payloads=fold_payloads,
        workers=workers,
        batch_size=batch_size,
        random_state=random_state,
        stochastic_seeds=screening_seeds,
        resume=resume,
        log_path=log_path,
        stage_dirname="exploratory/stage1_screen",
        stage_label="stage1",
    )
    stage2_names = select_stage2_candidate_names(
        stage1_leaderboard,
        top_global=stage1_top_global,
        top_per_feature_family=stage1_top_per_feature_family,
        top_per_model_family=stage1_top_per_model_family,
        forced_candidate_names=forced_candidate_names,
    )
    stage2_candidates = [candidate for candidate in candidates if candidate.name in set(stage2_names)]
    stage2_manifest = pd.DataFrame({"candidate_name": stage2_names})
    write_csv(stage2_manifest, output_dir / "exploratory" / "stage2_selected_candidates.csv")
    log(f"[stage2] candidatos selecionados para refinamento={len(stage2_candidates)}", log_path)

    stage2_leaderboard = run_exploratory_search(
        output_dir=output_dir,
        candidates=stage2_candidates,
        feature_lookup=feature_lookup,
        model_lookup=model_lookup,
        fold_payloads=fold_payloads,
        workers=workers,
        batch_size=batch_size,
        random_state=random_state,
        stochastic_seeds=stochastic_seeds,
        resume=resume,
        log_path=log_path,
        stage_dirname="exploratory/stage2_refine",
        stage_label="stage2",
    )
    final_leaderboard = sort_leaderboard(stage2_leaderboard)
    write_csv(stage1_leaderboard, output_dir / "exploratory" / "stage1_leaderboard.csv")
    write_csv(final_leaderboard, output_dir / "exploratory" / "exploratory_leaderboard.csv")
    return final_leaderboard, stage2_names


def run_stage3_shortlist_screen(
    *,
    output_dir: Path,
    stage2_leaderboard: pd.DataFrame,
    train_df: pd.DataFrame,
    feature_columns: list[str],
    official_features: list[str],
    feature_specs: list[FeatureSetSpec],
    model_lookup: dict[str, ModelSpec],
    stochastic_seeds: Sequence[int],
    random_state: int,
    preset: str,
    top_global: int,
    top_per_feature_family: int,
    top_per_model_family: int,
    forced_candidate_names: Sequence[str],
    repeated_cv_splits: int,
    repeated_cv_repeats: int,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame]:
    stage3_names = select_stage2_candidate_names(
        stage2_leaderboard,
        top_global=top_global,
        top_per_feature_family=top_per_feature_family,
        top_per_model_family=top_per_model_family,
        forced_candidate_names=forced_candidate_names,
    )
    stage3_pool = stage2_leaderboard.loc[
        stage2_leaderboard["candidate_name"].astype(str).isin(set(stage3_names))
    ].copy()
    stage3_pool = sort_leaderboard(stage3_pool)
    write_csv(pd.DataFrame({"candidate_name": stage3_names}), output_dir / "exploratory" / "stage3_selected_candidates.csv")
    write_csv(stage3_pool, output_dir / "exploratory" / "stage3_pool_leaderboard.csv")

    stage3_feature_space = freeze_feature_space(
        train_df,
        feature_columns=feature_columns,
        official_features=official_features,
        feature_specs=feature_specs,
        random_state=random_state,
        preset=preset,
        output_dir=output_dir / "exploratory" / "stage3_feature_space",
    )
    finalist_feature_map: dict[str, list[str]] = {}
    for row in stage3_pool.to_dict(orient="records"):
        finalist_feature_map[str(row["candidate_name"])] = list(stage3_feature_space["feature_sets"][str(row["feature_set_name"])])

    repeated_fold_df, repeated_summary_df = repeated_cv_finalists(
        train_df=train_df,
        shortlist=stage3_pool,
        finalist_feature_map=finalist_feature_map,
        model_lookup=model_lookup,
        random_state=random_state,
        stochastic_seeds=stochastic_seeds,
        n_splits=repeated_cv_splits,
        n_repeats=repeated_cv_repeats,
    )
    write_csv(repeated_fold_df, output_dir / "exploratory" / "stage3_repeated_cv_fold_metrics.csv")
    write_csv(repeated_summary_df, output_dir / "exploratory" / "stage3_repeated_cv_summary.csv")

    merged = stage3_pool.merge(repeated_summary_df, on="candidate_name", how="left", suffixes=("", "_stage3"))
    merged["temporal_auc_norm"] = normalize_series(merged["temporal_cv_roc_auc_mean"])
    merged["repeated_auc_norm"] = normalize_series(merged["repeated_cv_roc_auc_mean"])
    merged["temporal_logloss_inv_norm"] = 1.0 - normalize_series(merged["temporal_cv_log_loss_mean"])
    merged["repeated_logloss_inv_norm"] = 1.0 - normalize_series(merged["repeated_cv_log_loss_mean"])
    merged["temporal_brier_inv_norm"] = 1.0 - normalize_series(merged["temporal_cv_brier_mean"])
    merged["repeated_brier_inv_norm"] = 1.0 - normalize_series(merged["repeated_cv_brier_mean"])
    merged["temporal_seed_stability_norm"] = 1.0 - normalize_series(merged["temporal_cv_seed_roc_auc_std_mean"])
    merged["repeated_seed_stability_norm"] = 1.0 - normalize_series(merged["repeated_cv_seed_roc_auc_std_mean"])
    merged["stage3_shortlist_score"] = (
        0.30 * merged["temporal_auc_norm"]
        + 0.25 * merged["repeated_auc_norm"]
        + 0.12 * merged["temporal_logloss_inv_norm"]
        + 0.12 * merged["repeated_logloss_inv_norm"]
        + 0.07 * merged["temporal_brier_inv_norm"]
        + 0.07 * merged["repeated_brier_inv_norm"]
        + 0.03 * merged["temporal_seed_stability_norm"]
        + 0.04 * merged["repeated_seed_stability_norm"]
    )
    stage3_leaderboard = merged.sort_values(
        [
            "stage3_shortlist_score",
            "repeated_cv_roc_auc_mean",
            "temporal_cv_roc_auc_mean",
            "repeated_cv_log_loss_mean",
            "temporal_cv_log_loss_mean",
            "candidate_name",
        ],
        ascending=[False, False, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    stage3_leaderboard.insert(0, "stage3_rank", np.arange(1, len(stage3_leaderboard) + 1))
    write_csv(stage3_leaderboard, output_dir / "exploratory" / "stage3_shortlist_leaderboard.csv")
    return stage3_leaderboard, stage3_names, repeated_fold_df, repeated_summary_df


def freeze_shortlist(
    exploratory_df: pd.DataFrame,
    *,
    shortlist_size: int,
    output_path: Path,
    forced_candidate_names: Sequence[str] = (),
) -> pd.DataFrame:
    leaderboard = sort_leaderboard(exploratory_df)
    shortlist = leaderboard.head(shortlist_size).copy()
    if forced_candidate_names:
        forced = leaderboard.loc[leaderboard["candidate_name"].astype(str).isin({str(name) for name in forced_candidate_names})]
        shortlist = pd.concat([shortlist, forced], ignore_index=True)
        shortlist = shortlist.drop_duplicates(subset=["candidate_name"], keep="first")
        shortlist = sort_leaderboard(shortlist)
    shortlist = shortlist.reset_index(drop=True)
    shortlist.insert(0, "shortlist_rank", np.arange(1, len(shortlist) + 1))
    write_csv(shortlist, output_path)
    return shortlist


def repeated_cv_finalists(
    *,
    train_df: pd.DataFrame,
    shortlist: pd.DataFrame,
    finalist_feature_map: dict[str, list[str]],
    model_lookup: dict[str, ModelSpec],
    random_state: int,
    stochastic_seeds: Sequence[int],
    n_splits: int,
    n_repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = train_df["win_target"].astype(int).to_numpy()
    min_class_count = int(pd.Series(y).value_counts().min())
    n_splits = min(max(2, n_splits), min_class_count)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    split_plan = list(cv.split(np.zeros(len(train_df)), y))

    fold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for shortlist_row in shortlist.to_dict(orient="records"):
        candidate_name = str(shortlist_row["candidate_name"])
        model_spec = model_lookup[str(shortlist_row["model_name"])]
        features = finalist_feature_map[candidate_name]
        model_params = params_to_dict(model_spec.params)
        seed_list = resolve_seed_list(model_spec.family, stochastic_seeds, random_state)
        local_rows: list[dict[str, object]] = []
        for split_index, (train_idx, valid_idx) in enumerate(split_plan, start=1):
            split_train = train_df.iloc[np.asarray(train_idx, dtype=int)]
            split_valid = train_df.iloc[np.asarray(valid_idx, dtype=int)]
            for seed in seed_list:
                estimator = build_pipeline(model_spec.family, model_params, int(seed))
                estimator.fit(split_train[features], split_train["win_target"].astype(int))
                probabilities = predict_positive_probability(estimator, split_valid, features)
                metrics = evaluate_probabilities(split_valid["win_target"].astype(int).to_numpy(), probabilities, threshold=0.5)
                row = {
                    "candidate_name": candidate_name,
                    "split_index": split_index,
                    "seed": int(seed),
                    "feature_count": int(len(features)),
                    "features": ",".join(features),
                    **metrics,
                }
                local_rows.append(row)
                fold_rows.append(row)
        metric_frame = pd.DataFrame(local_rows)
        summary = {
            "candidate_name": candidate_name,
            "repeated_cv_split_count": int(metric_frame["split_index"].nunique()),
            "repeated_cv_eval_count": int(len(local_rows)),
            "stochastic_seed_count": int(len(seed_list)),
            "feature_count": int(len(features)),
        }
        summary.update(aggregate_metric_frame(metric_frame, "repeated_cv"))
        summary.update(aggregate_seed_variance(metric_frame, "repeated_cv"))
        summary_rows.append(summary)

    return pd.DataFrame(fold_rows), sort_leaderboard(pd.DataFrame(summary_rows).rename(columns={
        "repeated_cv_roc_auc_mean": "temporal_cv_roc_auc_mean",
        "repeated_cv_log_loss_mean": "temporal_cv_log_loss_mean",
        "repeated_cv_brier_mean": "temporal_cv_brier_mean",
        "repeated_cv_accuracy_mean": "temporal_cv_accuracy_mean",
    })).rename(columns={
        "temporal_cv_roc_auc_mean": "repeated_cv_roc_auc_mean",
        "temporal_cv_log_loss_mean": "repeated_cv_log_loss_mean",
        "temporal_cv_brier_mean": "repeated_cv_brier_mean",
        "temporal_cv_accuracy_mean": "repeated_cv_accuracy_mean",
    })


def time_ordered_calibration_split(train_df: pd.DataFrame, fraction: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = train_df.sort_values(["match_date", "season_label", "team_name", "opponent_name"], kind="stable").reset_index(drop=True)
    dates = pd.Series(ordered["match_date"].dropna().unique()).sort_values().tolist()
    if len(dates) < 2:
        raise ValueError("Nao ha datas suficientes para separar treino e calibracao.")
    target_index = max(1, int(len(dates) * (1.0 - fraction)))
    target_index = min(target_index, len(dates) - 1)
    for cutoff in dates[target_index:]:
        fit_train = ordered.loc[ordered["match_date"] < cutoff].copy()
        calib = ordered.loc[ordered["match_date"] >= cutoff].copy()
        if fit_train.empty or calib.empty:
            continue
        if fit_train["win_target"].nunique() < 2 or calib["win_target"].nunique() < 2:
            continue
        return fit_train, calib
    raise ValueError("Nao foi possivel gerar um split temporal valido para calibracao.")


def temporal_oof_predictions(
    *,
    train_df: pd.DataFrame,
    features: list[str],
    model_spec: ModelSpec,
    random_state: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    temporal_folds = build_temporal_folds(train_df)
    probabilities = np.full(len(train_df), np.nan, dtype=float)
    rows: list[dict[str, object]] = []
    model_params = params_to_dict(model_spec.params)

    for fold in temporal_folds:
        fold_train, fold_valid = fold_to_frames(train_df, fold)
        valid_index = fold_valid.index.to_numpy(dtype=int)
        estimator = build_pipeline(model_spec.family, model_params, random_state)
        estimator.fit(fold_train[features], fold_train["win_target"].astype(int))
        fold_prob = predict_positive_probability(estimator, fold_valid, features)
        probabilities[valid_index] = fold_prob
        rows.append(
            {
                "outer_fold": fold.fold_name,
                "valid_season": fold.valid_season,
                "rows": int(len(fold_valid)),
                "roc_auc": float(roc_auc_score(fold_valid["win_target"].astype(int), fold_prob)),
                "log_loss": float(log_loss(fold_valid["win_target"].astype(int), fold_prob, labels=[0, 1])),
                "brier": float(brier_score_loss(fold_valid["win_target"].astype(int), fold_prob)),
            }
        )
    return probabilities, pd.DataFrame(rows)


def threshold_objective_key(row: dict[str, float], objective: str) -> tuple[float, float, float, float]:
    if objective == "accuracy":
        return (row["accuracy"], row["f1"], row["precision"], -abs(row["threshold"] - 0.5))
    if objective == "precision":
        return (row["precision"], row["f1"], row["accuracy"], -abs(row["threshold"] - 0.5))
    if objective == "recall":
        return (row["recall"], row["f1"], row["accuracy"], -abs(row["threshold"] - 0.5))
    return (row["f1"], row["accuracy"], row["precision"], -abs(row["threshold"] - 0.5))


def select_best_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    objectives: Sequence[str],
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    rows: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        metrics = binary_metrics(y_true, y_prob, threshold=float(threshold))
        row = {"threshold": float(threshold), **metrics}
        rows.append(row)
    best_rows: list[dict[str, float]] = []
    grid_df = pd.DataFrame(rows)
    for objective in objectives:
        best_row = max(rows, key=lambda row: threshold_objective_key(row, objective))
        best_rows.append(
            {
                "objective": str(objective),
                "threshold": float(best_row["threshold"]),
                "accuracy": float(best_row["accuracy"]),
                "precision": float(best_row["precision"]),
                "recall": float(best_row["recall"]),
                "f1": float(best_row["f1"]),
            }
        )
    return grid_df, best_rows


def bootstrap_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float,
    runs: int,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    sample_size = len(y_true)
    for run in range(1, runs + 1):
        indices = rng.integers(0, sample_size, size=sample_size)
        sample_y = y_true[indices]
        if len(np.unique(sample_y)) < 2:
            continue
        sample_prob = y_prob[indices]
        metrics = evaluate_probabilities(sample_y, sample_prob, threshold=threshold)
        rows.append({"bootstrap_run": run, **metrics})
    detail = pd.DataFrame(rows)
    summary: dict[str, float] = {}
    if detail.empty:
        return detail, summary
    for metric in ["roc_auc", "log_loss", "brier", "accuracy", "precision", "recall", "f1", "ece"]:
        summary[f"{metric}_bootstrap_mean"] = float(detail[metric].mean())
        summary[f"{metric}_bootstrap_std"] = float(detail[metric].std(ddof=0))
        summary[f"{metric}_bootstrap_ci_low"] = float(detail[metric].quantile(0.025))
        summary[f"{metric}_bootstrap_ci_high"] = float(detail[metric].quantile(0.975))
    return detail, summary


def pairwise_bootstrap_differences(
    *,
    y_true: np.ndarray,
    left_name: str,
    left_prob: np.ndarray,
    right_name: str,
    right_prob: np.ndarray,
    runs: int,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    sample_size = len(y_true)
    for run in range(1, runs + 1):
        indices = rng.integers(0, sample_size, size=sample_size)
        sample_y = y_true[indices]
        if len(np.unique(sample_y)) < 2:
            continue
        left_sample = left_prob[indices]
        right_sample = right_prob[indices]
        rows.append(
            {
                "bootstrap_run": run,
                "delta_roc_auc": float(roc_auc_score(sample_y, left_sample) - roc_auc_score(sample_y, right_sample)),
                "delta_log_loss": float(log_loss(sample_y, left_sample, labels=[0, 1]) - log_loss(sample_y, right_sample, labels=[0, 1])),
                "delta_brier": float(brier_score_loss(sample_y, left_sample) - brier_score_loss(sample_y, right_sample)),
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, {}
    summary = {
        "left_candidate": left_name,
        "right_candidate": right_name,
        "delta_roc_auc_mean": float(detail["delta_roc_auc"].mean()),
        "delta_roc_auc_ci_low": float(detail["delta_roc_auc"].quantile(0.025)),
        "delta_roc_auc_ci_high": float(detail["delta_roc_auc"].quantile(0.975)),
        "delta_roc_auc_p_left_le_right": float((detail["delta_roc_auc"] <= 0.0).mean()),
        "delta_log_loss_mean": float(detail["delta_log_loss"].mean()),
        "delta_log_loss_ci_low": float(detail["delta_log_loss"].quantile(0.025)),
        "delta_log_loss_ci_high": float(detail["delta_log_loss"].quantile(0.975)),
        "delta_brier_mean": float(detail["delta_brier"].mean()),
        "delta_brier_ci_low": float(detail["delta_brier"].quantile(0.025)),
        "delta_brier_ci_high": float(detail["delta_brier"].quantile(0.975)),
    }
    return detail, summary


def evaluate_temporal_feature_configuration(
    *,
    train_df: pd.DataFrame,
    features: list[str],
    model_spec: ModelSpec,
    stochastic_seeds: Sequence[int],
    random_state: int,
) -> dict[str, float]:
    temporal_folds = build_temporal_folds(train_df)
    model_params = params_to_dict(model_spec.params)
    seed_list = resolve_seed_list(model_spec.family, stochastic_seeds, random_state)
    rows: list[dict[str, object]] = []
    for fold in temporal_folds:
        fold_train, fold_valid = fold_to_frames(train_df, fold)
        for seed in seed_list:
            estimator = build_pipeline(model_spec.family, model_params, int(seed))
            estimator.fit(fold_train[features], fold_train["win_target"].astype(int))
            probabilities = predict_positive_probability(estimator, fold_valid, features)
            metrics = evaluate_probabilities(fold_valid["win_target"].astype(int).to_numpy(), probabilities, threshold=0.5)
            rows.append({"outer_fold": fold.fold_name, "seed": int(seed), **metrics})
    metric_frame = pd.DataFrame(rows)
    summary = {
        "temporal_cv_fold_count": int(metric_frame["outer_fold"].nunique()),
        "temporal_cv_eval_count": int(len(metric_frame)),
        "stochastic_seed_count": int(len(seed_list)),
    }
    summary.update(aggregate_metric_frame(metric_frame, "temporal_cv"))
    summary.update(aggregate_seed_variance(metric_frame, "temporal_cv"))
    return summary


def run_ablation_study(
    *,
    output_dir: Path,
    shortlist: pd.DataFrame,
    finalist_feature_map: dict[str, list[str]],
    model_lookup: dict[str, ModelSpec],
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    stochastic_seeds: Sequence[int],
    ablation_top_n: int,
    random_state: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = shortlist.head(ablation_top_n).to_dict(orient="records")
    y_holdout = holdout_df["win_target"].astype(int).to_numpy()
    for offset, candidate_row in enumerate(selected, start=1):
        candidate_name = str(candidate_row["candidate_name"])
        model_spec = model_lookup[str(candidate_row["model_name"])]
        base_features = finalist_feature_map[candidate_name]
        feature_blocks = infer_feature_blocks(base_features)
        configs: list[tuple[str, str, list[str]]] = [("baseline", "all_blocks", base_features)]
        for block_name, block_features in sorted(feature_blocks.items()):
            minus_features = [feature for feature in base_features if feature not in set(block_features)]
            only_features = [feature for feature in base_features if feature in set(block_features)]
            if minus_features:
                configs.append(("minus_block", block_name, minus_features))
            if only_features:
                configs.append(("only_block", block_name, only_features))

        for config_index, (config_type, block_name, features) in enumerate(configs, start=1):
            temporal_summary = evaluate_temporal_feature_configuration(
                train_df=train_df,
                features=features,
                model_spec=model_spec,
                stochastic_seeds=stochastic_seeds,
                random_state=random_state + offset + config_index,
            )
            seed_list = resolve_seed_list(model_spec.family, stochastic_seeds, random_state + offset + config_index)
            holdout_prob_list: list[np.ndarray] = []
            for seed in seed_list:
                estimator = build_pipeline(model_spec.family, params_to_dict(model_spec.params), int(seed))
                estimator.fit(train_df[features], train_df["win_target"].astype(int))
                holdout_prob_list.append(clip_probabilities(predict_positive_probability(estimator, holdout_df, features)))
            holdout_prob = clip_probabilities(np.mean(np.vstack(holdout_prob_list), axis=0))
            holdout_metrics = evaluate_probabilities(y_holdout, holdout_prob, threshold=0.5)
            rows.append(
                {
                    "candidate_name": candidate_name,
                    "config_type": config_type,
                    "block_name": block_name,
                    "feature_count": int(len(features)),
                    "features": ",".join(features),
                    **temporal_summary,
                    **{f"holdout_{key}": value for key, value in holdout_metrics.items()},
                }
            )
    frame = pd.DataFrame(rows)
    write_csv(frame, output_dir / "ablation_study.csv")
    return frame


def build_finalist_ensembles(
    *,
    confirmatory_dir: Path,
    shortlist: pd.DataFrame,
    probability_registry: dict[str, np.ndarray],
    oof_registry: dict[str, np.ndarray],
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    ensemble_top_n: int,
    random_state: int,
) -> pd.DataFrame:
    finalist_names = [str(row["candidate_name"]) for row in shortlist.head(ensemble_top_n).to_dict(orient="records")]
    if len(finalist_names) < 2:
        frame = pd.DataFrame()
        write_csv(frame, confirmatory_dir / "ensemble_results.csv")
        return frame

    y_holdout = holdout_df["win_target"].astype(int).to_numpy()
    rows: list[dict[str, object]] = []

    holdout_matrix = np.column_stack([probability_registry[name] for name in finalist_names])
    mean_holdout = clip_probabilities(np.mean(holdout_matrix, axis=1))
    rows.append(
        {
            "ensemble_name": f"mean_ensemble_top{len(finalist_names)}",
            "members": ",".join(finalist_names),
            **evaluate_probabilities(y_holdout, mean_holdout, threshold=0.5),
        }
    )

    oof_matrix = np.column_stack([oof_registry[name] for name in finalist_names])
    valid_mask = ~np.isnan(oof_matrix).any(axis=1)
    if valid_mask.sum() >= 50 and len(np.unique(train_df.loc[valid_mask, "win_target"].astype(int))) >= 2:
        stacker = LogisticRegression(C=1.0, max_iter=4000, random_state=random_state)
        stacker.fit(oof_matrix[valid_mask], train_df.loc[valid_mask, "win_target"].astype(int))
        stacked_holdout = clip_probabilities(stacker.predict_proba(holdout_matrix)[:, 1])
        rows.append(
            {
                "ensemble_name": f"logreg_stack_top{len(finalist_names)}",
                "members": ",".join(finalist_names),
                **evaluate_probabilities(y_holdout, stacked_holdout, threshold=0.5),
            }
        )

    frame = pd.DataFrame(rows)
    write_csv(frame, confirmatory_dir / "ensemble_results.csv")
    return frame


def run_holdout_error_analysis(
    *,
    confirmatory_dir: Path,
    raw_leaderboard: pd.DataFrame,
    holdout_df: pd.DataFrame,
    probability_registry: dict[str, np.ndarray],
    top_n: int,
) -> dict[str, pd.DataFrame]:
    selected_names = [
        str(row["candidate_name"])
        for row in raw_leaderboard.head(top_n).to_dict(orient="records")
        if str(row["candidate_name"]) in probability_registry
    ]
    y_holdout = holdout_df["win_target"].astype(int).to_numpy()
    context_rows: list[dict[str, object]] = []
    decile_rows: list[dict[str, object]] = []
    hardest_frames: list[pd.DataFrame] = []

    for candidate_name in selected_names:
        probabilities = clip_probabilities(probability_registry[candidate_name])
        predictions = (probabilities >= 0.5).astype(int)
        errors = predictions != y_holdout
        abs_error = np.abs(probabilities - y_holdout)

        candidate_frame = holdout_df.copy()
        candidate_frame["candidate_name"] = candidate_name
        candidate_frame["predicted_probability"] = probabilities
        candidate_frame["predicted_class"] = predictions
        candidate_frame["error_flag"] = errors.astype(int)
        candidate_frame["absolute_error"] = abs_error

        for context_column in [column for column in CONTEXT_COLUMNS if column in candidate_frame.columns]:
            for context_value, group in candidate_frame.groupby(context_column, dropna=False):
                if group.empty:
                    continue
                y_group = group["win_target"].astype(int).to_numpy()
                prob_group = group["predicted_probability"].astype(float).to_numpy()
                row = {
                    "candidate_name": candidate_name,
                    "context_column": context_column,
                    "context_value": int(context_value) if pd.notna(context_value) else np.nan,
                    "n_rows": int(len(group)),
                    "positive_rate": float(y_group.mean()),
                    "accuracy": float((group["predicted_class"].astype(int).to_numpy() == y_group).mean()),
                    "mean_probability": float(prob_group.mean()),
                    "mean_absolute_error": float(group["absolute_error"].mean()),
                }
                if len(np.unique(y_group)) >= 2:
                    row["roc_auc"] = float(roc_auc_score(y_group, prob_group))
                    row["log_loss"] = float(log_loss(y_group, prob_group, labels=[0, 1]))
                    row["brier"] = float(brier_score_loss(y_group, prob_group))
                else:
                    row["roc_auc"] = np.nan
                    row["log_loss"] = np.nan
                    row["brier"] = np.nan
                context_rows.append(row)

        decile_codes = pd.qcut(
            pd.Series(probabilities).rank(method="first"),
            q=min(10, len(probabilities)),
            labels=False,
            duplicates="drop",
        )
        candidate_frame["probability_decile"] = decile_codes.astype(int) + 1
        for decile_value, group in candidate_frame.groupby("probability_decile", dropna=False):
            if group.empty:
                continue
            y_group = group["win_target"].astype(int).to_numpy()
            prob_group = group["predicted_probability"].astype(float).to_numpy()
            decile_rows.append(
                {
                    "candidate_name": candidate_name,
                    "probability_decile": int(decile_value),
                    "n_rows": int(len(group)),
                    "probability_min": float(group["predicted_probability"].min()),
                    "probability_max": float(group["predicted_probability"].max()),
                    "probability_mean": float(prob_group.mean()),
                    "observed_positive_rate": float(y_group.mean()),
                    "accuracy": float((group["predicted_class"].astype(int).to_numpy() == y_group).mean()),
                    "error_rate": float(group["error_flag"].mean()),
                    "mean_absolute_error": float(group["absolute_error"].mean()),
                }
            )

        hardest_cases = (
            candidate_frame.loc[candidate_frame["error_flag"] == 1]
            .sort_values(["absolute_error", "predicted_probability"], ascending=[False, False], kind="stable")
            .head(25)
            .copy()
        )
        hardest_frames.append(hardest_cases)

    consensus_frame = pd.DataFrame()
    if selected_names:
        consensus_frame = holdout_df[
            [column for column in ["actual_match_id", "match_date", "season_label", "team_name", "opponent_name", "win_target"] if column in holdout_df.columns]
        ].copy()
        for candidate_name in selected_names:
            prob = clip_probabilities(probability_registry[candidate_name])
            pred = (prob >= 0.5).astype(int)
            error = (pred != y_holdout).astype(int)
            safe_name = candidate_name.replace(".", "_")
            consensus_frame[f"{safe_name}__prob"] = prob
            consensus_frame[f"{safe_name}__error"] = error
        error_columns = [column for column in consensus_frame.columns if column.endswith("__error")]
        prob_columns = [column for column in consensus_frame.columns if column.endswith("__prob")]
        if error_columns:
            consensus_frame["error_count_across_top_models"] = consensus_frame[error_columns].sum(axis=1)
        if prob_columns:
            consensus_frame["mean_probability_across_top_models"] = consensus_frame[prob_columns].mean(axis=1)
        consensus_frame = consensus_frame.sort_values(
            ["error_count_across_top_models", "mean_probability_across_top_models"],
            ascending=[False, False],
            kind="stable",
        ).reset_index(drop=True)

    context_df = pd.DataFrame(context_rows)
    decile_df = pd.DataFrame(decile_rows)
    hardest_df = pd.concat(hardest_frames, ignore_index=True) if hardest_frames else pd.DataFrame()
    write_csv(context_df, confirmatory_dir / "error_by_context.csv")
    write_csv(decile_df, confirmatory_dir / "error_by_probability_decile.csv")
    write_csv(hardest_df, confirmatory_dir / "hardest_holdout_cases.csv")
    write_csv(consensus_frame, confirmatory_dir / "error_consensus.csv")
    return {
        "context_df": context_df,
        "decile_df": decile_df,
        "hardest_df": hardest_df,
        "consensus_df": consensus_frame,
    }


def build_frozen_recreation_report(
    *,
    confirmatory_dir: Path,
    confirmatory_df: pd.DataFrame,
    frozen_model_path: Path,
) -> pd.DataFrame:
    if not frozen_model_path.exists():
        frame = pd.DataFrame()
        write_csv(frame, confirmatory_dir / "frozen_recreation_report.csv")
        return frame

    payload = json.loads(frozen_model_path.read_text(encoding="utf-8"))
    frozen_metrics = payload.get("metrics", {})
    target_names = [
        "official11_frozen_recreation__logreg_l2_c3.0_cwbalanced_frozen",
        "official11_baseline__logreg_l2_c3.0_cwbalanced",
    ]
    current_rows = confirmatory_df.loc[
        (confirmatory_df["variant"] == "raw_default_threshold")
        & (confirmatory_df["seed"] == "aggregate")
        & (confirmatory_df["candidate_name"].astype(str).isin(target_names))
    ].copy()
    rows: list[dict[str, object]] = []
    metric_map = {
        "roc_auc": "holdout_roc_auc",
        "accuracy": "holdout_accuracy",
        "log_loss": "holdout_log_loss",
        "brier": "holdout_brier",
    }
    for _, row in current_rows.iterrows():
        candidate_name = str(row["candidate_name"])
        for current_metric, frozen_metric in metric_map.items():
            current_value = float(row[current_metric])
            frozen_value = float(frozen_metrics.get(frozen_metric, np.nan))
            rows.append(
                {
                    "candidate_name": candidate_name,
                    "metric_name": current_metric,
                    "current_value": current_value,
                    "frozen_reference_value": frozen_value,
                    "delta_current_minus_frozen": current_value - frozen_value if pd.notna(frozen_value) else np.nan,
                    "frozen_on": payload.get("frozen_on", ""),
                    "frozen_experiment": payload.get("experiment", ""),
                    "frozen_model_family": payload.get("model_family", ""),
                    "frozen_penalty": payload.get("penalty", ""),
                    "frozen_C": payload.get("C", np.nan),
                    "frozen_class_weight": payload.get("class_weight", ""),
                }
            )
    frame = pd.DataFrame(rows)
    write_csv(frame, confirmatory_dir / "frozen_recreation_report.csv")
    dump_json(
        {
            "frozen_model_path": str(frozen_model_path),
            "matched_candidate_names": target_names,
            "rows_written": int(len(frame)),
        },
        confirmatory_dir / "frozen_recreation_report.json",
    )
    return frame


def estimate_experiment_volume(
    *,
    train_df: pd.DataFrame,
    temporal_folds: list[TemporalFold],
    feature_columns: list[str],
    feature_specs: list[FeatureSetSpec],
    model_specs: list[ModelSpec],
    shortlist_size: int,
    stochastic_seeds: Sequence[int],
    screening_seeds: Sequence[int],
    stage3_pool_top_global: int,
    stage3_pool_top_per_feature_family: int,
    stage3_pool_top_per_model_family: int,
    stage3_repeated_cv_repeats: int,
    repeated_cv_splits: int,
    repeated_cv_repeats: int,
    bootstrap_runs: int,
    permutation_repeats: int,
    ablation_top_n: int,
    ensemble_top_n: int,
    stage1_top_global: int,
    stage1_top_per_feature_family: int,
    stage1_top_per_model_family: int,
    preset: str,
) -> dict[str, object]:
    feature_ranking_fits = 0
    l1_grid_size = 8
    inner_plan_rows: list[dict[str, object]] = []
    for fold in temporal_folds:
        fold_train, _ = fold_to_frames(train_df, fold)
        split_specs = build_inner_split_specs(
            fold_train,
            random_state=DEFAULT_RANDOM_STATE,
            fallback_splits=3 if preset == "smoke" else 5,
            fallback_repeats=1 if preset == "smoke" else 2,
        )
        inner_split_count = len(split_specs)
        feature_ranking_fits += len(feature_columns) * inner_split_count
        feature_ranking_fits += l1_grid_size * inner_split_count
        feature_ranking_fits += 2
        inner_plan_rows.append(
            {
                "outer_fold": fold.fold_name,
                "inner_split_count": inner_split_count,
                "inner_split_types": ",".join(sorted({str(item["split_type"]) for item in split_specs})),
                "fold_train_rows": int(len(fold_train)),
            }
        )

    candidate_count = len(feature_specs) * len(model_specs)
    full_seed_counts = [len(resolve_seed_list(spec.family, stochastic_seeds, DEFAULT_RANDOM_STATE)) for spec in model_specs]
    screening_seed_counts = [len(resolve_seed_list(spec.family, screening_seeds, DEFAULT_RANDOM_STATE)) for spec in model_specs]
    mean_seed_count = float(np.mean(full_seed_counts)) if full_seed_counts else 1.0
    stage1_exploratory_fits = len(feature_specs) * sum(screening_seed_counts) * len(temporal_folds)
    feature_family_count = len({spec.family for spec in feature_specs})
    model_family_count = len({spec.family for spec in model_specs})
    estimated_stage2_candidate_count = min(
        candidate_count,
        stage1_top_global
        + feature_family_count * stage1_top_per_feature_family
        + model_family_count * stage1_top_per_model_family
        + len(FORCED_CANDIDATE_NAMES),
    )
    estimated_stage3_candidate_count = min(
        estimated_stage2_candidate_count,
        stage3_pool_top_global
        + feature_family_count * stage3_pool_top_per_feature_family
        + model_family_count * stage3_pool_top_per_model_family
        + len(FORCED_CANDIDATE_NAMES),
    )
    stage2_exploratory_fits = int(round(estimated_stage2_candidate_count * mean_seed_count * len(temporal_folds)))
    exploratory_fits = int(stage1_exploratory_fits + stage2_exploratory_fits)
    estimated_stage3_stability_fits = int(
        round(estimated_stage3_candidate_count * repeated_cv_splits * stage3_repeated_cv_repeats * mean_seed_count)
    )
    repeated_cv_fits = int(round(shortlist_size * repeated_cv_splits * repeated_cv_repeats * mean_seed_count))
    threshold_oof_fits = int(round(shortlist_size * len(temporal_folds) * mean_seed_count))
    confirmatory_base_fits = int(round(shortlist_size * mean_seed_count))
    calibration_fits = confirmatory_base_fits
    estimated_ablation_fits = int(ablation_top_n * 12 * max(1, len(stochastic_seeds)))
    estimated_permutation_runs = int(shortlist_size * max(1, len(stochastic_seeds)) * permutation_repeats)
    estimated_ensemble_fits = int(max(0, ensemble_top_n - 1))
    estimated_total_model_fits = (
        feature_ranking_fits
        + exploratory_fits
        + estimated_stage3_stability_fits
        + repeated_cv_fits
        + threshold_oof_fits
        + confirmatory_base_fits
        + calibration_fits
        + estimated_ablation_fits
    )

    return {
        "feature_count": int(len(feature_columns)),
        "feature_set_count": int(len(feature_specs)),
        "model_count": int(len(model_specs)),
        "candidate_count": int(candidate_count),
        "estimated_stage2_candidate_count": int(estimated_stage2_candidate_count),
        "estimated_stage3_candidate_count": int(estimated_stage3_candidate_count),
        "temporal_outer_fold_count": int(len(temporal_folds)),
        "estimated_feature_ranking_fits": int(feature_ranking_fits),
        "estimated_stage1_exploratory_fits": int(stage1_exploratory_fits),
        "estimated_stage2_exploratory_fits": int(stage2_exploratory_fits),
        "estimated_exploratory_model_fits": int(exploratory_fits),
        "estimated_stage3_stability_fits": int(estimated_stage3_stability_fits),
        "estimated_repeated_cv_fits": int(repeated_cv_fits),
        "estimated_threshold_oof_fits": int(threshold_oof_fits),
        "estimated_confirmatory_base_fits": int(confirmatory_base_fits),
        "estimated_calibration_fits": int(calibration_fits),
        "estimated_ablation_fits": int(estimated_ablation_fits),
        "estimated_permutation_repeats_total": int(estimated_permutation_runs),
        "estimated_ensemble_fits": int(estimated_ensemble_fits),
        "estimated_total_model_fits": int(estimated_total_model_fits),
        "bootstrap_runs_per_finalist": int(bootstrap_runs),
        "stochastic_seed_count": int(len(stochastic_seeds)),
        "screening_seed_count": int(len(screening_seeds)),
        "inner_plan": inner_plan_rows,
    }


def evaluate_finalist_confirmatory(
    *,
    candidate_row: dict[str, object],
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    features: list[str],
    model_lookup: dict[str, ModelSpec],
    random_state: int,
    stochastic_seeds: Sequence[int],
    bootstrap_runs: int,
    calibration_fraction: float,
    threshold_objectives: Sequence[str],
    permutation_repeats: int,
) -> dict[str, object]:
    candidate_name = str(candidate_row["candidate_name"])
    model_name = str(candidate_row["model_name"])
    model_spec = model_lookup[model_name]
    model_params = params_to_dict(model_spec.params)
    seed_list = resolve_seed_list(model_spec.family, stochastic_seeds, random_state)
    y_holdout = holdout_df["win_target"].astype(int).to_numpy()

    raw_prob_by_seed: list[np.ndarray] = []
    calibrated_prob_by_seed: list[np.ndarray] = []
    seed_result_rows: list[dict[str, object]] = []
    oof_fold_rows: list[dict[str, object]] = []
    oof_prob_by_seed: list[np.ndarray] = []
    permutation_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    calibration_bin_rows: list[dict[str, object]] = []

    for seed in seed_list:
        raw_model = build_pipeline(model_spec.family, model_params, int(seed))
        raw_model.fit(train_df[features], train_df["win_target"].astype(int))
        seed_raw_prob = clip_probabilities(predict_positive_probability(raw_model, holdout_df, features))
        raw_prob_by_seed.append(seed_raw_prob)
        raw_metrics = evaluate_probabilities(y_holdout, seed_raw_prob, threshold=0.5)
        seed_result_rows.append(
            {
                "candidate_name": candidate_name,
                "variant": "raw_default_threshold_seed",
                "seed": int(seed),
                "feature_count": int(len(features)),
                "features": ",".join(features),
                "threshold": 0.5,
                **raw_metrics,
            }
        )

        oof_prob_seed, oof_fold_df = temporal_oof_predictions(
            train_df=train_df,
            features=features,
            model_spec=model_spec,
            random_state=int(seed),
        )
        oof_prob_by_seed.append(oof_prob_seed)
        for row in oof_fold_df.to_dict(orient="records"):
            oof_fold_rows.append({"candidate_name": candidate_name, "seed": int(seed), "feature_count": int(len(features)), **row})

        try:
            perm = permutation_importance(
                raw_model,
                holdout_df[features],
                holdout_df["win_target"].astype(int),
                n_repeats=permutation_repeats,
                random_state=int(seed),
                scoring="roc_auc",
            )
            for feature, mean_value, std_value in zip(features, perm.importances_mean, perm.importances_std):
                permutation_rows.append(
                    {
                        "candidate_name": candidate_name,
                        "seed": int(seed),
                        "feature": feature,
                        "importance_mean": float(mean_value),
                        "importance_std": float(std_value),
                    }
                )
        except Exception as exc:
            permutation_rows.append(
                {
                    "candidate_name": candidate_name,
                    "seed": int(seed),
                    "feature": "__permutation_failed__",
                    "importance_mean": np.nan,
                    "importance_std": np.nan,
                    "error": str(exc),
                }
            )

        try:
            calib_train, calib_df = time_ordered_calibration_split(train_df, fraction=calibration_fraction)
            if len(calib_df) >= 40:
                calib_model = build_pipeline(model_spec.family, model_params, int(seed))
                calib_model.fit(calib_train[features], calib_train["win_target"].astype(int))
                calib_prob = clip_probabilities(predict_positive_probability(calib_model, calib_df, features))
                calib_y = calib_df["win_target"].astype(int).to_numpy()
                if len(np.unique(calib_y)) >= 2:
                    calibrator = IsotonicRegression(out_of_bounds="clip")
                    calibrator.fit(calib_prob, calib_y)
                    raw_holdout_from_calib_model = clip_probabilities(predict_positive_probability(calib_model, holdout_df, features))
                    seed_calibrated = clip_probabilities(calibrator.transform(raw_holdout_from_calib_model))
                    calibrated_prob_by_seed.append(seed_calibrated)
                    calib_ece, calib_bins = compute_ece(y_holdout, seed_calibrated, n_bins=10)
                    calibration_rows.append(
                        {
                            "candidate_name": candidate_name,
                            "variant": "calibrated_isotonic_seed",
                            "seed": int(seed),
                            "calibration_method": "isotonic",
                            "calibration_train_rows": int(len(calib_train)),
                            "calibration_rows": int(len(calib_df)),
                            "holdout_ece": float(calib_ece),
                        }
                    )
                    for row in calib_bins.to_dict(orient="records"):
                        calibration_bin_rows.append({"candidate_name": candidate_name, "variant": "calibrated_isotonic_seed", "seed": int(seed), **row})
        except Exception as exc:
            calibration_rows.append(
                {
                    "candidate_name": candidate_name,
                    "variant": "calibration_skipped_seed",
                    "seed": int(seed),
                    "calibration_method": "isotonic",
                    "calibration_train_rows": 0,
                    "calibration_rows": 0,
                    "holdout_ece": np.nan,
                    "skip_reason": str(exc),
                }
            )

    raw_prob = clip_probabilities(np.mean(np.vstack(raw_prob_by_seed), axis=0))
    oof_matrix = np.vstack(oof_prob_by_seed)
    valid_counts = np.sum(~np.isnan(oof_matrix), axis=0)
    oof_sum = np.nansum(oof_matrix, axis=0)
    mean_oof_prob = np.divide(oof_sum, valid_counts, out=np.full_like(oof_sum, np.nan, dtype=float), where=valid_counts > 0)

    result_rows: list[dict[str, object]] = []
    raw_metrics = evaluate_probabilities(y_holdout, raw_prob, threshold=0.5)
    result_rows.append(
        {
            "candidate_name": candidate_name,
            "variant": "raw_default_threshold",
            "seed": "aggregate",
            "feature_count": int(len(features)),
            "features": ",".join(features),
            "threshold": 0.5,
            **raw_metrics,
        }
    )

    valid_mask = ~np.isnan(mean_oof_prob)
    threshold_grid = pd.DataFrame()
    threshold_best_rows: list[dict[str, float]] = []
    primary_threshold = 0.5
    if valid_mask.any():
        threshold_grid, threshold_best_rows = select_best_thresholds(
            train_df.iloc[valid_mask]["win_target"].astype(int).to_numpy(),
            mean_oof_prob[valid_mask],
            threshold_objectives,
        )
        for best_row in threshold_best_rows:
            objective = str(best_row["objective"])
            threshold = float(best_row["threshold"])
            if objective == str(threshold_objectives[0]):
                primary_threshold = threshold
            tuned_metrics = evaluate_probabilities(y_holdout, raw_prob, threshold=threshold)
            result_rows.append(
                {
                    "candidate_name": candidate_name,
                    "variant": f"raw_threshold_{objective}",
                    "seed": "aggregate",
                    "feature_count": int(len(features)),
                    "features": ",".join(features),
                    "threshold": threshold,
                    **tuned_metrics,
                }
            )

    calibrated_probabilities: np.ndarray | None = None
    if calibrated_prob_by_seed:
        calibrated_probabilities = clip_probabilities(np.mean(np.vstack(calibrated_prob_by_seed), axis=0))
        calibrated_metrics = evaluate_probabilities(y_holdout, calibrated_probabilities, threshold=0.5)
        result_rows.append(
            {
                "candidate_name": candidate_name,
                "variant": "calibrated_isotonic",
                "seed": "aggregate",
                "feature_count": int(len(features)),
                "features": ",".join(features),
                "threshold": 0.5,
                **calibrated_metrics,
            }
        )

    bootstrap_detail, bootstrap_summary = bootstrap_metrics(
        y_holdout,
        raw_prob,
        threshold=float(primary_threshold),
        runs=bootstrap_runs,
        random_state=random_state,
    )
    bootstrap_summary_row = {
        "candidate_name": candidate_name,
        "feature_count": int(len(features)),
        "bootstrap_threshold": float(primary_threshold),
        "stochastic_seed_count": int(len(seed_list)),
        **bootstrap_summary,
    }

    threshold_rows: list[dict[str, object]] = []
    if not threshold_grid.empty:
        threshold_rows = [
            {
                "candidate_name": candidate_name,
                "feature_count": int(len(features)),
                "seed": "aggregate",
                **row,
            }
            for row in threshold_grid.to_dict(orient="records")
        ]

    permutation_summary_rows: list[dict[str, object]] = []
    permutation_df = pd.DataFrame(permutation_rows)
    if not permutation_df.empty:
        permutation_summary = (
            permutation_df.loc[permutation_df["feature"] != "__permutation_failed__"]
            .groupby(["candidate_name", "feature"], as_index=False)[["importance_mean", "importance_std"]]
            .agg({"importance_mean": ["mean", "std"], "importance_std": ["mean", "max"]})
        )
        if not permutation_summary.empty:
            permutation_summary.columns = [
                "candidate_name",
                "feature",
                "importance_mean_mean",
                "importance_mean_std",
                "importance_std_mean",
                "importance_std_max",
            ]
            permutation_summary_rows = permutation_summary.sort_values(
                ["importance_mean_mean", "importance_mean_std"],
                ascending=[False, True],
                kind="stable",
            ).to_dict(orient="records")

    return {
        "result_rows": result_rows + seed_result_rows,
        "threshold_rows": threshold_rows,
        "threshold_best_rows": [{"candidate_name": candidate_name, **row} for row in threshold_best_rows],
        "oof_fold_rows": oof_fold_rows,
        "bootstrap_detail": bootstrap_detail.assign(candidate_name=candidate_name) if not bootstrap_detail.empty else bootstrap_detail,
        "bootstrap_summary_row": bootstrap_summary_row,
        "calibration_rows": calibration_rows,
        "calibration_bin_rows": calibration_bin_rows,
        "permutation_rows": permutation_rows,
        "permutation_summary_rows": permutation_summary_rows,
        "raw_holdout_prob": raw_prob,
        "raw_holdout_prob_by_seed": raw_prob_by_seed,
        "oof_train_prob": mean_oof_prob,
        "calibrated_holdout_prob": calibrated_probabilities,
        "primary_threshold": float(primary_threshold),
        "stochastic_seed_count": int(len(seed_list)),
    }


def run_confirmatory_phase(
    *,
    output_dir: Path,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    shortlist: pd.DataFrame,
    feature_columns: list[str],
    official_features: list[str],
    feature_specs: list[FeatureSetSpec],
    model_lookup: dict[str, ModelSpec],
    random_state: int,
    preset: str,
    stochastic_seeds: Sequence[int],
    repeated_cv_splits: int,
    repeated_cv_repeats: int,
    bootstrap_runs: int,
    pairwise_top_n: int,
    calibration_fraction: float,
    threshold_objectives: Sequence[str],
    permutation_repeats: int,
    ablation_top_n: int,
    ensemble_top_n: int,
    frozen_model_path: Path,
    error_analysis_top_n: int,
    log_path: Path,
) -> dict[str, object]:
    confirmatory_dir = output_dir / "confirmatory"
    ensure_dir(confirmatory_dir)

    full_train_feature_space = freeze_feature_space(
        train_df,
        feature_columns=feature_columns,
        official_features=official_features,
        feature_specs=feature_specs,
        random_state=random_state,
        preset=preset,
        output_dir=confirmatory_dir / "full_train_feature_space",
    )

    finalist_feature_rows: list[dict[str, object]] = []
    finalist_feature_map: dict[str, list[str]] = {}
    for row in shortlist.to_dict(orient="records"):
        candidate_name = str(row["candidate_name"])
        feature_set_name = str(row["feature_set_name"])
        features = list(full_train_feature_space["feature_sets"][feature_set_name])
        finalist_feature_map[candidate_name] = features
        finalist_feature_rows.append(
            {
                "candidate_name": candidate_name,
                "feature_set_name": feature_set_name,
                "feature_count": int(len(features)),
                "features": ",".join(features),
            }
        )
    write_csv(pd.DataFrame(finalist_feature_rows), confirmatory_dir / "finalist_feature_manifest.csv")

    repeated_fold_df, repeated_summary_df = repeated_cv_finalists(
        train_df=train_df,
        shortlist=shortlist,
        finalist_feature_map=finalist_feature_map,
        model_lookup=model_lookup,
        random_state=random_state,
        stochastic_seeds=stochastic_seeds,
        n_splits=repeated_cv_splits,
        n_repeats=repeated_cv_repeats,
    )
    write_csv(repeated_fold_df, confirmatory_dir / "repeated_cv_fold_metrics.csv")
    write_csv(repeated_summary_df, confirmatory_dir / "repeated_cv_summary.csv")

    confirmatory_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    threshold_best_rows: list[dict[str, object]] = []
    oof_fold_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    calibration_bin_rows: list[dict[str, object]] = []
    bootstrap_summary_rows: list[dict[str, object]] = []
    bootstrap_detail_frames: list[pd.DataFrame] = []
    probability_registry: dict[str, np.ndarray] = {}
    oof_registry: dict[str, np.ndarray] = {}
    permutation_rows: list[dict[str, object]] = []
    permutation_summary_rows: list[dict[str, object]] = []

    for index, shortlist_row in enumerate(shortlist.to_dict(orient="records"), start=1):
        candidate_name = str(shortlist_row["candidate_name"])
        log(f"[confirmatory] avaliando finalista {index}/{len(shortlist)}: {candidate_name}", log_path)
        result = evaluate_finalist_confirmatory(
            candidate_row=shortlist_row,
            train_df=train_df,
            holdout_df=holdout_df,
            features=finalist_feature_map[candidate_name],
            model_lookup=model_lookup,
            random_state=random_state + index,
            stochastic_seeds=stochastic_seeds,
            bootstrap_runs=bootstrap_runs,
            calibration_fraction=calibration_fraction,
            threshold_objectives=threshold_objectives,
            permutation_repeats=permutation_repeats,
        )
        confirmatory_rows.extend(result["result_rows"])
        threshold_rows.extend(result["threshold_rows"])
        threshold_best_rows.extend(result["threshold_best_rows"])
        oof_fold_rows.extend(result["oof_fold_rows"])
        calibration_rows.extend(result["calibration_rows"])
        calibration_bin_rows.extend(result["calibration_bin_rows"])
        bootstrap_summary_rows.append(result["bootstrap_summary_row"])
        permutation_rows.extend(result["permutation_rows"])
        permutation_summary_rows.extend(result["permutation_summary_rows"])
        if isinstance(result["bootstrap_detail"], pd.DataFrame) and not result["bootstrap_detail"].empty:
            bootstrap_detail_frames.append(result["bootstrap_detail"])
        probability_registry[candidate_name] = result["raw_holdout_prob"]
        oof_registry[candidate_name] = result["oof_train_prob"]

    confirmatory_df = pd.DataFrame(confirmatory_rows)
    write_csv(confirmatory_df, confirmatory_dir / "confirmatory_results.csv")
    write_csv(pd.DataFrame(threshold_rows), confirmatory_dir / "threshold_grid.csv")
    write_csv(pd.DataFrame(threshold_best_rows), confirmatory_dir / "threshold_best.csv")
    write_csv(pd.DataFrame(oof_fold_rows), confirmatory_dir / "temporal_oof_metrics.csv")
    write_csv(pd.DataFrame(calibration_rows), confirmatory_dir / "calibration_summary.csv")
    write_csv(pd.DataFrame(calibration_bin_rows), confirmatory_dir / "calibration_bins.csv")
    write_csv(pd.DataFrame(bootstrap_summary_rows), confirmatory_dir / "bootstrap_summary.csv")
    write_csv(pd.concat(bootstrap_detail_frames, ignore_index=True) if bootstrap_detail_frames else pd.DataFrame(), confirmatory_dir / "bootstrap_detail.csv")
    write_csv(pd.DataFrame(permutation_rows), confirmatory_dir / "permutation_importance_seed_detail.csv")
    write_csv(pd.DataFrame(permutation_summary_rows), confirmatory_dir / "permutation_importance_summary.csv")

    raw_leaderboard = sort_leaderboard(
        confirmatory_df.loc[(confirmatory_df["variant"] == "raw_default_threshold") & (confirmatory_df["seed"] == "aggregate")].rename(
            columns={
                "roc_auc": "temporal_cv_roc_auc_mean",
                "log_loss": "temporal_cv_log_loss_mean",
                "brier": "temporal_cv_brier_mean",
                "accuracy": "temporal_cv_accuracy_mean",
            }
        )
    ).rename(
        columns={
            "temporal_cv_roc_auc_mean": "roc_auc",
            "temporal_cv_log_loss_mean": "log_loss",
            "temporal_cv_brier_mean": "brier",
            "temporal_cv_accuracy_mean": "accuracy",
        }
    )
    write_csv(raw_leaderboard, confirmatory_dir / "holdout_raw_leaderboard.csv")

    tuned_primary_variant = f"raw_threshold_{threshold_objectives[0]}"
    tuned_leaderboard = sort_leaderboard(
        confirmatory_df.loc[(confirmatory_df["variant"] == tuned_primary_variant) & (confirmatory_df["seed"] == "aggregate")].rename(
            columns={
                "roc_auc": "temporal_cv_roc_auc_mean",
                "log_loss": "temporal_cv_log_loss_mean",
                "brier": "temporal_cv_brier_mean",
                "accuracy": "temporal_cv_accuracy_mean",
            }
        )
    ).rename(
        columns={
            "temporal_cv_roc_auc_mean": "roc_auc",
            "temporal_cv_log_loss_mean": "log_loss",
            "temporal_cv_brier_mean": "brier",
            "temporal_cv_accuracy_mean": "accuracy",
        }
    )
    write_csv(tuned_leaderboard, confirmatory_dir / "holdout_tuned_threshold_leaderboard.csv")

    pairwise_summary_rows: list[dict[str, object]] = []
    pairwise_detail_frames: list[pd.DataFrame] = []
    pairwise_oof_summary_rows: list[dict[str, object]] = []
    pairwise_oof_detail_frames: list[pd.DataFrame] = []
    shortlist_pairs = shortlist.head(pairwise_top_n).to_dict(orient="records")
    y_holdout = holdout_df["win_target"].astype(int).to_numpy()
    for left_row, right_row in combinations(shortlist_pairs, 2):
        left_name = str(left_row["candidate_name"])
        right_name = str(right_row["candidate_name"])
        detail, summary = pairwise_bootstrap_differences(
            y_true=y_holdout,
            left_name=left_name,
            left_prob=probability_registry[left_name],
            right_name=right_name,
            right_prob=probability_registry[right_name],
            runs=bootstrap_runs,
            random_state=random_state + len(pairwise_summary_rows) + 1000,
        )
        if not detail.empty:
            pairwise_detail_frames.append(detail.assign(left_candidate=left_name, right_candidate=right_name))
        if summary:
            pairwise_summary_rows.append(summary)
        left_oof = oof_registry[left_name]
        right_oof = oof_registry[right_name]
        valid_mask = ~np.isnan(left_oof) & ~np.isnan(right_oof)
        if valid_mask.any() and len(np.unique(train_df.loc[valid_mask, "win_target"].astype(int))) >= 2:
            oof_detail, oof_summary = pairwise_bootstrap_differences(
                y_true=train_df.loc[valid_mask, "win_target"].astype(int).to_numpy(),
                left_name=left_name,
                left_prob=left_oof[valid_mask],
                right_name=right_name,
                right_prob=right_oof[valid_mask],
                runs=bootstrap_runs,
                random_state=random_state + len(pairwise_oof_summary_rows) + 5000,
            )
            if not oof_detail.empty:
                pairwise_oof_detail_frames.append(oof_detail.assign(left_candidate=left_name, right_candidate=right_name))
            if oof_summary:
                pairwise_oof_summary_rows.append(oof_summary)
    write_csv(pd.DataFrame(pairwise_summary_rows), confirmatory_dir / "pairwise_bootstrap_summary.csv")
    write_csv(pd.concat(pairwise_detail_frames, ignore_index=True) if pairwise_detail_frames else pd.DataFrame(), confirmatory_dir / "pairwise_bootstrap_detail.csv")
    write_csv(pd.DataFrame(pairwise_oof_summary_rows), confirmatory_dir / "pairwise_oof_bootstrap_summary.csv")
    write_csv(pd.concat(pairwise_oof_detail_frames, ignore_index=True) if pairwise_oof_detail_frames else pd.DataFrame(), confirmatory_dir / "pairwise_oof_bootstrap_detail.csv")

    ablation_df = run_ablation_study(
        output_dir=confirmatory_dir,
        shortlist=shortlist,
        finalist_feature_map=finalist_feature_map,
        model_lookup=model_lookup,
        train_df=train_df,
        holdout_df=holdout_df,
        stochastic_seeds=stochastic_seeds,
        ablation_top_n=ablation_top_n,
        random_state=random_state + 7000,
    )
    ensemble_df = build_finalist_ensembles(
        confirmatory_dir=confirmatory_dir,
        shortlist=shortlist,
        probability_registry=probability_registry,
        oof_registry=oof_registry,
        train_df=train_df,
        holdout_df=holdout_df,
        ensemble_top_n=ensemble_top_n,
        random_state=random_state + 8000,
    )
    error_analysis = run_holdout_error_analysis(
        confirmatory_dir=confirmatory_dir,
        raw_leaderboard=raw_leaderboard,
        holdout_df=holdout_df,
        probability_registry=probability_registry,
        top_n=error_analysis_top_n,
    )
    frozen_recreation_df = build_frozen_recreation_report(
        confirmatory_dir=confirmatory_dir,
        confirmatory_df=confirmatory_df,
        frozen_model_path=frozen_model_path,
    )

    return {
        "confirmatory_df": confirmatory_df,
        "raw_leaderboard": raw_leaderboard,
        "tuned_leaderboard": tuned_leaderboard,
        "repeated_summary_df": repeated_summary_df,
        "ablation_df": ablation_df,
        "ensemble_df": ensemble_df,
        "error_analysis": error_analysis,
        "frozen_recreation_df": frozen_recreation_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bateria experimental de nivel de dissertacao para CS2.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--official-config-path", type=Path, default=DEFAULT_OFFICIAL_CONFIG_PATH)
    parser.add_argument("--frozen-model-path", type=Path, default=DEFAULT_FROZEN_MODEL_PATH)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--preset", choices=["smoke", "dissertation"], default="dissertation")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--shortlist-size", type=int, default=12)
    parser.add_argument("--stochastic-seeds", type=str, default="42,52,62,72,82,92,102,112")
    parser.add_argument("--screening-seeds", type=str, default="42,52")
    parser.add_argument("--stage1-top-global", type=int, default=DEFAULT_STAGE1_TOP_GLOBAL)
    parser.add_argument("--stage1-top-per-feature-family", type=int, default=DEFAULT_STAGE1_TOP_PER_FEATURE_FAMILY)
    parser.add_argument("--stage1-top-per-model-family", type=int, default=DEFAULT_STAGE1_TOP_PER_MODEL_FAMILY)
    parser.add_argument("--stage3-pool-top-global", type=int, default=DEFAULT_STAGE3_POOL_TOP_GLOBAL)
    parser.add_argument("--stage3-pool-top-per-feature-family", type=int, default=DEFAULT_STAGE3_POOL_TOP_PER_FEATURE_FAMILY)
    parser.add_argument("--stage3-pool-top-per-model-family", type=int, default=DEFAULT_STAGE3_POOL_TOP_PER_MODEL_FAMILY)
    parser.add_argument("--stage3-repeated-cv-repeats", type=int, default=DEFAULT_STAGE3_REPEATED_CV_REPEATS)
    parser.add_argument("--repeated-cv-splits", type=int, default=5)
    parser.add_argument("--repeated-cv-repeats", type=int, default=7)
    parser.add_argument("--bootstrap-runs", type=int, default=1200)
    parser.add_argument("--threshold-objectives", type=str, default="f1,accuracy,precision,recall")
    parser.add_argument("--permutation-repeats", type=int, default=60)
    parser.add_argument("--ablation-top-n", type=int, default=5)
    parser.add_argument("--ensemble-top-n", type=int, default=5)
    parser.add_argument("--pairwise-top-n", type=int, default=6)
    parser.add_argument("--error-analysis-top-n", type=int, default=DEFAULT_ERROR_ANALYSIS_TOP_N)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = "smoke" if args.smoke_test else args.preset
    shortlist_size = 4 if preset == "smoke" else args.shortlist_size
    repeated_cv_repeats = 2 if preset == "smoke" else args.repeated_cv_repeats
    bootstrap_runs = 60 if preset == "smoke" else args.bootstrap_runs
    batch_size = 8 if preset == "smoke" else args.batch_size
    stochastic_seeds = [int(seed) for seed in parse_csv_ints(args.stochastic_seeds)] or DEFAULT_STOCHASTIC_SEEDS
    screening_seeds = [int(seed) for seed in parse_csv_ints(args.screening_seeds)] or DEFAULT_SCREENING_SEEDS
    if preset == "smoke":
        screening_seeds = screening_seeds[:1] or [DEFAULT_RANDOM_STATE]
    threshold_objectives = parse_csv_strings(args.threshold_objectives) or DEFAULT_THRESHOLD_OBJECTIVES
    permutation_repeats = 6 if preset == "smoke" else args.permutation_repeats
    ablation_top_n = 1 if preset == "smoke" else args.ablation_top_n
    ensemble_top_n = 2 if preset == "smoke" else args.ensemble_top_n
    stage1_top_global = 24 if preset == "smoke" else args.stage1_top_global
    stage1_top_per_feature_family = 2 if preset == "smoke" else args.stage1_top_per_feature_family
    stage1_top_per_model_family = 3 if preset == "smoke" else args.stage1_top_per_model_family
    stage3_pool_top_global = 12 if preset == "smoke" else args.stage3_pool_top_global
    stage3_pool_top_per_feature_family = 2 if preset == "smoke" else args.stage3_pool_top_per_feature_family
    stage3_pool_top_per_model_family = 2 if preset == "smoke" else args.stage3_pool_top_per_model_family
    stage3_repeated_cv_repeats = 2 if preset == "smoke" else args.stage3_repeated_cv_repeats
    error_analysis_top_n = 2 if preset == "smoke" else args.error_analysis_top_n

    output_dir = resolve_output_dir(args.reports_root, args.experiment_name)
    ensure_dir(output_dir)
    log_path = output_dir / "run.log"

    dataset = load_dataset(args.data_path)
    base_feature_columns = resolve_feature_columns(dataset, args.data_path)
    dataset, feature_columns, derived_feature_manifest = derive_v5_feature_columns(dataset, base_feature_columns)
    write_csv(derived_feature_manifest, output_dir / "derived_feature_manifest.csv")
    log(
        f"[feature-factory] base_features={len(base_feature_columns)} derived_features={len(derived_feature_manifest)} total_features={len(feature_columns)}",
        log_path,
    )
    official_features = read_official_feature_list(args.official_config_path)
    train_df, holdout_df = split_train_holdout(dataset)
    temporal_folds = build_temporal_folds(train_df)

    audit = build_pipeline_audit(
        data_path=args.data_path,
        official_config_path=args.official_config_path,
        feature_columns=feature_columns,
        official_features=official_features,
        train_df=train_df,
        holdout_df=holdout_df,
    )
    dump_json(audit, output_dir / "pipeline_audit.json")

    feature_specs = build_feature_set_specs(official_features, feature_columns, preset)
    model_specs = build_model_specs(preset)
    candidates = build_candidate_specs(feature_specs, model_specs)
    if args.max_candidates is not None:
        candidates = candidates[: max(1, int(args.max_candidates))]
    feature_lookup = {spec.name: spec for spec in feature_specs}
    model_lookup = {spec.name: spec for spec in model_specs}

    write_csv(
        pd.DataFrame(
            [
                {"feature_set_name": spec.name, "family": spec.family, "params_json": json.dumps(params_to_dict(spec.params), ensure_ascii=False)}
                for spec in feature_specs
            ]
        ),
        output_dir / "feature_set_spec_manifest.csv",
    )
    write_csv(
        pd.DataFrame(
            [
                {"model_name": spec.name, "family": spec.family, "params_json": json.dumps(params_to_dict(spec.params), ensure_ascii=False)}
                for spec in model_specs
            ]
        ),
        output_dir / "model_spec_manifest.csv",
    )
    write_csv(
        candidate_manifest_frame(
            candidates=candidates,
            feature_lookup=feature_lookup,
            model_lookup=model_lookup,
            stochastic_seeds=stochastic_seeds,
        ),
        output_dir / "candidate_manifest.csv",
    )

    volume = estimate_experiment_volume(
        train_df=train_df,
        temporal_folds=temporal_folds,
        feature_columns=feature_columns,
        feature_specs=feature_specs,
        model_specs=model_specs,
        shortlist_size=shortlist_size,
        stochastic_seeds=stochastic_seeds,
        screening_seeds=screening_seeds,
        stage3_pool_top_global=stage3_pool_top_global,
        stage3_pool_top_per_feature_family=stage3_pool_top_per_feature_family,
        stage3_pool_top_per_model_family=stage3_pool_top_per_model_family,
        stage3_repeated_cv_repeats=stage3_repeated_cv_repeats,
        repeated_cv_splits=args.repeated_cv_splits,
        repeated_cv_repeats=repeated_cv_repeats,
        bootstrap_runs=bootstrap_runs,
        permutation_repeats=permutation_repeats,
        ablation_top_n=ablation_top_n,
        ensemble_top_n=ensemble_top_n,
        stage1_top_global=stage1_top_global,
        stage1_top_per_feature_family=stage1_top_per_feature_family,
        stage1_top_per_model_family=stage1_top_per_model_family,
        preset=preset,
    )
    dump_json(
        {
            "generated_at": datetime.now().isoformat(),
            "preset": preset,
            "workers": int(args.workers),
            "batch_size": int(batch_size),
            "random_state": int(args.random_state),
            "shortlist_size": int(shortlist_size),
            "stochastic_seeds": stochastic_seeds,
            "screening_seeds": screening_seeds,
            "stage1_top_global": int(stage1_top_global),
            "stage1_top_per_feature_family": int(stage1_top_per_feature_family),
            "stage1_top_per_model_family": int(stage1_top_per_model_family),
            "stage3_pool_top_global": int(stage3_pool_top_global),
            "stage3_pool_top_per_feature_family": int(stage3_pool_top_per_feature_family),
            "stage3_pool_top_per_model_family": int(stage3_pool_top_per_model_family),
            "stage3_repeated_cv_repeats": int(stage3_repeated_cv_repeats),
            "repeated_cv_splits": int(args.repeated_cv_splits),
            "repeated_cv_repeats": int(repeated_cv_repeats),
            "bootstrap_runs": int(bootstrap_runs),
            "threshold_objectives": threshold_objectives,
            "permutation_repeats": int(permutation_repeats),
            "ablation_top_n": int(ablation_top_n),
            "ensemble_top_n": int(ensemble_top_n),
            "pairwise_top_n": int(args.pairwise_top_n),
            "error_analysis_top_n": int(error_analysis_top_n),
            "calibration_fraction": float(args.calibration_fraction),
            "frozen_model_path": str(args.frozen_model_path),
            "base_feature_count": int(len(base_feature_columns)),
            "derived_feature_count": int(len(derived_feature_manifest)),
            "optional_model_availability": available_optional_families(),
            "volume": volume,
        },
        output_dir / "experiment_config.json",
    )
    log(
        f"[setup] preset={preset} features={len(feature_columns)} feature_sets={len(feature_specs)} models={len(model_specs)} candidates={len(candidates)} workers={args.workers}",
        log_path,
    )
    log(
        f"[volume] ranking_fits={volume['estimated_feature_ranking_fits']} exploratory_fits={volume['estimated_exploratory_model_fits']} total_fits~={volume['estimated_total_model_fits']}",
        log_path,
    )

    fold_payloads = prepare_exploratory_folds(
        train_df=train_df,
        temporal_folds=temporal_folds,
        feature_columns=feature_columns,
        official_features=official_features,
        feature_specs=feature_specs,
        random_state=args.random_state,
        preset=preset,
        output_dir=output_dir / "feature_rankings",
        log_path=log_path,
    )

    if args.dry_run:
        log("[dry-run] configuracao validada; execucao longa nao iniciada.", log_path)
        return

    exploratory_df, stage2_candidate_names = run_staged_search(
        output_dir=output_dir,
        candidates=candidates,
        feature_lookup=feature_lookup,
        model_lookup=model_lookup,
        fold_payloads=fold_payloads,
        workers=args.workers,
        batch_size=batch_size,
        random_state=args.random_state,
        stochastic_seeds=stochastic_seeds,
        screening_seeds=screening_seeds,
        stage1_top_global=stage1_top_global,
        stage1_top_per_feature_family=stage1_top_per_feature_family,
        stage1_top_per_model_family=stage1_top_per_model_family,
        forced_candidate_names=sorted(FORCED_CANDIDATE_NAMES),
        resume=args.resume,
        log_path=log_path,
    )
    log(f"[stage3] preparando pool de estabilizacao a partir de {len(stage2_candidate_names)} candidatos do stage2", log_path)
    stage3_leaderboard, stage3_candidate_names, stage3_repeated_fold_df, stage3_repeated_summary_df = run_stage3_shortlist_screen(
        output_dir=output_dir,
        stage2_leaderboard=exploratory_df,
        train_df=train_df,
        feature_columns=feature_columns,
        official_features=official_features,
        feature_specs=feature_specs,
        model_lookup=model_lookup,
        stochastic_seeds=stochastic_seeds,
        random_state=args.random_state + 17000,
        preset=preset,
        top_global=stage3_pool_top_global,
        top_per_feature_family=stage3_pool_top_per_feature_family,
        top_per_model_family=stage3_pool_top_per_model_family,
        forced_candidate_names=sorted(FORCED_CANDIDATE_NAMES),
        repeated_cv_splits=args.repeated_cv_splits,
        repeated_cv_repeats=stage3_repeated_cv_repeats,
    )
    log(f"[stage3] candidatos avaliados no stage3={len(stage3_candidate_names)}", log_path)
    shortlist = freeze_shortlist(
        stage3_leaderboard,
        shortlist_size=shortlist_size,
        output_path=output_dir / "shortlist.csv",
        forced_candidate_names=sorted(FORCED_CANDIDATE_NAMES),
    )
    log(f"[shortlist] finalistas congelados={len(shortlist)}", log_path)

    confirmatory = run_confirmatory_phase(
        output_dir=output_dir,
        train_df=train_df,
        holdout_df=holdout_df,
        shortlist=shortlist,
        feature_columns=feature_columns,
        official_features=official_features,
        feature_specs=feature_specs,
        model_lookup=model_lookup,
        random_state=args.random_state,
        preset=preset,
        stochastic_seeds=stochastic_seeds,
        repeated_cv_splits=args.repeated_cv_splits,
        repeated_cv_repeats=repeated_cv_repeats,
        bootstrap_runs=bootstrap_runs,
        pairwise_top_n=min(args.pairwise_top_n, len(shortlist)),
        calibration_fraction=args.calibration_fraction,
        threshold_objectives=threshold_objectives,
        permutation_repeats=permutation_repeats,
        ablation_top_n=min(ablation_top_n, len(shortlist)),
        ensemble_top_n=min(ensemble_top_n, len(shortlist)),
        frozen_model_path=args.frozen_model_path,
        error_analysis_top_n=min(error_analysis_top_n, len(shortlist)),
        log_path=log_path,
    )

    summary_payload = {
        "generated_at": datetime.now().isoformat(),
        "preset": preset,
        "output_dir": str(output_dir),
        "volume": volume,
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "feature_count": int(len(feature_columns)),
        "base_feature_count": int(len(base_feature_columns)),
        "derived_feature_count": int(len(derived_feature_manifest)),
        "shortlist_count": int(len(shortlist)),
        "stochastic_seed_count": int(len(stochastic_seeds)),
        "screening_seed_count": int(len(screening_seeds)),
        "stage2_candidate_count": int(len(stage2_candidate_names)),
        "stage3_candidate_count": int(len(stage3_candidate_names)),
        "exploratory_best_candidate": exploratory_df.iloc[0]["candidate_name"] if not exploratory_df.empty else "",
        "stage3_best_candidate": stage3_leaderboard.iloc[0]["candidate_name"] if not stage3_leaderboard.empty else "",
        "holdout_best_raw_candidate": confirmatory["raw_leaderboard"].iloc[0]["candidate_name"] if not confirmatory["raw_leaderboard"].empty else "",
        "holdout_best_tuned_candidate": confirmatory["tuned_leaderboard"].iloc[0]["candidate_name"] if not confirmatory["tuned_leaderboard"].empty else "",
        "ensemble_best": confirmatory["ensemble_df"].iloc[0]["ensemble_name"] if not confirmatory["ensemble_df"].empty else "",
        "frozen_recreation_rows": int(len(confirmatory["frozen_recreation_df"])),
    }
    dump_json(summary_payload, output_dir / "summary.json")
    log("[done] bateria experimental concluida.", log_path)


if __name__ == "__main__":
    main()
