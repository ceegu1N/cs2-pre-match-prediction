#!/usr/bin/env python3
"""Utilitarios de suporte para a bateria experimental principal.

Objetivo:
- manter uma busca cega por subsets compactos e interpretaveis;
- comparar familias classicas de modelos pedidas no TCC;
- apoiar repeated CV, threshold tuning e fechamento confirmatorio;
- gerar artefatos finais prontos para integracao ao fluxo oficial.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from train_model import DEFAULT_LOGREG_CONFIG_PATH, save_latex_table

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experimental_battery_core import (
    DEFAULT_DATA_PATH,
    DEFAULT_RANDOM_STATE,
    DEFAULT_REPORTS_ROOT,
    DEFAULT_WORKERS,
    CandidateSpec,
    FeatureSetSpec,
    ModelSpec,
    aggregate_metric_frame,
    aggregate_seed_variance,
    binary_metrics,
    bootstrap_metrics,
    build_inner_split_specs,
    build_pipeline,
    build_temporal_folds,
    candidate_manifest_frame,
    dedup_features,
    dump_json,
    ensure_dir,
    evaluate_probabilities,
    fold_to_frames,
    freeze_feature_space as freeze_feature_space_core,
    infer_feature_blocks,
    kv_pairs,
    load_dataset,
    log,
    params_to_dict,
    parse_csv_ints,
    predict_positive_probability,
    resolve_feature_columns,
    resolve_output_dir,
    resolve_seed_list,
    run_exploratory_search,
    season_sort_key,
    select_best_thresholds,
    sort_leaderboard,
    split_train_holdout,
    timestamp_now,
    write_csv,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STOCHASTIC_SEEDS_V6 = [42, 52, 62, 72, 82]
DEFAULT_BOOTSTRAP_RUNS = 400
DEFAULT_SHORTLIST_SIZE = 14
DEFAULT_SHORTLIST_PER_MODEL_FAMILY = 1
DEFAULT_REPEATED_CV_SPLITS = 5
DEFAULT_REPEATED_CV_REPEATS = 4
DEFAULT_BATCH_SIZE = 24
DEFAULT_TOP_KS = [5, 8, 11, 14]
BALANCED_EXTRA_TOP_KS = [17, 20]
DEFAULT_FEATURE_SEARCH_PROFILE = "balanced"
FEATURE_SEARCH_PROFILES = ("focused", "balanced", "expanded")
DEFAULT_THRESHOLD_OBJECTIVES_V6_1 = ["accuracy", "f1"]
TARGETED_L2_FEATURE_SETS_V6_1 = (
    "univariate_top_14",
    "univariate_top_17",
    "univariate_top_20",
    "univariate_top_8",
    "univariate_top_11",
    "consensus_top_11",
    "block_budget_compact_11",
    "block_budget_compact_11_conservative",
    "block_budget_compact_11_univariate",
)
TARGETED_L2_PER_FEATURE_SET_V6_1 = 2


def build_block_budget_feature_specs_v6(feature_columns: list[str], *, k: int) -> list[FeatureSetSpec]:
    feature_blocks = infer_feature_blocks(feature_columns)
    required_blocks = {"recent_form", "elo_context", "combat_core", "utility_vision", "entry_trade"}
    if not required_blocks.issubset(set(feature_blocks)):
        return []

    return [
        FeatureSetSpec(
            name=f"block_budget_compact_{k}",
            family="block_budget_top_k",
            params=kv_pairs(
                k=k,
                scheme="combined_rank_score",
                budget_recent_form=1,
                budget_elo_context=1,
                budget_combat_core=4,
                budget_utility_vision=2,
                budget_entry_trade=3,
            ),
        ),
        FeatureSetSpec(
            name=f"block_budget_compact_{k}_conservative",
            family="block_budget_top_k",
            params=kv_pairs(
                k=k,
                scheme="conservative_rank_score",
                budget_recent_form=1,
                budget_elo_context=1,
                budget_combat_core=4,
                budget_utility_vision=2,
                budget_entry_trade=3,
            ),
        ),
        FeatureSetSpec(
            name=f"block_budget_compact_{k}_univariate",
            family="block_budget_top_k",
            params=kv_pairs(
                k=k,
                scheme="univariate_tilt_rank_score",
                budget_recent_form=1,
                budget_elo_context=1,
                budget_combat_core=4,
                budget_utility_vision=2,
                budget_entry_trade=3,
            ),
        ),
    ]


def build_feature_set_specs_v6(
    feature_columns: list[str],
    *,
    smoke_test: bool,
    search_profile: str,
) -> list[FeatureSetSpec]:
    feature_count = len(dedup_features(feature_columns))

    def capped(value: int) -> int:
        return min(feature_count, int(value))

    specs: list[FeatureSetSpec] = [
        FeatureSetSpec(name="all_snapshot_features", family="all_features", params=kv_pairs()),
    ]

    if smoke_test:
        for raw_k in [5, 11]:
            k = capped(raw_k)
            specs.append(FeatureSetSpec(name=f"univariate_top_{k}", family="univariate_top_k", params=kv_pairs(k=k)))
            specs.append(FeatureSetSpec(name=f"l1_top_{k}", family="l1_top_k", params=kv_pairs(k=k)))
            specs.append(FeatureSetSpec(name=f"tree_top_{k}", family="tree_top_k", params=kv_pairs(k=k)))
            specs.append(
                FeatureSetSpec(
                    name=f"consensus_top_{k}",
                    family="rank_scheme_top_k",
                    params=kv_pairs(k=k, scheme="combined_rank_score"),
                )
            )
        return dedup_feature_specs(specs)

    core_ks = [capped(value) for value in DEFAULT_TOP_KS if capped(value) > 0]
    for k in core_ks:
        specs.append(FeatureSetSpec(name=f"univariate_top_{k}", family="univariate_top_k", params=kv_pairs(k=k)))
        specs.append(FeatureSetSpec(name=f"l1_top_{k}", family="l1_top_k", params=kv_pairs(k=k)))
        specs.append(FeatureSetSpec(name=f"tree_top_{k}", family="tree_top_k", params=kv_pairs(k=k)))
        specs.append(
            FeatureSetSpec(
                name=f"consensus_top_{k}",
                family="rank_scheme_top_k",
                params=kv_pairs(k=k, scheme="combined_rank_score"),
            )
        )

    if search_profile == "focused":
        specs.append(
            FeatureSetSpec(
                name="stable_l1_rate_050",
                family="stable_l1_min_rate",
                params=kv_pairs(min_rate=0.50, fallback_k=capped(8)),
            )
        )
        return dedup_feature_specs(specs)

    if search_profile == "balanced":
        k = capped(11)
        extra_ks = [capped(value) for value in BALANCED_EXTRA_TOP_KS if capped(value) > 0]
        for extra_k in extra_ks:
            specs.append(FeatureSetSpec(name=f"univariate_top_{extra_k}", family="univariate_top_k", params=kv_pairs(k=extra_k)))
            specs.append(FeatureSetSpec(name=f"l1_top_{extra_k}", family="l1_top_k", params=kv_pairs(k=extra_k)))
            specs.append(FeatureSetSpec(name=f"tree_top_{extra_k}", family="tree_top_k", params=kv_pairs(k=extra_k)))
            specs.append(
                FeatureSetSpec(
                    name=f"consensus_top_{extra_k}",
                    family="rank_scheme_top_k",
                    params=kv_pairs(k=extra_k, scheme="combined_rank_score"),
                )
            )
        specs.extend(
            [
                FeatureSetSpec(name=f"tree_tilt_top_{k}", family="rank_scheme_top_k", params=kv_pairs(k=k, scheme="tree_tilt_rank_score")),
                FeatureSetSpec(name=f"l1_tilt_top_{k}", family="rank_scheme_top_k", params=kv_pairs(k=k, scheme="l1_tilt_rank_score")),
                FeatureSetSpec(name=f"univariate_tilt_top_{k}", family="rank_scheme_top_k", params=kv_pairs(k=k, scheme="univariate_tilt_rank_score")),
                FeatureSetSpec(name=f"conservative_top_{k}", family="rank_scheme_top_k", params=kv_pairs(k=k, scheme="conservative_rank_score")),
                FeatureSetSpec(name=f"intersection_vote_top_{k}_min2", family="intersection_vote_top_k", params=kv_pairs(k=k, min_votes=2)),
                FeatureSetSpec(name=f"intersection_vote_top_{k}_min3", family="intersection_vote_top_k", params=kv_pairs(k=k, min_votes=3)),
                FeatureSetSpec(name=f"union_vote_top_{k}", family="union_vote_top_k", params=kv_pairs(k=k)),
                FeatureSetSpec(name=f"stable_combined_top_{k}", family="stable_combined_top_k", params=kv_pairs(k=k, min_rate=0.50)),
                FeatureSetSpec(name=f"stable_combined_top_{k}_strict", family="stable_combined_top_k", params=kv_pairs(k=k, min_rate=0.67)),
                FeatureSetSpec(name="stable_l1_rate_050", family="stable_l1_min_rate", params=kv_pairs(min_rate=0.50, fallback_k=k)),
                FeatureSetSpec(name="stable_l1_rate_067", family="stable_l1_min_rate", params=kv_pairs(min_rate=0.67, fallback_k=capped(8))),
            ]
        )
        specs.extend(build_block_budget_feature_specs_v6(feature_columns, k=k))
        return dedup_feature_specs(specs)

    if search_profile != "expanded":
        raise ValueError(f"Perfil de busca invalido: {search_profile}")

    primary_ks = [capped(value) for value in [5, 8, 10, 11, 12, 14] if capped(value) > 0]
    scheme_ks = [capped(value) for value in [10, 11, 12] if capped(value) > 0]
    vote_ks = [capped(value) for value in [8, 10, 11, 12, 14] if capped(value) > 0]

    for k in primary_ks:
        specs.append(FeatureSetSpec(name=f"univariate_top_{k}", family="univariate_top_k", params=kv_pairs(k=k)))
        specs.append(FeatureSetSpec(name=f"l1_top_{k}", family="l1_top_k", params=kv_pairs(k=k)))
        specs.append(FeatureSetSpec(name=f"tree_top_{k}", family="tree_top_k", params=kv_pairs(k=k)))
        specs.append(
            FeatureSetSpec(
                name=f"consensus_top_{k}",
                family="rank_scheme_top_k",
                params=kv_pairs(k=k, scheme="combined_rank_score"),
            )
        )

    for k in scheme_ks:
        specs.extend(
            [
                FeatureSetSpec(name=f"tree_tilt_top_{k}", family="rank_scheme_top_k", params=kv_pairs(k=k, scheme="tree_tilt_rank_score")),
                FeatureSetSpec(name=f"l1_tilt_top_{k}", family="rank_scheme_top_k", params=kv_pairs(k=k, scheme="l1_tilt_rank_score")),
                FeatureSetSpec(name=f"univariate_tilt_top_{k}", family="rank_scheme_top_k", params=kv_pairs(k=k, scheme="univariate_tilt_rank_score")),
                FeatureSetSpec(name=f"conservative_top_{k}", family="rank_scheme_top_k", params=kv_pairs(k=k, scheme="conservative_rank_score")),
            ]
        )

    for k in vote_ks:
        specs.extend(
            [
                FeatureSetSpec(name=f"intersection_vote_top_{k}_min2", family="intersection_vote_top_k", params=kv_pairs(k=k, min_votes=2)),
                FeatureSetSpec(name=f"intersection_vote_top_{k}_min3", family="intersection_vote_top_k", params=kv_pairs(k=k, min_votes=3)),
                FeatureSetSpec(name=f"union_vote_top_{k}", family="union_vote_top_k", params=kv_pairs(k=k)),
                FeatureSetSpec(name=f"stable_combined_top_{k}", family="stable_combined_top_k", params=kv_pairs(k=k, min_rate=0.50)),
                FeatureSetSpec(name=f"stable_combined_top_{k}_strict", family="stable_combined_top_k", params=kv_pairs(k=k, min_rate=0.67)),
            ]
        )

    specs.extend(
        [
            FeatureSetSpec(name="stable_l1_rate_050", family="stable_l1_min_rate", params=kv_pairs(min_rate=0.50, fallback_k=capped(11))),
            FeatureSetSpec(name="stable_l1_rate_067", family="stable_l1_min_rate", params=kv_pairs(min_rate=0.67, fallback_k=capped(8))),
            FeatureSetSpec(name="stable_l1_rate_080", family="stable_l1_min_rate", params=kv_pairs(min_rate=0.80, fallback_k=capped(8))),
        ]
    )
    specs.extend(build_block_budget_feature_specs_v6(feature_columns, k=capped(11)))
    return dedup_feature_specs(specs)


def dedup_feature_specs(specs: list[FeatureSetSpec]) -> list[FeatureSetSpec]:
    deduped: list[FeatureSetSpec] = []
    seen: set[str] = set()
    for spec in specs:
        if spec.name not in seen:
            deduped.append(spec)
            seen.add(spec.name)
    return deduped


def tuned_metric_name_for_objective(objective: str) -> str:
    objective_name = str(objective).strip().lower()
    if objective_name in {"accuracy", "precision", "recall", "f1"}:
        return objective_name
    return "f1"


def sort_tuned_threshold_rows(frame: pd.DataFrame, *, primary_objective: str) -> pd.DataFrame:
    metric_name = tuned_metric_name_for_objective(primary_objective)
    if frame.empty:
        return frame.copy()

    return frame.sort_values(
        [metric_name, "roc_auc", "log_loss", "brier", "candidate_name"],
        ascending=[False, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)


def build_threshold_tuning_tables_v6_1(
    prediction_df: pd.DataFrame,
    *,
    objectives: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if prediction_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    grid_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    for candidate_name, candidate_frame in prediction_df.groupby("candidate_name", sort=False):
        y_true = candidate_frame["y_true"].astype(int).to_numpy()
        y_prob = candidate_frame["probability"].astype(float).to_numpy()
        threshold_grid, threshold_best = select_best_thresholds(y_true, y_prob, objectives)

        for row in threshold_grid.to_dict(orient="records"):
            grid_rows.append({"candidate_name": str(candidate_name), **row})
        for row in threshold_best:
            best_rows.append({"candidate_name": str(candidate_name), **row})

    return pd.DataFrame(grid_rows), pd.DataFrame(best_rows)


def summarize_threshold_recommendations_v6_1(
    threshold_best: pd.DataFrame,
    *,
    candidate_name: str,
) -> dict[str, object]:
    if threshold_best.empty:
        return {}

    candidate_thresholds = threshold_best.loc[threshold_best["candidate_name"].astype(str) == str(candidate_name)].copy()
    if candidate_thresholds.empty:
        return {}

    summary: dict[str, object] = {}
    for row in candidate_thresholds.to_dict(orient="records"):
        objective = str(row["objective"])
        metric_name = tuned_metric_name_for_objective(objective)
        suffix = objective.lower()
        summary[f"recommended_threshold_{suffix}"] = float(row["threshold"])
        summary[f"repeated_cv_{metric_name}_{suffix}"] = float(row[metric_name])
    if not candidate_thresholds.empty:
        primary_row = candidate_thresholds.iloc[0]
        summary["recommended_threshold_primary_objective"] = str(primary_row["objective"])
        summary["recommended_threshold_primary_value"] = float(primary_row["threshold"])
    return summary


def build_model_specs_v6(*, smoke_test: bool) -> list[ModelSpec]:
    specs: list[ModelSpec] = [ModelSpec(name="dummy_prior", family="dummy_prior", params=kv_pairs())]

    if smoke_test:
        return [
            ModelSpec(name="dummy_prior", family="dummy_prior", params=kv_pairs()),
            ModelSpec(name="logreg_l2_c1.0_cwbalanced", family="logreg_l2", params=kv_pairs(C=1.0, class_weight="balanced")),
            ModelSpec(name="logreg_l1_c0.1_cwbalanced", family="logreg_l1", params=kv_pairs(C=0.1, class_weight="balanced")),
            ModelSpec(name="gaussian_nb_1e-8", family="gaussian_nb", params=kv_pairs(var_smoothing=1e-8)),
            ModelSpec(name="knn_k5_distance", family="knn", params=kv_pairs(n_neighbors=5, weights="distance")),
            ModelSpec(name="svm_rbf_c1.0_scale", family="svm_rbf", params=kv_pairs(C=1.0, gamma="scale")),
            ModelSpec(
                name="rf_n400_dnone_l2",
                family="random_forest",
                params=kv_pairs(n_estimators=400, max_depth=None, min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample"),
            ),
        ]

    for c in [0.03, 0.10, 0.30, 1.0, 3.0, 5.0, 10.0]:
        for class_weight in [None, "balanced"]:
            cw_name = "balanced" if class_weight else "none"
            specs.append(ModelSpec(name=f"logreg_l2_c{c}_cw{cw_name}", family="logreg_l2", params=kv_pairs(C=c, class_weight=class_weight)))

    for c in [0.03, 0.10, 0.30, 1.0]:
        for class_weight in [None, "balanced"]:
            cw_name = "balanced" if class_weight else "none"
            specs.append(ModelSpec(name=f"logreg_l1_c{c}_cw{cw_name}", family="logreg_l1", params=kv_pairs(C=c, class_weight=class_weight)))

    for c in [0.10, 0.30, 1.0]:
        for l1_ratio in [0.20, 0.50, 0.80]:
            ratio_name = str(l1_ratio).replace(".", "")
            specs.append(
                ModelSpec(
                    name=f"logreg_elastic_c{c}_r{ratio_name}",
                    family="logreg_elasticnet",
                    params=kv_pairs(C=c, class_weight="balanced", l1_ratio=l1_ratio),
                )
            )

    for smoothing in [1e-9, 1e-8, 1e-7, 1e-6]:
        specs.append(ModelSpec(name=f"gaussian_nb_{smoothing:.0e}", family="gaussian_nb", params=kv_pairs(var_smoothing=smoothing)))

    for n_neighbors in [1, 5, 10]:
        for weights in ["uniform", "distance"]:
            specs.append(ModelSpec(name=f"knn_k{n_neighbors}_{weights}", family="knn", params=kv_pairs(n_neighbors=n_neighbors, weights=weights)))

    for c in [0.10, 1.0, 3.0]:
        for class_weight in [None, "balanced"]:
            cw_name = "balanced" if class_weight else "none"
            specs.append(ModelSpec(name=f"svm_linear_c{c}_cw{cw_name}", family="svm_linear", params=kv_pairs(C=c, class_weight=class_weight)))

    for c in [0.30, 1.0, 3.0]:
        for gamma in ["scale", "auto"]:
            specs.append(ModelSpec(name=f"svm_rbf_c{c}_{gamma}", family="svm_rbf", params=kv_pairs(C=c, gamma=gamma)))

    for n_estimators, max_depth, min_samples_leaf, max_features in [
        (400, None, 2, "sqrt"),
        (800, None, 1, "sqrt"),
        (800, 16, 2, "sqrt"),
        (1200, 20, 1, 0.5),
    ]:
        specs.append(
            ModelSpec(
                name=f"rf_n{n_estimators}_d{str(max_depth).lower()}_l{min_samples_leaf}_f{str(max_features).replace('.', 'p')}",
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

    return specs


def build_candidate_specs_v6(feature_specs: list[FeatureSetSpec], model_specs: list[ModelSpec]) -> list[CandidateSpec]:
    return [
        CandidateSpec(
            name=f"{feature_spec.name}__{model_spec.name}",
            feature_set_name=feature_spec.name,
            model_name=model_spec.name,
        )
        for feature_spec in feature_specs
        for model_spec in model_specs
    ]


def freeze_feature_space_v6(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    feature_specs: list[FeatureSetSpec],
    random_state: int,
    output_dir: Path | None,
    smoke_test: bool,
    search_profile: str,
) -> dict[str, object]:
    return freeze_feature_space_core(
        frame,
        feature_columns=feature_columns,
        official_features=[],
        feature_specs=feature_specs,
        random_state=random_state,
        preset="smoke" if smoke_test else search_profile,
        output_dir=output_dir,
    )


def prepare_exploratory_folds_v6(
    *,
    train_df: pd.DataFrame,
    temporal_folds: list,
    feature_columns: list[str],
    feature_specs: list[FeatureSetSpec],
    random_state: int,
    output_dir: Path,
    log_path: Path,
    smoke_test: bool,
    search_profile: str,
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
        feature_space = freeze_feature_space_v6(
            fold_train,
            feature_columns=feature_columns,
            feature_specs=feature_specs,
            random_state=random_state,
            output_dir=fold_dir,
            smoke_test=smoke_test,
            search_profile=search_profile,
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


def freeze_shortlist_v6(
    leaderboard: pd.DataFrame,
    *,
    shortlist_size: int,
    per_model_family: int,
    output_path: Path,
) -> pd.DataFrame:
    ranked = sort_leaderboard(leaderboard)
    selected_names: list[str] = []
    seen: set[str] = set()

    def add_names(values: Iterable[str]) -> None:
        for value in values:
            candidate_name = str(value)
            if candidate_name and candidate_name not in seen:
                selected_names.append(candidate_name)
                seen.add(candidate_name)

    add_names(ranked.head(shortlist_size)["candidate_name"].astype(str).tolist())
    if "model_family" in ranked.columns and per_model_family > 0:
        for _, group in ranked.groupby("model_family", sort=False):
            add_names(group.head(per_model_family)["candidate_name"].astype(str).tolist())
    if {"feature_set_name", "model_family"}.issubset(ranked.columns):
        for feature_set_name in TARGETED_L2_FEATURE_SETS_V6_1:
            l2_group = ranked.loc[
                (ranked["feature_set_name"].astype(str) == feature_set_name)
                & (ranked["model_family"].astype(str) == "logreg_l2")
            ].copy()
            if not l2_group.empty:
                add_names(
                    l2_group.head(TARGETED_L2_PER_FEATURE_SET_V6_1)["candidate_name"].astype(str).tolist()
                )

    shortlist = ranked.loc[ranked["candidate_name"].astype(str).isin(selected_names)].copy()
    shortlist = sort_leaderboard(shortlist).reset_index(drop=True)
    shortlist.insert(0, "shortlist_rank", np.arange(1, len(shortlist) + 1))
    write_csv(shortlist, output_path)
    return shortlist


def evaluate_finalist_holdout_v6(
    *,
    candidate_row: dict[str, object],
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    features: list[str],
    model_lookup: dict[str, ModelSpec],
    random_state: int,
    stochastic_seeds: Sequence[int],
    bootstrap_runs: int,
) -> dict[str, object]:
    candidate_name = str(candidate_row["candidate_name"])
    feature_set_name = str(candidate_row["feature_set_name"])
    model_name = str(candidate_row["model_name"])
    model_family = str(candidate_row["model_family"])
    model_spec = model_lookup[model_name]
    model_params = params_to_dict(model_spec.params)
    seed_list = resolve_seed_list(model_spec.family, stochastic_seeds, random_state)
    y_holdout = holdout_df["win_target"].astype(int).to_numpy()

    holdout_prob_by_seed: list[np.ndarray] = []
    result_rows: list[dict[str, object]] = []
    for seed in seed_list:
        estimator = build_pipeline(model_spec.family, model_params, int(seed))
        estimator.fit(train_df[features], train_df["win_target"].astype(int))
        probabilities = predict_positive_probability(estimator, holdout_df, features)
        holdout_prob_by_seed.append(probabilities)
        metrics = evaluate_probabilities(y_holdout, probabilities, threshold=0.5)
        result_rows.append(
            {
                "candidate_name": candidate_name,
                "feature_set_name": feature_set_name,
                "model_name": model_name,
                "model_family": model_family,
                "feature_count": int(len(features)),
                "features": ",".join(features),
                "seed": int(seed),
                "variant": "seed",
                **metrics,
            }
        )

    aggregate_prob = np.mean(np.vstack(holdout_prob_by_seed), axis=0)
    aggregate_metrics = evaluate_probabilities(y_holdout, aggregate_prob, threshold=0.5)
    result_rows.append(
        {
            "candidate_name": candidate_name,
            "feature_set_name": feature_set_name,
            "model_name": model_name,
            "model_family": model_family,
            "feature_count": int(len(features)),
            "features": ",".join(features),
            "seed": "aggregate",
            "variant": "aggregate",
            **aggregate_metrics,
        }
    )

    bootstrap_detail = pd.DataFrame()
    bootstrap_summary_row: dict[str, object] = {
        "candidate_name": candidate_name,
        "bootstrap_runs": int(bootstrap_runs),
    }
    if bootstrap_runs > 0:
        bootstrap_detail, bootstrap_summary = bootstrap_metrics(
            y_holdout,
            aggregate_prob,
            threshold=0.5,
            runs=bootstrap_runs,
            random_state=random_state + 9000,
        )
        bootstrap_summary_row.update(bootstrap_summary)
    return {
        "result_rows": result_rows,
        "bootstrap_detail": bootstrap_detail.assign(candidate_name=candidate_name) if not bootstrap_detail.empty else bootstrap_detail,
        "bootstrap_summary_row": bootstrap_summary_row,
        "aggregate_probabilities": aggregate_prob,
        "y_holdout": y_holdout,
    }


def repeated_cv_finalists_v6(
    *,
    train_df: pd.DataFrame,
    shortlist: pd.DataFrame,
    feature_columns: list[str],
    feature_specs: list[FeatureSetSpec],
    model_lookup: dict[str, ModelSpec],
    random_state: int,
    stochastic_seeds: Sequence[int],
    n_splits: int,
    n_repeats: int,
    smoke_test: bool,
    search_profile: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = train_df["win_target"].astype(int).to_numpy()
    min_class_count = int(pd.Series(y).value_counts().min())
    n_splits = min(max(2, n_splits), min_class_count)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    split_plan = list(cv.split(np.zeros(len(train_df)), y))

    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    local_rows_by_candidate: dict[str, list[dict[str, object]]] = {
        str(row["candidate_name"]): [] for row in shortlist.to_dict(orient="records")
    }
    feature_count_trace_by_candidate: dict[str, list[str]] = {
        str(row["candidate_name"]): [] for row in shortlist.to_dict(orient="records")
    }

    for split_index, (train_idx, valid_idx) in enumerate(split_plan, start=1):
        split_train = train_df.iloc[np.asarray(train_idx, dtype=int)].copy()
        split_valid = train_df.iloc[np.asarray(valid_idx, dtype=int)].copy()
        split_feature_space = freeze_feature_space_v6(
            split_train,
            feature_columns=feature_columns,
            feature_specs=feature_specs,
            random_state=random_state + split_index,
            output_dir=None,
            smoke_test=smoke_test,
            search_profile=search_profile,
        )

        for shortlist_row in shortlist.to_dict(orient="records"):
            candidate_name = str(shortlist_row["candidate_name"])
            model_spec = model_lookup[str(shortlist_row["model_name"])]
            feature_set_name = str(shortlist_row["feature_set_name"])
            features = list(split_feature_space["feature_sets"][feature_set_name])
            if not features:
                raise ValueError(f"Subset vazio no repeated CV para {candidate_name} no split {split_index}.")
            feature_count_trace_by_candidate[candidate_name].append(f"split_{split_index}:{len(features)}")
            model_params = params_to_dict(model_spec.params)
            seed_list = resolve_seed_list(model_spec.family, stochastic_seeds, random_state)

            for seed in seed_list:
                estimator = build_pipeline(model_spec.family, model_params, int(seed))
                estimator.fit(split_train[features], split_train["win_target"].astype(int))
                probabilities = predict_positive_probability(estimator, split_valid, features)
                y_valid = split_valid["win_target"].astype(int).to_numpy()
                metrics = evaluate_probabilities(y_valid, probabilities, threshold=0.5)
                row = {
                    "candidate_name": candidate_name,
                    "feature_set_name": feature_set_name,
                    "model_name": str(shortlist_row["model_name"]),
                    "model_family": str(shortlist_row["model_family"]),
                    "split_index": split_index,
                    "seed": int(seed),
                    "feature_count": int(len(features)),
                    "features": ",".join(features),
                    **metrics,
                }
                local_rows_by_candidate[candidate_name].append(row)
                fold_rows.append(row)
                for row_index, probability, y_true in zip(split_valid.index.tolist(), probabilities.tolist(), y_valid.tolist()):
                    prediction_rows.append(
                        {
                            "candidate_name": candidate_name,
                            "feature_set_name": feature_set_name,
                            "model_name": str(shortlist_row["model_name"]),
                            "model_family": str(shortlist_row["model_family"]),
                            "split_index": int(split_index),
                            "seed": int(seed),
                            "row_id": int(row_index),
                            "y_true": int(y_true),
                            "probability": float(probability),
                        }
                    )

    for shortlist_row in shortlist.to_dict(orient="records"):
        candidate_name = str(shortlist_row["candidate_name"])
        model_spec = model_lookup[str(shortlist_row["model_name"])]
        metric_frame = pd.DataFrame(local_rows_by_candidate[candidate_name])
        feature_counts = metric_frame["feature_count"].astype(int).tolist()
        summary = {
            "candidate_name": candidate_name,
            "feature_set_name": str(shortlist_row["feature_set_name"]),
            "model_name": str(shortlist_row["model_name"]),
            "model_family": model_spec.family,
            "repeated_cv_split_count": int(metric_frame["split_index"].nunique()),
            "repeated_cv_eval_count": int(len(metric_frame)),
            "stochastic_seed_count": int(metric_frame["seed"].nunique()),
            "resolved_feature_count_min": int(min(feature_counts)),
            "resolved_feature_count_mean": float(np.mean(feature_counts)),
            "resolved_feature_count_max": int(max(feature_counts)),
            "resolved_feature_counts": ";".join(feature_count_trace_by_candidate[candidate_name]),
            "resolved_features_last_split": str(metric_frame.iloc[-1]["features"]),
        }
        summary.update(aggregate_metric_frame(metric_frame, "repeated_cv"))
        summary.update(aggregate_seed_variance(metric_frame, "repeated_cv"))
        summary_rows.append(summary)

    repeated_summary = sort_leaderboard(
        pd.DataFrame(summary_rows).rename(
            columns={
                "repeated_cv_roc_auc_mean": "temporal_cv_roc_auc_mean",
                "repeated_cv_log_loss_mean": "temporal_cv_log_loss_mean",
                "repeated_cv_brier_mean": "temporal_cv_brier_mean",
                "repeated_cv_accuracy_mean": "temporal_cv_accuracy_mean",
            }
        )
    ).rename(
        columns={
            "temporal_cv_roc_auc_mean": "repeated_cv_roc_auc_mean",
            "temporal_cv_log_loss_mean": "repeated_cv_log_loss_mean",
            "temporal_cv_brier_mean": "repeated_cv_brier_mean",
            "temporal_cv_accuracy_mean": "repeated_cv_accuracy_mean",
        }
    )
    return pd.DataFrame(fold_rows), repeated_summary, pd.DataFrame(prediction_rows)


def run_confirmatory_phase_v6(
    *,
    output_dir: Path,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    shortlist: pd.DataFrame,
    feature_columns: list[str],
    feature_specs: list[FeatureSetSpec],
    model_lookup: dict[str, ModelSpec],
    random_state: int,
    stochastic_seeds: Sequence[int],
    repeated_cv_splits: int,
    repeated_cv_repeats: int,
    bootstrap_runs: int,
    threshold_objectives: Sequence[str],
    smoke_test: bool,
    search_profile: str,
    log_path: Path,
) -> dict[str, pd.DataFrame]:
    confirmatory_dir = output_dir / "confirmatory"
    ensure_dir(confirmatory_dir)
    y_holdout = holdout_df["win_target"].astype(int).to_numpy()

    full_train_feature_space = freeze_feature_space_v6(
        train_df,
        feature_columns=feature_columns,
        feature_specs=feature_specs,
        random_state=random_state,
        output_dir=confirmatory_dir / "full_train_feature_space",
        smoke_test=smoke_test,
        search_profile=search_profile,
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

    repeated_fold_df, repeated_summary_df, repeated_prediction_df = repeated_cv_finalists_v6(
        train_df=train_df,
        shortlist=shortlist,
        feature_columns=feature_columns,
        feature_specs=feature_specs,
        model_lookup=model_lookup,
        random_state=random_state,
        stochastic_seeds=stochastic_seeds,
        n_splits=repeated_cv_splits,
        n_repeats=repeated_cv_repeats,
        smoke_test=smoke_test,
        search_profile=search_profile,
    )
    write_csv(repeated_fold_df, confirmatory_dir / "repeated_cv_fold_metrics.csv")
    write_csv(repeated_summary_df, confirmatory_dir / "repeated_cv_summary.csv")
    write_csv(repeated_prediction_df, confirmatory_dir / "repeated_cv_predictions.csv")

    threshold_grid_df, threshold_best_df = build_threshold_tuning_tables_v6_1(
        repeated_prediction_df,
        objectives=threshold_objectives,
    )
    write_csv(threshold_grid_df, confirmatory_dir / "threshold_grid.csv")
    write_csv(threshold_best_df, confirmatory_dir / "threshold_best.csv")
    primary_threshold_objective = str(threshold_objectives[0]) if threshold_objectives else "accuracy"

    holdout_rows: list[dict[str, object]] = []
    holdout_tuned_rows: list[dict[str, object]] = []
    bootstrap_summary_rows: list[dict[str, object]] = []
    bootstrap_detail_frames: list[pd.DataFrame] = []
    holdout_probability_by_candidate: dict[str, np.ndarray] = {}
    for index, shortlist_row in enumerate(shortlist.to_dict(orient="records"), start=1):
        candidate_name = str(shortlist_row["candidate_name"])
        log(f"[confirmatory] avaliando finalista {index}/{len(shortlist)}: {candidate_name}", log_path)
        result = evaluate_finalist_holdout_v6(
            candidate_row=shortlist_row,
            train_df=train_df,
            holdout_df=holdout_df,
            features=finalist_feature_map[candidate_name],
            model_lookup=model_lookup,
            random_state=random_state + index,
            stochastic_seeds=stochastic_seeds,
            bootstrap_runs=bootstrap_runs,
        )
        holdout_rows.extend(result["result_rows"])
        holdout_probability_by_candidate[candidate_name] = np.asarray(result["aggregate_probabilities"], dtype=float)
        bootstrap_summary_rows.append(result["bootstrap_summary_row"])
        if isinstance(result["bootstrap_detail"], pd.DataFrame) and not result["bootstrap_detail"].empty:
            bootstrap_detail_frames.append(result["bootstrap_detail"])

    holdout_df_all = pd.DataFrame(holdout_rows)
    write_csv(holdout_df_all, confirmatory_dir / "holdout_metrics.csv")
    write_csv(pd.DataFrame(bootstrap_summary_rows), confirmatory_dir / "bootstrap_summary.csv")
    write_csv(
        pd.concat(bootstrap_detail_frames, ignore_index=True) if bootstrap_detail_frames else pd.DataFrame(),
        confirmatory_dir / "bootstrap_detail.csv",
    )

    holdout_aggregate = holdout_df_all.loc[holdout_df_all["variant"] == "aggregate"].copy()
    holdout_leaderboard = sort_leaderboard(
        holdout_aggregate.rename(
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
    write_csv(holdout_leaderboard, confirmatory_dir / "holdout_leaderboard.csv")

    if not threshold_best_df.empty:
        best_rows_lookup = {
            (str(row["candidate_name"]), str(row["objective"])): row
            for row in threshold_best_df.to_dict(orient="records")
        }
        for shortlist_row in shortlist.to_dict(orient="records"):
            candidate_name = str(shortlist_row["candidate_name"])
            aggregate_prob = holdout_probability_by_candidate.get(candidate_name)
            if aggregate_prob is None:
                continue
            for objective in threshold_objectives:
                key = (candidate_name, str(objective))
                if key not in best_rows_lookup:
                    continue
                best_row = best_rows_lookup[key]
                threshold = float(best_row["threshold"])
                tuned_metrics = evaluate_probabilities(y_holdout, aggregate_prob, threshold=threshold)
                holdout_tuned_rows.append(
                    {
                        "candidate_name": candidate_name,
                        "feature_set_name": str(shortlist_row["feature_set_name"]),
                        "model_name": str(shortlist_row["model_name"]),
                        "model_family": str(shortlist_row["model_family"]),
                        "feature_count": int(len(finalist_feature_map[candidate_name])),
                        "features": ",".join(finalist_feature_map[candidate_name]),
                        "objective": str(objective),
                        "threshold_source": "repeated_cv_pool",
                        "threshold": threshold,
                        "variant": f"tuned_{objective}",
                        **tuned_metrics,
                    }
                )

    holdout_tuned_df = pd.DataFrame(
        holdout_tuned_rows,
        columns=[
            "candidate_name",
            "feature_set_name",
            "model_name",
            "model_family",
            "feature_count",
            "features",
            "objective",
            "threshold_source",
            "threshold",
            "variant",
            "roc_auc",
            "log_loss",
            "brier",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "ece",
        ],
    )
    write_csv(holdout_tuned_df, confirmatory_dir / "holdout_tuned_threshold_metrics.csv")
    tuned_primary = holdout_tuned_df.loc[holdout_tuned_df["objective"].astype(str) == primary_threshold_objective].copy()
    holdout_tuned_leaderboard = sort_tuned_threshold_rows(
        tuned_primary,
        primary_objective=primary_threshold_objective,
    )
    if not holdout_tuned_leaderboard.empty:
        holdout_tuned_leaderboard.insert(0, "tuned_rank", np.arange(1, len(holdout_tuned_leaderboard) + 1))
    write_csv(holdout_tuned_leaderboard, confirmatory_dir / "holdout_tuned_threshold_leaderboard.csv")

    shortlist_clean = shortlist.drop(columns=[column for column in ["rank"] if column in shortlist.columns]).copy()
    shortlist_clean = shortlist_clean[
        [
            "shortlist_rank",
            "candidate_name",
            "feature_set_name",
            "feature_family",
            "model_name",
            "model_family",
            "temporal_cv_roc_auc_mean",
            "temporal_cv_log_loss_mean",
            "temporal_cv_brier_mean",
            "temporal_cv_accuracy_mean",
        ]
    ]
    repeated_summary_clean = repeated_summary_df.drop(columns=[column for column in ["rank"] if column in repeated_summary_df.columns]).copy()
    repeated_summary_clean = repeated_summary_clean.rename(
        columns={
            "resolved_feature_count_mean": "repeated_cv_resolved_feature_count_mean",
            "resolved_feature_count_min": "repeated_cv_resolved_feature_count_min",
            "resolved_feature_count_max": "repeated_cv_resolved_feature_count_max",
        }
    )
    repeated_summary_clean = repeated_summary_clean[
        [
            "candidate_name",
            "repeated_cv_split_count",
            "repeated_cv_eval_count",
            "stochastic_seed_count",
            "repeated_cv_resolved_feature_count_mean",
            "repeated_cv_resolved_feature_count_min",
            "repeated_cv_resolved_feature_count_max",
            "repeated_cv_roc_auc_mean",
            "repeated_cv_roc_auc_std",
            "repeated_cv_log_loss_mean",
            "repeated_cv_brier_mean",
            "repeated_cv_accuracy_mean",
            "repeated_cv_seed_roc_auc_std_mean",
        ]
    ]
    holdout_clean = holdout_aggregate[
        ["candidate_name", "roc_auc", "log_loss", "brier", "accuracy", "precision", "recall", "f1", "ece"]
    ].rename(
        columns={
            "roc_auc": "holdout_roc_auc",
            "log_loss": "holdout_log_loss",
            "brier": "holdout_brier",
            "accuracy": "holdout_accuracy",
            "precision": "holdout_precision",
            "recall": "holdout_recall",
            "f1": "holdout_f1",
            "ece": "holdout_ece",
        }
    )
    holdout_tuned_clean = tuned_primary[
        ["candidate_name", "objective", "threshold", "accuracy", "precision", "recall", "f1"]
    ].rename(
        columns={
            "objective": "holdout_tuned_objective",
            "threshold": "holdout_tuned_threshold",
            "accuracy": "holdout_tuned_accuracy",
            "precision": "holdout_tuned_precision",
            "recall": "holdout_tuned_recall",
            "f1": "holdout_tuned_f1",
        }
    ) if not tuned_primary.empty else pd.DataFrame(
        columns=[
            "candidate_name",
            "holdout_tuned_objective",
            "holdout_tuned_threshold",
            "holdout_tuned_accuracy",
            "holdout_tuned_precision",
            "holdout_tuned_recall",
            "holdout_tuned_f1",
        ]
    )
    comparison = (
        shortlist_clean
        .merge(repeated_summary_clean, on="candidate_name", how="left")
        .merge(holdout_clean, on="candidate_name", how="left")
        .merge(holdout_tuned_clean, on="candidate_name", how="left")
    )
    comparison["holdout_tuned_accuracy_gain"] = comparison["holdout_tuned_accuracy"] - comparison["holdout_accuracy"]
    comparison = comparison.sort_values(
        ["holdout_roc_auc", "holdout_log_loss", "holdout_brier", "holdout_accuracy", "candidate_name"],
        ascending=[False, True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    comparison.insert(0, "final_rank", np.arange(1, len(comparison) + 1))
    comparison = comparison[
        [
            "final_rank",
            "shortlist_rank",
            "candidate_name",
            "feature_set_name",
            "feature_family",
            "model_name",
            "model_family",
            "temporal_cv_roc_auc_mean",
            "temporal_cv_log_loss_mean",
            "temporal_cv_brier_mean",
            "temporal_cv_accuracy_mean",
            "repeated_cv_split_count",
            "repeated_cv_eval_count",
            "stochastic_seed_count",
            "repeated_cv_resolved_feature_count_mean",
            "repeated_cv_resolved_feature_count_min",
            "repeated_cv_resolved_feature_count_max",
            "repeated_cv_roc_auc_mean",
            "repeated_cv_roc_auc_std",
            "repeated_cv_log_loss_mean",
            "repeated_cv_brier_mean",
            "repeated_cv_accuracy_mean",
            "repeated_cv_seed_roc_auc_std_mean",
            "holdout_roc_auc",
            "holdout_log_loss",
            "holdout_brier",
            "holdout_accuracy",
            "holdout_precision",
            "holdout_recall",
            "holdout_f1",
            "holdout_ece",
            "holdout_tuned_objective",
            "holdout_tuned_threshold",
            "holdout_tuned_accuracy",
            "holdout_tuned_precision",
            "holdout_tuned_recall",
            "holdout_tuned_f1",
            "holdout_tuned_accuracy_gain",
        ]
    ]
    write_csv(comparison, confirmatory_dir / "final_candidate_summary.csv")

    return {
        "repeated_summary_df": repeated_summary_df,
        "threshold_best_df": threshold_best_df,
        "holdout_leaderboard": holdout_leaderboard,
        "holdout_tuned_leaderboard": holdout_tuned_leaderboard,
        "comparison": comparison,
    }


def estimate_v6_volume(
    *,
    feature_columns: list[str],
    feature_specs: list[FeatureSetSpec],
    model_specs: list[ModelSpec],
    train_df: pd.DataFrame,
    stochastic_seeds: Sequence[int],
    shortlist_size: int,
    repeated_cv_splits: int,
    repeated_cv_repeats: int,
    bootstrap_runs: int,
    smoke_test: bool,
) -> dict[str, object]:
    split_specs = build_inner_split_specs(
        train_df,
        random_state=DEFAULT_RANDOM_STATE,
        fallback_splits=3 if smoke_test else 5,
        fallback_repeats=1 if smoke_test else 2,
    )
    seed_counts = [len(resolve_seed_list(spec.family, stochastic_seeds, DEFAULT_RANDOM_STATE)) for spec in model_specs]
    candidate_count = len(feature_specs) * len(model_specs)
    temporal_fold_count = len(build_temporal_folds(train_df))
    exploratory_eval_count = int(len(feature_specs) * sum(seed_counts) * temporal_fold_count)
    ranking_fit_count = int(len(feature_columns) * len(split_specs) + (8 * len(split_specs)) + 2)
    mean_seed_count = float(np.mean(seed_counts)) if seed_counts else 1.0
    repeated_cv_split_count = int(repeated_cv_splits * repeated_cv_repeats)
    repeated_cv_feature_ranking_fit_count = int(repeated_cv_split_count * ranking_fit_count)
    repeated_cv_model_fit_count = int(round(shortlist_size * repeated_cv_split_count * mean_seed_count))
    repeated_cv_fit_count = int(repeated_cv_feature_ranking_fit_count + repeated_cv_model_fit_count)
    holdout_fit_count = int(round(shortlist_size * mean_seed_count))
    estimated_threshold_grid_scans = int(shortlist_size * repeated_cv_split_count * max(1, len(DEFAULT_THRESHOLD_OBJECTIVES_V6_1)))
    estimated_total_fits = int(ranking_fit_count + exploratory_eval_count + repeated_cv_fit_count + holdout_fit_count)
    return {
        "feature_count": int(len(feature_columns)),
        "feature_set_count": int(len(feature_specs)),
        "model_count": int(len(model_specs)),
        "candidate_count": int(candidate_count),
        "temporal_outer_fold_count": int(temporal_fold_count),
        "estimated_feature_ranking_fits": int(ranking_fit_count),
        "estimated_exploratory_fits": int(exploratory_eval_count),
        "estimated_repeated_cv_feature_ranking_fits": int(repeated_cv_feature_ranking_fit_count),
        "estimated_repeated_cv_model_fits": int(repeated_cv_model_fit_count),
        "estimated_repeated_cv_fits": int(repeated_cv_fit_count),
        "estimated_holdout_fits": int(holdout_fit_count),
        "estimated_threshold_grid_scans": int(estimated_threshold_grid_scans),
        "estimated_total_model_fits": int(estimated_total_fits),
        "bootstrap_runs_per_finalist": int(bootstrap_runs),
        "stochastic_seed_count": int(len(stochastic_seeds)),
        "inner_split_count": int(len(split_specs)),
    }


def build_pipeline_audit_v6(
    *,
    data_path: Path,
    feature_columns: list[str],
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    feature_specs: list[FeatureSetSpec],
    search_profile: str,
) -> dict[str, object]:
    return {
        "generated_at": datetime.now().isoformat(),
        "runner": str(ROOT / "run_experimental_battery.py"),
        "dataset_path": str(data_path),
        "dataset_feature_count": int(len(feature_columns)),
        "dataset_features": feature_columns,
        "predefined_reference_policy": {
            "predefined_feature_baseline_used": False,
            "predefined_config_loaded": False,
            "forced_reference_candidates": False,
            "selection_goal": "busca cega por subsets compactos e intermediarios a partir das features base do snapshot",
            "threshold_tuning_policy": "ajustado apenas para finalistas com previsoes internas da repeated CV do treino",
        },
        "feature_search_scope": {
            "search_profile": search_profile,
            "feature_set_count": int(len(feature_specs)),
            "families": sorted({spec.family for spec in feature_specs}),
            "base_feature_only": True,
            "compact_subset_focus": True,
            "intermediate_subset_focus": True,
        },
        "isolation_strategy": {
            "runner": str(ROOT / "run_experimental_battery.py"),
            "experimental_output_root": str(DEFAULT_REPORTS_ROOT),
            "write_policy": "somente diretorios versionados dentro de reports/experiments",
            "no_overwrite_targets": [
                str(ROOT / "models"),
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
        "scope": {
            "base_feature_only": True,
            "derived_features_used": False,
            "blind_to_predefined_subset": True,
            "intended_use": "runner experimental final com busca cega, shortlist protegida de L2 e threshold tuning apenas para finalistas",
        },
    }


def load_json_file(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_feature_list(raw_value: object) -> list[str]:
    if raw_value is None:
        return []
    return [item.strip() for item in str(raw_value).split(",") if item.strip()]


def parse_params_json(raw_value: object) -> dict[str, object]:
    if raw_value is None:
        return {}
    text = str(raw_value).strip()
    if not text:
        return {}
    return json.loads(text)


def dissertation_base_figure(width: float = 8.0, height: float = 5.0):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def load_v6_final_artifacts(output_dir: Path) -> dict[str, object]:
    confirmatory_dir = output_dir / "confirmatory"
    final_summary_path = confirmatory_dir / "final_candidate_summary.csv"
    feature_manifest_path = confirmatory_dir / "finalist_feature_manifest.csv"
    model_manifest_path = output_dir / "model_spec_manifest.csv"
    audit_path = output_dir / "pipeline_audit.json"

    required_paths = [
        final_summary_path,
        feature_manifest_path,
        model_manifest_path,
        audit_path,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "A pasta experimental ainda nao esta pronta para finalizacao. Artefatos ausentes: "
            + ", ".join(missing)
        )

    comparison = pd.read_csv(final_summary_path)
    if comparison.empty:
        raise ValueError("final_candidate_summary.csv existe, mas esta vazio.")

    return {
        "output_dir": output_dir,
        "confirmatory_dir": confirmatory_dir,
        "comparison": comparison,
        "feature_manifest": pd.read_csv(feature_manifest_path),
        "model_manifest": pd.read_csv(model_manifest_path),
        "audit": load_json_file(audit_path),
        "summary": load_json_file(output_dir / "summary.json") if (output_dir / "summary.json").exists() else {},
        "holdout_leaderboard": pd.read_csv(confirmatory_dir / "holdout_leaderboard.csv")
        if (confirmatory_dir / "holdout_leaderboard.csv").exists()
        else pd.DataFrame(),
        "repeated_summary": pd.read_csv(confirmatory_dir / "repeated_cv_summary.csv")
        if (confirmatory_dir / "repeated_cv_summary.csv").exists()
        else pd.DataFrame(),
        "threshold_best": pd.read_csv(confirmatory_dir / "threshold_best.csv")
        if (confirmatory_dir / "threshold_best.csv").exists()
        else pd.DataFrame(),
        "holdout_tuned_leaderboard": pd.read_csv(confirmatory_dir / "holdout_tuned_threshold_leaderboard.csv")
        if (confirmatory_dir / "holdout_tuned_threshold_leaderboard.csv").exists()
        else pd.DataFrame(),
    }


def select_logistic_candidate_for_official_integration(comparison: pd.DataFrame) -> tuple[pd.Series, dict[str, object]]:
    best_overall = comparison.sort_values("final_rank", ascending=True, kind="stable").iloc[0]
    logistic_families = {"logreg_l1", "logreg_l2", "logreg_elasticnet"}
    logistic_rows = comparison.loc[comparison["model_family"].isin(logistic_families)].copy()
    if logistic_rows.empty:
        raise ValueError("Nao ha finalistas de regressao logistica para integrar ao fluxo oficial.")

    logistic_rows = logistic_rows.sort_values(
        ["holdout_roc_auc", "holdout_log_loss", "holdout_brier", "holdout_accuracy", "final_rank"],
        ascending=[False, True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    selected = logistic_rows.iloc[0]
    metadata = {
        "selection_rule": "best_logistic_by_holdout_roc_auc_then_log_loss_then_brier_then_accuracy",
        "best_overall_candidate_name": str(best_overall["candidate_name"]),
        "best_overall_model_family": str(best_overall["model_family"]),
        "best_overall_holdout_roc_auc": float(best_overall["holdout_roc_auc"]),
        "selected_matches_best_overall": bool(str(selected["candidate_name"]) == str(best_overall["candidate_name"])),
    }
    return selected, metadata


def resolve_logistic_training_config(
    *,
    output_dir: Path,
    candidate_row: pd.Series,
    feature_manifest: pd.DataFrame,
    model_manifest: pd.DataFrame,
    selection_metadata: dict[str, object],
    threshold_best: pd.DataFrame,
) -> tuple[dict[str, object], list[str], dict[str, object]]:
    candidate_name = str(candidate_row["candidate_name"])
    feature_row = feature_manifest.loc[feature_manifest["candidate_name"] == candidate_name]
    if feature_row.empty:
        raise ValueError(f"Nao foi possivel localizar as features do candidato {candidate_name}.")
    features = parse_feature_list(feature_row.iloc[0]["features"])
    if not features:
        raise ValueError(f"O candidato {candidate_name} nao possui features resolvidas para exportacao.")

    model_name = str(candidate_row["model_name"])
    model_row = model_manifest.loc[model_manifest["model_name"] == model_name]
    if model_row.empty:
        raise ValueError(f"Nao foi possivel localizar os hiperparametros do modelo {model_name}.")
    model_params = parse_params_json(model_row.iloc[0]["params_json"])

    model_family = str(candidate_row["model_family"])
    if model_family == "logreg_l1":
        penalty = "l1"
        solver = "liblinear"
    elif model_family == "logreg_elasticnet":
        penalty = "elasticnet"
        solver = "saga"
    else:
        penalty = "l2"
        solver = "lbfgs"

    config: dict[str, object] = {
        "status": "training_configuration",
        "description": "Configuracao exportada automaticamente a partir do final da bateria experimental.",
        "source_experiment_dir": str(output_dir),
        "source_candidate_name": candidate_name,
        "feature_set_name": str(candidate_row["feature_set_name"]),
        "model_name": model_name,
        "model_family": model_family,
        "selection_rule": selection_metadata["selection_rule"],
        "best_overall_candidate_name": selection_metadata["best_overall_candidate_name"],
        "best_overall_model_family": selection_metadata["best_overall_model_family"],
        "holdout_roc_auc": float(candidate_row["holdout_roc_auc"]),
        "holdout_accuracy": float(candidate_row["holdout_accuracy"]),
        "repeated_cv_roc_auc_mean": float(candidate_row["repeated_cv_roc_auc_mean"]),
        "repeated_cv_roc_auc_std": float(candidate_row["repeated_cv_roc_auc_std"]),
        "penalty": penalty,
        "C": float(model_params.get("C", 1.0)),
        "class_weight": model_params.get("class_weight"),
        "solver": solver,
        "features": features,
    }
    if model_family == "logreg_elasticnet":
        config["l1_ratio"] = float(model_params.get("l1_ratio", 0.5))
    config.update(summarize_threshold_recommendations_v6_1(threshold_best, candidate_name=candidate_name))
    return config, features, model_params


def export_official_integration_from_v6(
    output_dir: Path,
    *,
    official_config_copy: Path | None,
    log_path: Path | None,
) -> dict[str, object]:
    artifacts = load_v6_final_artifacts(output_dir)
    comparison = artifacts["comparison"]
    selected, selection_metadata = select_logistic_candidate_for_official_integration(comparison)
    config, features, model_params = resolve_logistic_training_config(
        output_dir=output_dir,
        candidate_row=selected,
        feature_manifest=artifacts["feature_manifest"],
        model_manifest=artifacts["model_manifest"],
        selection_metadata=selection_metadata,
        threshold_best=artifacts["threshold_best"],
    )

    integration_dir = output_dir / "official_integration"
    ensure_dir(integration_dir)
    exported_config_path = integration_dir / "config_modelo_principal.json"
    dump_json(config, exported_config_path)

    copied_config_path = None
    if official_config_copy is not None:
        dump_json(config, official_config_copy)
        copied_config_path = str(official_config_copy)

    commands_text = "\n".join(
        [
            "Comandos sugeridos para integrar o modelo principal ao fluxo oficial:",
            f'python "{ROOT / "train_model.py"}" --training-config "{exported_config_path}"',
            f'python "{ROOT / "run_pipeline.py"}" --training-config "{exported_config_path}"',
            "",
            "Se quiser atualizar o caminho oficial padrao de configuracao, use:",
            f'python "{ROOT / "run_experimental_battery.py"}" --finalize-only --existing-experiment-dir "{output_dir}" --official-config-copy "{DEFAULT_LOGREG_CONFIG_PATH}"',
            "",
        ]
    )
    commands_path = integration_dir / "commands.txt"
    commands_path.write_text(commands_text, encoding="utf-8")

    integration_summary = {
        "generated_at": datetime.now().isoformat(),
        "source_experiment_dir": str(output_dir),
        "selected_candidate_name": str(selected["candidate_name"]),
        "selected_model_family": str(selected["model_family"]),
        "selected_feature_set_name": str(selected["feature_set_name"]),
        "selected_feature_count": int(len(features)),
        "selected_holdout_roc_auc": float(selected["holdout_roc_auc"]),
        "selected_holdout_accuracy": float(selected["holdout_accuracy"]),
        "selection_rule": selection_metadata["selection_rule"],
        "best_overall_candidate_name": selection_metadata["best_overall_candidate_name"],
        "best_overall_model_family": selection_metadata["best_overall_model_family"],
        "best_overall_holdout_roc_auc": selection_metadata["best_overall_holdout_roc_auc"],
        "selected_matches_best_overall": bool(selection_metadata["selected_matches_best_overall"]),
        "exported_config_path": str(exported_config_path),
        "official_config_copy_path": copied_config_path,
        "commands_path": str(commands_path),
        "model_params": model_params,
    }
    integration_summary.update(summarize_threshold_recommendations_v6_1(artifacts["threshold_best"], candidate_name=str(selected["candidate_name"])))
    dump_json(integration_summary, integration_dir / "integration_summary.json")

    if log_path is not None:
        log(
            f"[finalize] config oficial exportada para {exported_config_path}",
            log_path,
        )

    return integration_summary


def save_bar_figure_from_frame(
    frame: pd.DataFrame,
    *,
    label_column: str,
    value_column: str,
    title: str,
    xlabel: str,
    output_path: Path,
    color: str,
) -> None:
    ordered = frame.sort_values(value_column, ascending=True, kind="stable")
    fig, ax = dissertation_base_figure(9.5, max(4.8, 0.55 * len(ordered) + 1.8))
    ax.barh(ordered[label_column], ordered[value_column], color=color)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for index, value in enumerate(ordered[value_column].tolist()):
        ax.text(float(value) + 0.002, index, f"{float(value):.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def build_dissertation_assets_from_v6(
    output_dir: Path,
    *,
    official_candidate_name: str,
    log_path: Path | None,
) -> dict[str, object]:
    artifacts = load_v6_final_artifacts(output_dir)
    comparison = artifacts["comparison"].copy()
    feature_manifest = artifacts["feature_manifest"].copy()
    model_manifest = artifacts["model_manifest"].copy()
    holdout_tuned_leaderboard = artifacts["holdout_tuned_leaderboard"].copy()
    dataset_path = Path(str(artifacts["audit"]["dataset_path"]))

    assets_dir = output_dir / "dissertation_assets"
    tables_dir = assets_dir / "tables"
    figures_dir = assets_dir / "figures"
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)

    comparison["display_label"] = comparison.apply(
        lambda row: f"{int(row['final_rank'])}. {row['model_family']} | {row['feature_set_name']}",
        axis=1,
    )

    leaderboard_table = comparison[
        [
            "final_rank",
            "candidate_name",
            "model_family",
            "feature_set_name",
            "holdout_roc_auc",
            "holdout_accuracy",
            "holdout_log_loss",
            "holdout_brier",
        ]
    ].rename(
        columns={
            "final_rank": "Rank",
            "candidate_name": "Candidato",
            "model_family": "Familia do modelo",
            "feature_set_name": "Conjunto de features",
            "holdout_roc_auc": "ROC-AUC holdout",
            "holdout_accuracy": "Accuracy holdout",
            "holdout_log_loss": "Log-loss holdout",
            "holdout_brier": "Brier holdout",
        }
    )
    write_csv(leaderboard_table, tables_dir / "v6_final_leaderboard.csv")
    save_latex_table(
        leaderboard_table,
        tables_dir / "v6_final_leaderboard.tex",
        "Leaderboard final da bateria experimental no holdout confirmatorio.",
        "tab:v61_final_leaderboard",
        column_format="rlllrrrr",
    )

    stability_table = comparison[
        [
            "final_rank",
            "candidate_name",
            "repeated_cv_roc_auc_mean",
            "repeated_cv_roc_auc_std",
            "repeated_cv_accuracy_mean",
            "repeated_cv_seed_roc_auc_std_mean",
            "holdout_roc_auc",
        ]
    ].rename(
        columns={
            "final_rank": "Rank",
            "candidate_name": "Candidato",
            "repeated_cv_roc_auc_mean": "ROC-AUC CV repetida",
            "repeated_cv_roc_auc_std": "Desvio ROC-AUC CV",
            "repeated_cv_accuracy_mean": "Accuracy CV repetida",
            "repeated_cv_seed_roc_auc_std_mean": "Desvio entre seeds",
            "holdout_roc_auc": "ROC-AUC holdout",
        }
    )
    write_csv(stability_table, tables_dir / "v6_final_stability.csv")
    save_latex_table(
        stability_table,
        tables_dir / "v6_final_stability.tex",
        "Estabilidade dos finalistas da bateria experimental.",
        "tab:v61_final_stability",
        column_format="rllllll",
    )

    if not holdout_tuned_leaderboard.empty:
        tuned_table = holdout_tuned_leaderboard[
            [
                "tuned_rank",
                "candidate_name",
                "model_family",
                "feature_set_name",
                "threshold",
                "accuracy",
                "roc_auc",
                "f1",
            ]
        ].rename(
            columns={
                "tuned_rank": "Rank",
                "candidate_name": "Candidato",
                "model_family": "Familia do modelo",
                "feature_set_name": "Conjunto de features",
                "threshold": "Threshold",
                "accuracy": "Accuracy ajustada",
                "roc_auc": "ROC-AUC holdout",
                "f1": "F1 ajustado",
            }
        )
        write_csv(tuned_table, tables_dir / "v6_tuned_accuracy_leaderboard.csv")
        save_latex_table(
            tuned_table,
            tables_dir / "v6_tuned_accuracy_leaderboard.tex",
            "Leaderboard dos finalistas apos ajuste de threshold com base na CV repetida.",
            "tab:v61_tuned_accuracy_leaderboard",
            column_format="rlllrrrr",
        )

    selected_row = comparison.loc[comparison["candidate_name"] == official_candidate_name]
    if selected_row.empty:
        raise ValueError(f"Nao foi possivel localizar o candidato oficial {official_candidate_name}.")
    selected_row = selected_row.iloc[0]
    selected_feature_row = feature_manifest.loc[feature_manifest["candidate_name"] == official_candidate_name]
    selected_model_row = model_manifest.loc[model_manifest["model_name"] == str(selected_row["model_name"])]
    if selected_feature_row.empty or selected_model_row.empty:
        raise ValueError("Os artefatos do candidato oficial estao incompletos.")

    selected_features = parse_feature_list(selected_feature_row.iloc[0]["features"])
    selected_model_params = parse_params_json(selected_model_row.iloc[0]["params_json"])

    selected_overview = pd.DataFrame(
        [
            {
                "Candidato": official_candidate_name,
                "Familia do modelo": str(selected_row["model_family"]),
                "Conjunto de features": str(selected_row["feature_set_name"]),
                "Quantidade de features": int(len(selected_features)),
                "ROC-AUC holdout": float(selected_row["holdout_roc_auc"]),
                "Accuracy holdout": float(selected_row["holdout_accuracy"]),
                "Threshold recomendado": selected_row.get("holdout_tuned_threshold"),
                "Accuracy ajustada": selected_row.get("holdout_tuned_accuracy"),
                "Ganho de accuracy": selected_row.get("holdout_tuned_accuracy_gain"),
                "ROC-AUC CV repetida": float(selected_row["repeated_cv_roc_auc_mean"]),
                "Desvio ROC-AUC CV": float(selected_row["repeated_cv_roc_auc_std"]),
                "C": float(selected_model_params.get("C", 1.0)),
                "Class weight": selected_model_params.get("class_weight"),
                "l1_ratio": selected_model_params.get("l1_ratio"),
            }
        ]
    )
    write_csv(selected_overview, tables_dir / "v6_modelo_oficial_escolhido.csv")
    save_latex_table(
        selected_overview,
        tables_dir / "v6_modelo_oficial_escolhido.tex",
        "Resumo do candidato logistico escolhido para integracao ao fluxo oficial.",
        "tab:v61_modelo_oficial",
        column_format="llllllllllllll",
    )

    selected_features_table = pd.DataFrame(
        {
            "Posicao": np.arange(1, len(selected_features) + 1),
            "Feature": selected_features,
        }
    )
    write_csv(selected_features_table, tables_dir / "v6_modelo_oficial_features.csv")
    save_latex_table(
        selected_features_table,
        tables_dir / "v6_modelo_oficial_features.tex",
        "Features do candidato logistico exportado para o fluxo oficial.",
        "tab:v61_modelo_oficial_features",
        column_format="rl",
    )

    save_bar_figure_from_frame(
        comparison[["display_label", "holdout_roc_auc"]].copy(),
        label_column="display_label",
        value_column="holdout_roc_auc",
        title="ROC-AUC no holdout dos finalistas",
        xlabel="ROC-AUC",
        output_path=figures_dir / "v6_finalistas_holdout_roc_auc.pdf",
        color="#1565c0",
    )
    save_bar_figure_from_frame(
        comparison[["display_label", "holdout_accuracy"]].copy(),
        label_column="display_label",
        value_column="holdout_accuracy",
        title="Accuracy no holdout dos finalistas",
        xlabel="Accuracy",
        output_path=figures_dir / "v6_finalistas_holdout_accuracy.pdf",
        color="#00897b",
    )

    if str(selected_row["model_family"]) in {"logreg_l1", "logreg_l2", "logreg_elasticnet"}:
        dataset = load_dataset(dataset_path)
        train_df, _ = split_train_holdout(dataset)
        fit_model = build_pipeline(
            str(selected_row["model_family"]),
            selected_model_params,
            random_state=DEFAULT_RANDOM_STATE,
        )
        fit_model.fit(train_df[selected_features], train_df["win_target"].astype(int))
        coefficients = pd.DataFrame(
            {
                "feature": selected_features,
                "coefficient": fit_model.named_steps["clf"].coef_.ravel(),
            }
        ).sort_values("coefficient", ascending=False, kind="stable")
        write_csv(coefficients, tables_dir / "v6_modelo_oficial_coeficientes.csv")
        save_latex_table(
            coefficients.rename(columns={"feature": "Feature", "coefficient": "Coeficiente"}),
            tables_dir / "v6_modelo_oficial_coeficientes.tex",
            "Coeficientes do modelo logistico escolhido, ajustado sobre o treino completo.",
            "tab:v61_modelo_oficial_coeficientes",
            column_format="lr",
        )

        coefficient_plot = coefficients.copy()
        coefficient_plot["abs_coefficient"] = coefficient_plot["coefficient"].abs()
        coefficient_plot = coefficient_plot.sort_values("abs_coefficient", ascending=False, kind="stable").head(20)
        coefficient_plot = coefficient_plot.sort_values("coefficient", ascending=True, kind="stable")
        fig, ax = dissertation_base_figure(9.5, max(5.0, 0.45 * len(coefficient_plot) + 1.8))
        colors = ["#c62828" if value < 0 else "#2e7d32" for value in coefficient_plot["coefficient"]]
        ax.barh(coefficient_plot["feature"], coefficient_plot["coefficient"], color=colors)
        ax.set_xlabel("Coeficiente padronizado")
        ax.set_title("Coeficientes do modelo logistico exportado")
        fig.tight_layout()
        fig.savefig(figures_dir / "v6_modelo_oficial_coeficientes.pdf")
        plt.close(fig)

    assets_summary = {
        "generated_at": datetime.now().isoformat(),
        "assets_dir": str(assets_dir),
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
        "official_candidate_name": official_candidate_name,
    }
    dump_json(assets_summary, assets_dir / "assets_summary.json")

    if log_path is not None:
        log(f"[finalize] assets da dissertacao gerados em {assets_dir}", log_path)

    return assets_summary


def finalize_v6_outputs(
    output_dir: Path,
    *,
    official_config_copy: Path | None,
    log_path: Path | None,
) -> dict[str, object]:
    integration_summary = export_official_integration_from_v6(
        output_dir,
        official_config_copy=official_config_copy,
        log_path=log_path,
    )
    assets_summary = build_dissertation_assets_from_v6(
        output_dir,
        official_candidate_name=str(integration_summary["selected_candidate_name"]),
        log_path=log_path,
    )
    return {
        "integration_summary": integration_summary,
        "assets_summary": assets_summary,
    }


build_threshold_tuning_tables = build_threshold_tuning_tables_v6_1
summarize_threshold_recommendations = summarize_threshold_recommendations_v6_1
estimate_experiment_volume = estimate_v6_volume
build_pipeline_audit = build_pipeline_audit_v6
prepare_exploratory_folds = prepare_exploratory_folds_v6
freeze_shortlist = freeze_shortlist_v6
run_confirmatory_phase = run_confirmatory_phase_v6
export_official_integration = export_official_integration_from_v6
build_dissertation_assets = build_dissertation_assets_from_v6
finalize_outputs = finalize_v6_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Bateria experimental principal, cega e defensavel para o TCC.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--stochastic-seeds", type=str, default="42,52,62,72,82")
    parser.add_argument("--shortlist-size", type=int, default=DEFAULT_SHORTLIST_SIZE)
    parser.add_argument("--shortlist-per-model-family", type=int, default=DEFAULT_SHORTLIST_PER_MODEL_FAMILY)
    parser.add_argument("--repeated-cv-splits", type=int, default=DEFAULT_REPEATED_CV_SPLITS)
    parser.add_argument("--repeated-cv-repeats", type=int, default=DEFAULT_REPEATED_CV_REPEATS)
    parser.add_argument("--bootstrap-runs", type=int, default=DEFAULT_BOOTSTRAP_RUNS)
    parser.add_argument("--threshold-objectives", type=str, default="accuracy,f1")
    parser.add_argument("--feature-search-profile", type=str, choices=FEATURE_SEARCH_PROFILES, default=DEFAULT_FEATURE_SEARCH_PROFILE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Nao roda a bateria; apenas exporta a integracao oficial e os assets finais a partir de uma pasta experimental ja concluida.",
    )
    parser.add_argument(
        "--existing-experiment-dir",
        type=Path,
        default=None,
        help="Pasta de uma execucao experimental ja concluida para usar com --finalize-only.",
    )
    parser.add_argument(
        "--official-config-copy",
        type=Path,
        default=None,
        help="Copia opcional da configuracao exportada. Use para espelhar o JSON no caminho oficial.",
    )
    args = parser.parse_args()

    if args.finalize_only:
        if args.existing_experiment_dir is None:
            raise SystemExit("Informe --existing-experiment-dir ao usar --finalize-only.")
        existing_dir = args.existing_experiment_dir.resolve()
        log_path = existing_dir / "run.log"
        finalization = finalize_v6_outputs(
            existing_dir,
            official_config_copy=args.official_config_copy.resolve() if args.official_config_copy else None,
            log_path=log_path,
        )
        print(f"Config exportada em: {finalization['integration_summary']['exported_config_path']}")
        print(f"Assets finais em: {finalization['assets_summary']['assets_dir']}")
        return

    smoke_test = bool(args.smoke_test)
    search_profile = str(args.feature_search_profile)
    stochastic_seeds = parse_csv_ints(args.stochastic_seeds) or list(DEFAULT_STOCHASTIC_SEEDS_V6)
    threshold_objectives = [item.strip() for item in str(args.threshold_objectives).split(",") if item.strip()] or list(DEFAULT_THRESHOLD_OBJECTIVES_V6_1)
    if smoke_test:
        stochastic_seeds = stochastic_seeds[:1] or [DEFAULT_RANDOM_STATE]
        threshold_objectives = threshold_objectives[:1] or ["accuracy"]
    effective_bootstrap_runs = 40 if smoke_test else args.bootstrap_runs

    default_name = f"dissertation_v6_1_{search_profile}_smoke" if smoke_test else f"dissertation_v6_1_{search_profile}_{timestamp_now()}"
    output_dir = resolve_output_dir(args.reports_root, args.experiment_name or default_name)
    ensure_dir(output_dir)
    log_path = output_dir / "run.log"

    dataset = load_dataset(args.data_path)
    all_feature_columns = resolve_feature_columns(dataset, args.data_path)
    feature_columns = [
        column
        for column in all_feature_columns
        if not (
            column.startswith("interaction_")
            or column.startswith("delta_")
            or column.startswith("ratio_")
            or column.startswith("abs_")
        )
    ]

    train_df, holdout_df = split_train_holdout(dataset)
    temporal_folds = build_temporal_folds(train_df)
    feature_specs = build_feature_set_specs_v6(feature_columns, smoke_test=smoke_test, search_profile=search_profile)
    model_specs = build_model_specs_v6(smoke_test=smoke_test)
    candidates = build_candidate_specs_v6(feature_specs, model_specs)
    feature_lookup = {spec.name: spec for spec in feature_specs}
    model_lookup = {spec.name: spec for spec in model_specs}

    audit = build_pipeline_audit_v6(
        data_path=args.data_path,
        feature_columns=feature_columns,
        train_df=train_df,
        holdout_df=holdout_df,
        feature_specs=feature_specs,
        search_profile=search_profile,
    )
    dump_json(audit, output_dir / "pipeline_audit.json")

    write_csv(
        pd.DataFrame(
            [{"feature_set_name": spec.name, "family": spec.family, "params_json": json.dumps(params_to_dict(spec.params), ensure_ascii=False)} for spec in feature_specs]
        ),
        output_dir / "feature_set_spec_manifest.csv",
    )
    write_csv(
        pd.DataFrame(
            [{"model_name": spec.name, "family": spec.family, "params_json": json.dumps(params_to_dict(spec.params), ensure_ascii=False)} for spec in model_specs]
        ),
        output_dir / "model_spec_manifest.csv",
    )
    write_csv(
        candidate_manifest_frame(candidates=candidates, feature_lookup=feature_lookup, model_lookup=model_lookup, stochastic_seeds=stochastic_seeds),
        output_dir / "candidate_manifest.csv",
    )

    volume = estimate_v6_volume(
        feature_columns=feature_columns,
        feature_specs=feature_specs,
        model_specs=model_specs,
        train_df=train_df,
        stochastic_seeds=stochastic_seeds,
        shortlist_size=args.shortlist_size,
        repeated_cv_splits=args.repeated_cv_splits,
        repeated_cv_repeats=args.repeated_cv_repeats,
        bootstrap_runs=effective_bootstrap_runs,
        smoke_test=smoke_test,
    )
    dump_json(
        {
            "generated_at": datetime.now().isoformat(),
            "runner": "run_experimental_battery.py",
            "smoke_test": smoke_test,
            "workers": int(args.workers),
            "batch_size": int(args.batch_size),
            "random_state": int(args.random_state),
            "stochastic_seeds": [int(seed) for seed in stochastic_seeds],
            "shortlist_size": int(args.shortlist_size),
            "shortlist_per_model_family": int(args.shortlist_per_model_family),
            "repeated_cv_splits": int(args.repeated_cv_splits),
            "repeated_cv_repeats": int(args.repeated_cv_repeats),
            "bootstrap_runs": int(effective_bootstrap_runs),
            "threshold_objectives": threshold_objectives,
            "feature_search_profile": search_profile,
            "base_feature_only": True,
            "base_feature_count": int(len(feature_columns)),
            "predefined_reference_used": False,
            "volume": volume,
        },
        output_dir / "experiment_config.json",
    )

    log(
        f"[setup] profile={search_profile} features_base={len(feature_columns)} feature_sets={len(feature_specs)} models={len(model_specs)} candidates={len(candidates)} workers={args.workers}",
        log_path,
    )
    log(
        f"[volume] ranking_fits={volume['estimated_feature_ranking_fits']} exploratory_fits={volume['estimated_exploratory_fits']} total_fits~={volume['estimated_total_model_fits']} threshold_scans~={volume['estimated_threshold_grid_scans']}",
        log_path,
    )

    fold_payloads = prepare_exploratory_folds_v6(
        train_df=train_df,
        temporal_folds=temporal_folds,
        feature_columns=feature_columns,
        feature_specs=feature_specs,
        random_state=args.random_state,
        output_dir=output_dir / "feature_rankings",
        log_path=log_path,
        smoke_test=smoke_test,
        search_profile=search_profile,
    )

    exploratory_leaderboard = run_exploratory_search(
        output_dir=output_dir,
        candidates=candidates,
        feature_lookup=feature_lookup,
        model_lookup=model_lookup,
        fold_payloads=fold_payloads,
        workers=args.workers,
        batch_size=8 if smoke_test else args.batch_size,
        random_state=args.random_state,
        stochastic_seeds=stochastic_seeds,
        resume=args.resume,
        log_path=log_path,
        stage_dirname="exploratory",
        stage_label=f"exploratory-{search_profile}",
    )

    shortlist = freeze_shortlist_v6(
        exploratory_leaderboard,
        shortlist_size=4 if smoke_test else args.shortlist_size,
        per_model_family=1 if smoke_test else args.shortlist_per_model_family,
        output_path=output_dir / "shortlist.csv",
    )
    log(f"[shortlist] finalistas congelados={len(shortlist)}", log_path)

    confirmatory = run_confirmatory_phase_v6(
        output_dir=output_dir,
        train_df=train_df,
        holdout_df=holdout_df,
        shortlist=shortlist,
        feature_columns=feature_columns,
        feature_specs=feature_specs,
        model_lookup=model_lookup,
        random_state=args.random_state,
        stochastic_seeds=stochastic_seeds,
        repeated_cv_splits=3 if smoke_test else args.repeated_cv_splits,
        repeated_cv_repeats=2 if smoke_test else args.repeated_cv_repeats,
        bootstrap_runs=effective_bootstrap_runs,
        threshold_objectives=threshold_objectives,
        smoke_test=smoke_test,
        search_profile=search_profile,
        log_path=log_path,
    )

    summary = {
        "generated_at": datetime.now().isoformat(),
        "runner": "run_experimental_battery.py",
        "output_dir": str(output_dir),
        "feature_search_profile": search_profile,
        "base_feature_count": int(len(feature_columns)),
        "feature_set_count": int(len(feature_specs)),
        "model_count": int(len(model_specs)),
        "candidate_count": int(len(candidates)),
        "shortlist_count": int(len(shortlist)),
        "predefined_reference_used": False,
        "threshold_objectives": threshold_objectives,
        "best_exploratory_candidate": exploratory_leaderboard.iloc[0]["candidate_name"] if not exploratory_leaderboard.empty else "",
        "best_holdout_candidate": confirmatory["holdout_leaderboard"].iloc[0]["candidate_name"] if not confirmatory["holdout_leaderboard"].empty else "",
        "best_tuned_threshold_candidate": confirmatory["holdout_tuned_leaderboard"].iloc[0]["candidate_name"] if not confirmatory["holdout_tuned_leaderboard"].empty else "",
    }
    finalization = finalize_v6_outputs(
        output_dir,
        official_config_copy=args.official_config_copy.resolve() if args.official_config_copy else None,
        log_path=log_path,
    )
    summary.update(
        {
            "official_selected_candidate": finalization["integration_summary"]["selected_candidate_name"],
            "official_selected_model_family": finalization["integration_summary"]["selected_model_family"],
            "official_selected_config_path": finalization["integration_summary"]["exported_config_path"],
            "dissertation_assets_dir": finalization["assets_summary"]["assets_dir"],
        }
    )
    dump_json(summary, output_dir / "summary.json")


if __name__ == "__main__":
    main()
