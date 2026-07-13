#!/usr/bin/env python3
"""Bateria experimental final para fechamento do projeto.

Objetivo:
- manter a busca cega por variaveis;
- permitir tanto uma trilha compacta quanto uma trilha realmente cega e mais ampla;
- preservar comparacao entre regressao logistica, Naive Bayes, KNN, SVM e Random Forest;
- gerar uma trilha reproduzivel e explicavel.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.experimental_battery_support import (
    build_feature_set_specs_v6,
    build_model_specs_v6,
    build_pipeline_audit,
    dedup_feature_specs,
    estimate_experiment_volume,
    finalize_outputs,
    freeze_feature_space_v6,
    freeze_shortlist,
    prepare_exploratory_folds,
    run_confirmatory_phase,
)
from src.experimental_battery_core import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_REPORTS_ROOT,
    DEFAULT_WORKERS,
    CandidateSpec,
    FeatureSetSpec,
    ModelSpec,
    build_temporal_folds,
    candidate_manifest_frame,
    dump_json,
    ensure_dir,
    kv_pairs,
    load_dataset,
    log,
    params_to_dict,
    parse_csv_ints,
    resolve_feature_columns,
    resolve_output_dir,
    run_exploratory_search,
    season_sort_key,
    sort_leaderboard,
    split_train_holdout,
    timestamp_now,
    write_csv,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_DATA_PATH = ROOT / "data" / "processed" / "match_feature_differences.csv"
DEFAULT_BASE_DATASET_METADATA_PATH = ROOT / "data" / "processed" / "dataset_metadata.json"
DEFAULT_COMPACT_DATA_PATH = ROOT / "data" / "processed" / "match_feature_differences_temporal22.csv"
DEFAULT_COMPACT_DATASET_METADATA_PATH = ROOT / "data" / "processed" / "match_feature_differences_temporal22_metadata.json"
DEFAULT_BLIND_DATA_PATH = ROOT / "data" / "processed" / "match_feature_differences_blind49.csv"
DEFAULT_BLIND_DATASET_METADATA_PATH = ROOT / "data" / "processed" / "match_feature_differences_blind49_metadata.json"
DEFAULT_STOCHASTIC_SEEDS = [42, 52, 62, 72, 82]
DEFAULT_BOOTSTRAP_RUNS = 200
DEFAULT_SHORTLIST_SIZE = 12
DEFAULT_SHORTLIST_PER_MODEL_FAMILY = 1
DEFAULT_REPEATED_CV_SPLITS = 5
DEFAULT_REPEATED_CV_REPEATS = 2
DEFAULT_BATCH_SIZE = 24
DEFAULT_THRESHOLD_OBJECTIVES = ["accuracy", "f1"]
TARGETED_L2_FEATURE_SETS = (
    "univariate_top_11",
    "univariate_top_14",
    "univariate_top_17",
    "univariate_top_20",
    "consensus_top_11",
    "block_budget_compact_11",
    "block_budget_compact_11_conservative",
    "block_budget_compact_11_univariate",
)
TARGETED_L2_PER_FEATURE_SET = 2
SEARCH_MODES = ("focused", "extended")
LOCAL_REFINE_TOP_BASE_SUBSETS = 4
LOCAL_REFINE_TOP_LOGISTIC_MODELS = 6
LOCAL_REFINE_MAX_SUBSET_SIZE = 20


def resolve_representation_spec(representation: str) -> dict[str, object]:
    normalized = str(representation).strip().lower()
    specs = {
        "compact22": {
            "representation_name": "compact22_temporal",
            "data_path": DEFAULT_COMPACT_DATA_PATH,
            "metadata_path": DEFAULT_COMPACT_DATASET_METADATA_PATH,
            "description": "11 variaveis classicas + 11 temporais. Busca cega dentro do conjunto compacto.",
        },
        "blind49": {
            "representation_name": "blind49_classic_plus_temporal",
            "data_path": DEFAULT_BLIND_DATA_PATH,
            "metadata_path": DEFAULT_BLIND_DATASET_METADATA_PATH,
            "description": "38 variaveis classicas do dataset base + 11 variaveis temporais exclusivas, totalizando 49 features de busca.",
        },
    }
    if normalized not in specs:
        raise ValueError(f"Representacao invalida: {representation}")
    return specs[normalized]


def ensure_source_datasets() -> None:
    required_paths = [
        DEFAULT_BASE_DATA_PATH,
        DEFAULT_BASE_DATASET_METADATA_PATH,
        DEFAULT_COMPACT_DATA_PATH,
        DEFAULT_COMPACT_DATASET_METADATA_PATH,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Arquivos base do projeto nao foram encontrados. Rode build_datasets.py antes da bateria experimental. "
            f"Ausentes: {missing}"
        )


def write_dataset_metadata(data_path: Path, metadata_path: Path, status: str, source_paths: list[str]) -> None:
    source_df = pd.read_csv(data_path)
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "status": status,
        "target_path": str(data_path),
        "source_paths": source_paths,
        "row_count": int(len(source_df)),
        "feature_columns": [
            column
            for column in source_df.columns
            if column
            not in {
                "actual_match_id",
                "match_date",
                "season_label",
                "season_usage",
                "team_name",
                "opponent_name",
                "win_target",
            }
        ],
    }
    dump_json(metadata, metadata_path)


def materialize_blind49_dataset() -> None:
    ensure_source_datasets()
    base_df = pd.read_csv(DEFAULT_BASE_DATA_PATH)
    compact_df = pd.read_csv(DEFAULT_COMPACT_DATA_PATH)
    base_metadata = json.loads(DEFAULT_BASE_DATASET_METADATA_PATH.read_text(encoding="utf-8"))
    compact_metadata = json.loads(DEFAULT_COMPACT_DATASET_METADATA_PATH.read_text(encoding="utf-8"))

    base_feature_columns = [str(item).strip() for item in base_metadata.get("feature_columns", []) if str(item).strip()]
    compact_feature_columns = [str(item).strip() for item in compact_metadata.get("feature_columns", []) if str(item).strip()]
    temporal_only_columns = [column for column in compact_feature_columns if column.startswith("diff_temporal_") and column not in base_df.columns]
    if not temporal_only_columns:
        raise ValueError("Nao foi possivel identificar colunas temporais exclusivas para montar o dataset blind49.")

    blind_df = base_df.merge(
        compact_df[["actual_match_id", *temporal_only_columns]],
        on="actual_match_id",
        how="left",
        validate="one_to_one",
    )
    blind_df.to_csv(DEFAULT_BLIND_DATA_PATH, index=False)
    write_dataset_metadata(
        data_path=DEFAULT_BLIND_DATA_PATH,
        metadata_path=DEFAULT_BLIND_DATASET_METADATA_PATH,
        status="materialized_blind49_dataset",
        source_paths=[str(DEFAULT_BASE_DATA_PATH), str(DEFAULT_COMPACT_DATA_PATH)],
    )


def ensure_representation_dataset(representation_spec: dict[str, object]) -> None:
    data_path = Path(representation_spec["data_path"])
    metadata_path = Path(representation_spec["metadata_path"])
    representation_name = str(representation_spec["representation_name"])
    if representation_name == "blind49_classic_plus_temporal":
        materialize_blind49_dataset()
        return
    if not data_path.exists():
        raise FileNotFoundError(
            "O dataset compacto ainda nao existe em "
            f"{data_path}. Rode build_datasets.py antes da bateria experimental."
        )
    if not metadata_path.exists():
        write_dataset_metadata(
            data_path=data_path,
            metadata_path=metadata_path,
            status="materialized_compact22_dataset",
            source_paths=[str(data_path)],
        )


def build_feature_set_specs(
    feature_columns: list[str],
    *,
    smoke_test: bool,
    search_mode: str,
) -> list[FeatureSetSpec]:
    if search_mode == "extended":
        return build_feature_set_specs_v6(feature_columns, smoke_test=smoke_test, search_profile="balanced")
    if search_mode != "focused":
        raise ValueError(f"Modo de busca invalido: {search_mode}")

    feature_count = len(feature_columns)

    def capped(value: int) -> int:
        return min(feature_count, int(value))

    k5 = capped(5)
    k8 = capped(8)
    k11 = capped(11)

    specs: list[FeatureSetSpec] = [
        FeatureSetSpec(name="all_snapshot_features", family="all_features", params=kv_pairs()),
        FeatureSetSpec(name=f"univariate_top_{k5}", family="univariate_top_k", params=kv_pairs(k=k5)),
        FeatureSetSpec(name=f"univariate_top_{k8}", family="univariate_top_k", params=kv_pairs(k=k8)),
        FeatureSetSpec(name=f"univariate_top_{k11}", family="univariate_top_k", params=kv_pairs(k=k11)),
        FeatureSetSpec(name=f"l1_top_{k5}", family="l1_top_k", params=kv_pairs(k=k5)),
        FeatureSetSpec(name=f"l1_top_{k8}", family="l1_top_k", params=kv_pairs(k=k8)),
        FeatureSetSpec(name=f"l1_top_{k11}", family="l1_top_k", params=kv_pairs(k=k11)),
        FeatureSetSpec(name=f"tree_top_{k5}", family="tree_top_k", params=kv_pairs(k=k5)),
        FeatureSetSpec(name=f"tree_top_{k8}", family="tree_top_k", params=kv_pairs(k=k8)),
        FeatureSetSpec(name=f"tree_top_{k11}", family="tree_top_k", params=kv_pairs(k=k11)),
        FeatureSetSpec(
            name=f"consensus_top_{k11}",
            family="rank_scheme_top_k",
            params=kv_pairs(k=k11, scheme="combined_rank_score"),
        ),
        FeatureSetSpec(
            name="block_budget_compact_11",
            family="block_budget_top_k",
            params=kv_pairs(
                k=k11,
                scheme="combined_rank_score",
                budget_recent_form=1,
                budget_temporal_form=3,
                budget_elo_context=1,
                budget_combat_core=1,
                budget_utility_vision=2,
                budget_entry_trade=3,
            ),
        ),
        FeatureSetSpec(
            name="block_budget_compact_11_conservative",
            family="block_budget_top_k",
            params=kv_pairs(
                k=k11,
                scheme="conservative_rank_score",
                budget_recent_form=1,
                budget_temporal_form=3,
                budget_elo_context=1,
                budget_combat_core=1,
                budget_utility_vision=2,
                budget_entry_trade=3,
            ),
        ),
        FeatureSetSpec(
            name="block_budget_compact_11_univariate",
            family="block_budget_top_k",
            params=kv_pairs(
                k=k11,
                scheme="univariate_tilt_rank_score",
                budget_recent_form=1,
                budget_temporal_form=3,
                budget_elo_context=1,
                budget_combat_core=1,
                budget_utility_vision=2,
                budget_entry_trade=3,
            ),
        ),
    ]

    if smoke_test:
        keep = {
            "all_snapshot_features",
            f"univariate_top_{k5}",
            f"univariate_top_{k11}",
            "block_budget_compact_11",
        }
        specs = [spec for spec in specs if spec.name in keep]
    return dedup_feature_specs(specs)


def build_model_specs(*, smoke_test: bool, search_mode: str) -> list[ModelSpec]:
    if search_mode == "extended":
        return build_model_specs_v6(smoke_test=smoke_test)
    if search_mode != "focused":
        raise ValueError(f"Modo de busca invalido: {search_mode}")

    if smoke_test:
        return [
            ModelSpec(name="dummy_prior", family="dummy_prior", params=kv_pairs()),
            ModelSpec(name="logreg_l2_c0.03_cwbalanced", family="logreg_l2", params=kv_pairs(C=0.03, class_weight="balanced")),
            ModelSpec(name="logreg_l1_c0.1_cwnone", family="logreg_l1", params=kv_pairs(C=0.1, class_weight=None)),
            ModelSpec(name="gaussian_nb_1e-8", family="gaussian_nb", params=kv_pairs(var_smoothing=1e-8)),
            ModelSpec(name="knn_k5_uniform", family="knn", params=kv_pairs(n_neighbors=5, weights="uniform")),
            ModelSpec(name="svm_linear_c0.1_cwnone", family="svm_linear", params=kv_pairs(C=0.1, class_weight=None)),
            ModelSpec(
                name="rf_n400_dnone_l2",
                family="random_forest",
                params=kv_pairs(n_estimators=400, max_depth=None, min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample"),
            ),
        ]

    specs: list[ModelSpec] = [ModelSpec(name="dummy_prior", family="dummy_prior", params=kv_pairs())]

    for c in [0.03, 0.10, 0.30, 1.0]:
        for class_weight in [None, "balanced"]:
            cw_name = "balanced" if class_weight else "none"
            specs.append(ModelSpec(name=f"logreg_l2_c{c}_cw{cw_name}", family="logreg_l2", params=kv_pairs(C=c, class_weight=class_weight)))

    for c in [0.10, 0.30]:
        for class_weight in [None, "balanced"]:
            cw_name = "balanced" if class_weight else "none"
            specs.append(ModelSpec(name=f"logreg_l1_c{c}_cw{cw_name}", family="logreg_l1", params=kv_pairs(C=c, class_weight=class_weight)))

    for c in [0.10, 0.30]:
        specs.append(
            ModelSpec(
                name=f"logreg_elastic_c{c}_r05",
                family="logreg_elasticnet",
                params=kv_pairs(C=c, class_weight="balanced", l1_ratio=0.5),
            )
        )

    for smoothing in [1e-8, 1e-6]:
        specs.append(ModelSpec(name=f"gaussian_nb_{smoothing:.0e}", family="gaussian_nb", params=kv_pairs(var_smoothing=smoothing)))

    for n_neighbors in [1, 5, 10]:
        specs.append(ModelSpec(name=f"knn_k{n_neighbors}_uniform", family="knn", params=kv_pairs(n_neighbors=n_neighbors, weights="uniform")))

    for c in [0.10, 1.0]:
        for class_weight in [None, "balanced"]:
            cw_name = "balanced" if class_weight else "none"
            specs.append(ModelSpec(name=f"svm_linear_c{c}_cw{cw_name}", family="svm_linear", params=kv_pairs(C=c, class_weight=class_weight)))

    for c in [0.30, 1.0]:
        specs.append(ModelSpec(name=f"svm_rbf_c{c}_scale", family="svm_rbf", params=kv_pairs(C=c, gamma="scale")))

    specs.extend(
        [
            ModelSpec(
                name="rf_n400_dnone_l2",
                family="random_forest",
                params=kv_pairs(n_estimators=400, max_depth=None, min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample"),
            ),
            ModelSpec(
                name="rf_n800_d16_l2",
                family="random_forest",
                params=kv_pairs(n_estimators=800, max_depth=16, min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample"),
            ),
        ]
    )
    return specs


def build_candidate_specs(feature_specs: list[FeatureSetSpec], model_specs: list[ModelSpec]) -> list[CandidateSpec]:
    return [
        CandidateSpec(name=f"{feature_spec.name}__{model_spec.name}", feature_set_name=feature_spec.name, model_name=model_spec.name)
        for feature_spec in feature_specs
        for model_spec in model_specs
    ]


def parse_feature_list(raw_value: object) -> list[str]:
    text = str(raw_value or "").strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def rank_ordered_features(
    features: Sequence[str],
    *,
    score_map: dict[str, float],
    freq_map: dict[str, int] | None = None,
) -> list[str]:
    freq_map = freq_map or {}
    return sorted(
        {str(feature).strip() for feature in features if str(feature).strip()},
        key=lambda feature: (
            -int(freq_map.get(feature, 0)),
            -float(score_map.get(feature, -1e12)),
            feature,
        ),
    )


def build_candidate_feature_frequency_lookup(
    *,
    fold_metrics_path: Path,
    candidate_names: Sequence[str],
) -> dict[str, Counter]:
    if not fold_metrics_path.exists() or not candidate_names:
        return {}
    fold_df = pd.read_csv(fold_metrics_path, usecols=["candidate_name", "outer_fold", "features"])
    fold_df = fold_df.loc[fold_df["candidate_name"].astype(str).isin({str(name) for name in candidate_names})].copy()
    if fold_df.empty:
        return {}
    counters: dict[str, Counter] = {}
    deduped = fold_df.drop_duplicates(subset=["candidate_name", "outer_fold"], keep="last")
    for row in deduped.to_dict(orient="records"):
        candidate_name = str(row["candidate_name"])
        counter = counters.setdefault(candidate_name, Counter())
        counter.update(parse_feature_list(row["features"]))
    return counters


def select_local_subset_refinement_model_specs(
    *,
    exploratory_leaderboard: pd.DataFrame,
    model_lookup: dict[str, ModelSpec],
    top_unique: int = LOCAL_REFINE_TOP_LOGISTIC_MODELS,
) -> list[ModelSpec]:
    if exploratory_leaderboard.empty:
        return []
    logistic_rows = sort_leaderboard(exploratory_leaderboard)
    logistic_rows = logistic_rows.loc[
        logistic_rows["model_family"].astype(str).isin({"logreg_l2", "logreg_l1", "logreg_elasticnet"})
    ].copy()
    if logistic_rows.empty:
        return []

    selected_names: list[str] = []
    seen: set[str] = set()

    def add_name(model_name: object) -> None:
        name = str(model_name)
        if name and name in model_lookup and name not in seen:
            selected_names.append(name)
            seen.add(name)

    for row in logistic_rows.to_dict(orient="records"):
        add_name(row["model_name"])
        if len(selected_names) >= max(1, int(top_unique)):
            break

    l2_rows = logistic_rows.loc[logistic_rows["model_family"].astype(str) == "logreg_l2"].copy()
    for row in l2_rows.head(4).to_dict(orient="records"):
        add_name(row["model_name"])

    return [model_lookup[name] for name in selected_names]


def build_local_subset_refinement_feature_specs(
    *,
    exploratory_leaderboard: pd.DataFrame,
    feature_lookup: dict[str, FeatureSetSpec],
    full_train_feature_space: dict[str, object],
    exploratory_fold_metrics_path: Path,
    top_base_subsets: int = LOCAL_REFINE_TOP_BASE_SUBSETS,
) -> list[FeatureSetSpec]:
    if exploratory_leaderboard.empty:
        return []

    combined_df = pd.DataFrame(full_train_feature_space["combined_df"]).copy()
    if combined_df.empty:
        return []
    combined_df["feature"] = combined_df["feature"].astype(str)
    score_map = dict(zip(combined_df["feature"], combined_df["combined_rank_score"]))
    l1_rate_map = dict(zip(combined_df["feature"], combined_df["l1_frequency_rate"]))
    ranked_all = combined_df.sort_values(
        ["combined_rank_score", "roc_auc_mean", "log_loss_mean", "feature"],
        ascending=[False, False, True, True],
        kind="stable",
    )["feature"].astype(str).tolist()

    logistic_rows = sort_leaderboard(exploratory_leaderboard)
    logistic_rows = logistic_rows.loc[
        logistic_rows["model_family"].astype(str).isin({"logreg_l2", "logreg_l1", "logreg_elasticnet"})
    ].copy()
    if logistic_rows.empty:
        return []

    selected_base_rows: list[dict[str, object]] = []
    seen_feature_sets: set[str] = set()
    for row in logistic_rows.to_dict(orient="records"):
        feature_set_name = str(row["feature_set_name"])
        if feature_set_name in seen_feature_sets:
            continue
        seen_feature_sets.add(feature_set_name)
        selected_base_rows.append(row)
        if len(selected_base_rows) >= max(1, int(top_base_subsets)):
            break

    fold_frequency_lookup = build_candidate_feature_frequency_lookup(
        fold_metrics_path=exploratory_fold_metrics_path,
        candidate_names=[str(row["candidate_name"]) for row in selected_base_rows],
    )

    specs: list[FeatureSetSpec] = []
    seen_signatures: set[tuple[str, ...]] = set()

    def add_spec(base_name: str, variant_name: str, features: Sequence[str]) -> None:
        ordered = rank_ordered_features(features, score_map=score_map)
        ordered = ordered[:LOCAL_REFINE_MAX_SUBSET_SIZE]
        if len(ordered) < 4:
            return
        signature = tuple(ordered)
        if signature in seen_signatures:
            return
        seen_signatures.add(signature)
        specs.append(
            FeatureSetSpec(
                name=f"local_refine__{base_name}__{variant_name}__k{len(ordered)}",
                family="explicit",
                params=kv_pairs(features=tuple(ordered)),
            )
        )

    for row in selected_base_rows:
        candidate_name = str(row["candidate_name"])
        base_name = str(row["feature_set_name"])
        fallback_features = parse_feature_list(row.get("resolved_features_last_fold", ""))
        freq_counter = fold_frequency_lookup.get(candidate_name, Counter())
        ordered_base = rank_ordered_features(
            freq_counter.keys() if freq_counter else fallback_features,
            score_map=score_map,
            freq_map=dict(freq_counter),
        )
        if not ordered_base:
            ordered_base = rank_ordered_features(fallback_features, score_map=score_map)
        if len(ordered_base) < 4:
            continue

        base_k = min(len(ordered_base), LOCAL_REFINE_MAX_SUBSET_SIZE)
        ordered_base = ordered_base[:base_k]
        reserves = [feature for feature in ranked_all if feature not in ordered_base][:8]
        weakest = ordered_base[-1]
        weakest_two = ordered_base[-2:] if len(ordered_base) >= 6 else ordered_base[-1:]
        stable_core = [feature for feature in ordered_base if int(freq_counter.get(feature, 0)) >= 2]
        if len(stable_core) < 4:
            stable_core = [feature for feature in ordered_base if float(l1_rate_map.get(feature, 0.0)) >= 0.50]

        add_spec(base_name, "trim1", ordered_base[:-1])
        if len(ordered_base) >= 6:
            add_spec(base_name, "trim2", ordered_base[:-2])
        if reserves and len(ordered_base) < LOCAL_REFINE_MAX_SUBSET_SIZE:
            add_spec(base_name, "plus1", [*ordered_base, reserves[0]])
        if len(reserves) >= 2 and len(ordered_base) + 1 < LOCAL_REFINE_MAX_SUBSET_SIZE:
            add_spec(base_name, "plus2", [*ordered_base, reserves[0], reserves[1]])
        if reserves:
            add_spec(base_name, "swap1", [feature for feature in ordered_base if feature != weakest] + [reserves[0]])
        if len(reserves) >= 2 and len(weakest_two) >= 2:
            add_spec(base_name, "swap2", [feature for feature in ordered_base if feature not in set(weakest_two)] + reserves[:2])
        if len(stable_core) >= 4:
            fill_candidates = [feature for feature in ranked_all if feature not in stable_core]
            fill_count = max(0, len(ordered_base) - len(stable_core))
            add_spec(base_name, "stablefill", [*stable_core, *fill_candidates[:fill_count]])
        repack_pool = rank_ordered_features([*ordered_base, *reserves[:4]], score_map=score_map)
        add_spec(base_name, "repack", repack_pool[:len(ordered_base)])

    return dedup_feature_specs(specs)


def attach_explicit_feature_specs_to_fold_payloads(
    *,
    fold_payloads: list[dict[str, object]],
    feature_specs: Sequence[FeatureSetSpec],
) -> None:
    for payload in fold_payloads:
        feature_sets = payload["feature_space"]["feature_sets"]
        for spec in feature_specs:
            params = params_to_dict(spec.params)
            feature_sets[spec.name] = [str(feature) for feature in params.get("features", []) if str(feature)]


def update_candidate_manifest(
    *,
    candidate_manifest_path: Path,
    candidates: list[CandidateSpec],
    feature_lookup: dict[str, FeatureSetSpec],
    model_lookup: dict[str, ModelSpec],
) -> None:
    new_manifest = candidate_manifest_frame(
        candidates=candidates,
        feature_lookup=feature_lookup,
        model_lookup=model_lookup,
        stochastic_seeds=[],
    )
    if candidate_manifest_path.exists():
        current = pd.read_csv(candidate_manifest_path)
        merged = pd.concat([current, new_manifest], ignore_index=True)
        merged = merged.drop_duplicates(subset=["candidate_name"], keep="last").reset_index(drop=True)
        write_csv(merged, candidate_manifest_path)
        return
    write_csv(new_manifest, candidate_manifest_path)


def run_local_subset_refinement_stage(
    *,
    output_dir: Path,
    train_df: pd.DataFrame,
    feature_columns: list[str],
    feature_specs: list[FeatureSetSpec],
    feature_lookup: dict[str, FeatureSetSpec],
    model_lookup: dict[str, ModelSpec],
    fold_payloads: list[dict[str, object]],
    exploratory_leaderboard: pd.DataFrame,
    random_state: int,
    workers: int,
    batch_size: int,
    stochastic_seeds: Sequence[int],
    smoke_test: bool,
    search_mode: str,
    log_path: Path,
) -> dict[str, object]:
    if exploratory_leaderboard.empty:
        return {
            "used": False,
            "feature_specs": feature_specs,
            "feature_lookup": feature_lookup,
            "leaderboard": exploratory_leaderboard,
            "refined_feature_set_count": 0,
            "refined_candidate_count": 0,
        }

    profile_name = "focused" if search_mode == "focused" else "balanced"
    refinement_dir = output_dir / "subset_refinement"
    ensure_dir(refinement_dir)

    full_train_feature_space = freeze_feature_space_v6(
        train_df,
        feature_columns=feature_columns,
        feature_specs=feature_specs,
        random_state=random_state + 7000,
        output_dir=refinement_dir / "full_train_rankings",
        smoke_test=smoke_test,
        search_profile=profile_name,
    )
    refined_feature_specs = build_local_subset_refinement_feature_specs(
        exploratory_leaderboard=exploratory_leaderboard,
        feature_lookup=feature_lookup,
        full_train_feature_space=full_train_feature_space,
        exploratory_fold_metrics_path=output_dir / "exploratory" / "exploratory_fold_metrics.csv",
        top_base_subsets=2 if smoke_test else LOCAL_REFINE_TOP_BASE_SUBSETS,
    )
    if not refined_feature_specs:
        log("[stage2-subset-refine] nenhum subset refinado novo foi gerado.", log_path)
        return {
            "used": False,
            "feature_specs": feature_specs,
            "feature_lookup": feature_lookup,
            "leaderboard": exploratory_leaderboard,
            "refined_feature_set_count": 0,
            "refined_candidate_count": 0,
        }

    selected_model_specs = select_local_subset_refinement_model_specs(
        exploratory_leaderboard=exploratory_leaderboard,
        model_lookup=model_lookup,
        top_unique=2 if smoke_test else LOCAL_REFINE_TOP_LOGISTIC_MODELS,
    )
    if not selected_model_specs:
        log("[stage2-subset-refine] nenhum modelo logistico elegivel foi selecionado para o refino local.", log_path)
        return {
            "used": False,
            "feature_specs": feature_specs,
            "feature_lookup": feature_lookup,
            "leaderboard": exploratory_leaderboard,
            "refined_feature_set_count": 0,
            "refined_candidate_count": 0,
        }

    attach_explicit_feature_specs_to_fold_payloads(fold_payloads=fold_payloads, feature_specs=refined_feature_specs)
    expanded_feature_specs = dedup_feature_specs([*feature_specs, *refined_feature_specs])
    expanded_feature_lookup = {spec.name: spec for spec in expanded_feature_specs}
    refined_candidates = build_candidate_specs(refined_feature_specs, selected_model_specs)

    write_csv(
        pd.DataFrame(
            [
                {
                    "feature_set_name": spec.name,
                    "family": spec.family,
                    "params_json": json.dumps(params_to_dict(spec.params), ensure_ascii=False),
                }
                for spec in refined_feature_specs
            ]
        ),
        refinement_dir / "refined_feature_set_spec_manifest.csv",
    )
    write_csv(
        pd.DataFrame(
            [
                {
                    "candidate_name": candidate.name,
                    "feature_set_name": candidate.feature_set_name,
                    "model_name": candidate.model_name,
                }
                for candidate in refined_candidates
            ]
        ),
        refinement_dir / "refined_candidate_manifest.csv",
    )
    update_candidate_manifest(
        candidate_manifest_path=output_dir / "candidate_manifest.csv",
        candidates=refined_candidates,
        feature_lookup=expanded_feature_lookup,
        model_lookup=model_lookup,
    )

    log(
        f"[stage2-subset-refine] subsets_refinados={len(refined_feature_specs)} candidatos_refinados={len(refined_candidates)}",
        log_path,
    )
    refined_leaderboard = run_exploratory_search(
        output_dir=output_dir,
        candidates=refined_candidates,
        feature_lookup=expanded_feature_lookup,
        model_lookup=model_lookup,
        fold_payloads=fold_payloads,
        workers=workers,
        batch_size=batch_size,
        random_state=random_state + 9000,
        stochastic_seeds=stochastic_seeds,
        resume=False,
        log_path=log_path,
        stage_dirname="exploratory/stage2_subset_refine",
        stage_label="stage2-subset-refine",
    )
    merged_leaderboard = sort_leaderboard(
        pd.concat([exploratory_leaderboard, refined_leaderboard], ignore_index=True)
        .drop_duplicates(subset=["candidate_name"], keep="last")
        .reset_index(drop=True)
    )
    write_csv(merged_leaderboard, output_dir / "exploratory" / "exploratory_leaderboard.csv")
    return {
        "used": True,
        "feature_specs": expanded_feature_specs,
        "feature_lookup": expanded_feature_lookup,
        "leaderboard": merged_leaderboard,
        "refined_feature_set_count": int(len(refined_feature_specs)),
        "refined_candidate_count": int(len(refined_candidates)),
    }


def freeze_shortlist(
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
        for feature_set_name in TARGETED_L2_FEATURE_SETS:
            l2_group = ranked.loc[
                (ranked["feature_set_name"].astype(str) == feature_set_name)
                & (ranked["model_family"].astype(str) == "logreg_l2")
            ].copy()
            if not l2_group.empty:
                add_names(l2_group.head(TARGETED_L2_PER_FEATURE_SET)["candidate_name"].astype(str).tolist())

    shortlist = ranked.loc[ranked["candidate_name"].astype(str).isin(selected_names)].copy()
    shortlist = sort_leaderboard(shortlist).reset_index(drop=True)
    shortlist.insert(0, "shortlist_rank", np.arange(1, len(shortlist) + 1))
    write_csv(shortlist, output_path)
    return shortlist


def build_pipeline_audit(
    *,
    data_path: Path,
    representation_name: str,
    representation_description: str,
    search_mode: str,
    feature_columns: list[str],
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    feature_specs: list[FeatureSetSpec],
) -> dict[str, object]:
    def estimated_feature_set_size(spec: FeatureSetSpec) -> int:
        params = params_to_dict(spec.params)
        if spec.family == "all_features":
            return len(feature_columns)
        if spec.family == "explicit":
            return len(params.get("features", []))
        if "k" in params:
            return int(params["k"])
        return min(11, len(feature_columns))

    return {
        "generated_at": datetime.now().isoformat(),
        "runner": str(ROOT / "run_experimental_battery.py"),
        "dataset_path": str(data_path),
        "dataset_feature_count": int(len(feature_columns)),
        "dataset_features": feature_columns,
        "dataset_policy": {
            "representation_name": representation_name,
            "representation_description": representation_description,
            "representation_source": str(data_path),
            "predefined_feature_subset_used": False,
            "blind_feature_search": True,
            "derived_interactions_used": False,
        },
        "feature_search_scope": {
            "search_mode": search_mode,
            "feature_set_count": int(len(feature_specs)),
            "families": sorted({spec.family for spec in feature_specs}),
            "compact_subset_focus": bool(search_mode == "focused"),
            "max_subset_size": int(max(estimated_feature_set_size(spec) for spec in feature_specs)),
        },
        "split_summary": {
            "train_rows": int(len(train_df)),
            "holdout_rows": int(len(holdout_df)),
            "train_seasons": sorted(train_df["season_label"].astype(str).unique().tolist(), key=season_sort_key),
            "holdout_seasons": sorted(holdout_df["season_label"].astype(str).unique().tolist(), key=season_sort_key),
            "holdout_date_min": str(holdout_df["match_date"].min().date()) if holdout_df["match_date"].notna().any() else "",
            "holdout_date_max": str(holdout_df["match_date"].max().date()) if holdout_df["match_date"].notna().any() else "",
        },
        "selection_policy": {
            "official_choice": "melhor regressao logistica por holdout_roc_auc, depois log_loss, brier e accuracy",
            "threshold_tuning": "apenas com previsoes internas da repeated CV do treino",
        },
    }


def write_texto_base(
    *,
    output_dir: Path,
    data_path: Path,
    representation_name: str,
    representation_description: str,
    search_mode: str,
    feature_columns: list[str],
    feature_specs: list[FeatureSetSpec],
    model_specs: list[ModelSpec],
    summary: dict[str, object],
    selected_config_path: Path,
) -> Path:
    selected_config = json.loads(selected_config_path.read_text(encoding="utf-8"))
    model_rows = []
    for spec in model_specs:
        params = params_to_dict(spec.params)
        model_rows.append(
            {
                "nome": spec.name,
                "familia": spec.family,
                "params": params,
            }
        )

    text_lines = [
        "# Bateria Experimental Final",
        "",
        "## O que esta bateria faz",
        "",
        "Esta bateria experimental final foi configurada para fazer uma busca cega por subconjuntos de variaveis e por familias de modelos.",
        f"Nesta execucao, a representacao usada foi `{representation_name}`.",
        f"O modo de busca usado foi `{search_mode}`.",
        representation_description,
        "Importante: ela nao carrega um conjunto fixo de 11 variaveis para forcar o resultado. A selecao nasce do ranking interno feito apenas no treino.",
        "",
        "## Dataset usado",
        "",
        f"- Arquivo base: `{data_path}`",
        f"- Quantidade de features base: `{len(feature_columns)}`",
        f"- Representacao: `{representation_name}`.",
        "",
        "## Como a busca cega funciona",
        "",
        "1. O holdout final fica separado por temporada e nao participa da busca principal.",
        "2. Dentro do treino, a bateria monta rankings de features por sinal univariado, estabilidade L1 e importancia por arvores.",
        "3. A partir desses rankings, ela monta conjuntos compactos de features, como top-5, top-8, top-11 e conjuntos por blocos tematicos.",
        "4. Cada conjunto e combinado com varios modelos candidatos.",
        "5. So depois a shortlist vai para repeated CV e, por fim, para o holdout final.",
        "",
        "## Conjuntos de features testados",
        "",
        "Os conjuntos testados nesta versao sao:",
    ]
    for spec in feature_specs:
        params = params_to_dict(spec.params)
        text_lines.append(f"- `{spec.name}`: familia `{spec.family}` com parametros `{json.dumps(params, ensure_ascii=False)}`.")

    text_lines.extend(
        [
            "",
            "## Modelos cobertos",
            "",
            "### Regressao logistica L2",
            "",
            "Modelo linear probabilistico com regularizacao L2. O parametro `C` controla a forca da regularizacao: quanto menor o `C`, maior o shrinkage e mais conservador fica o modelo.",
            "",
            "### Regressao logistica L1",
            "",
            "Parecida com a L2, mas a regularizacao L1 tende a zerar coeficientes e, por isso, ajuda a fazer selecao de variaveis.",
            "",
            "### Regressao logistica Elastic Net",
            "",
            "Mistura L1 e L2. Aqui ela entra como meio-termo entre compactacao e estabilidade.",
            "",
            "### Gaussian Naive Bayes",
            "",
            "Modelo probabilistico simples que assume independencia condicional entre as variaveis. E um baseline util porque e rapido e facil de comparar.",
            "",
            "### KNN",
            "",
            "O KNN compara uma partida com exemplos parecidos do treino. Nesta bateria ele testa `k=1`, `k=5` e `k=10`. Como e um metodo baseado em distancia, ele exige normalizacao z-score no pipeline.",
            "",
            "### SVM linear e SVM RBF",
            "",
            "A SVM procura uma fronteira de separacao forte entre as classes. A versao linear procura uma fronteira reta; a versao RBF permite curvaturas. Como tambem depende de escala, usa normalizacao z-score.",
            "",
            "### Random Forest",
            "",
            "Conjunto de varias arvores de decisao treinadas com aleatoriedade controlada. Ela entra como comparativo nao linear e ajuda a verificar se existem padroes que a regressao logistica nao captura.",
            "",
            "### Dummy baseline",
            "",
            "Baseline propositalmente simples para garantir que os modelos reais estejam aprendendo algo alem de uma previsao trivial.",
            "",
            "## Grade de modelos desta execucao",
            "",
        ]
    )
    for row in model_rows:
        text_lines.append(f"- `{row['nome']}`: familia `{row['familia']}`, parametros `{json.dumps(row['params'], ensure_ascii=False)}`.")

    text_lines.extend(
        [
            "",
            "## Protocolo de avaliacao",
            "",
            "- Holdout final temporal por temporada.",
            "- Busca principal feita apenas no treino.",
            "- Repeated CV sobre a shortlist para medir media e desvio-padrao.",
            "- Threshold tuning feito apenas com previsoes internas da repeated CV.",
            "",
            "## Resultado oficial desta execucao",
            "",
            f"- Candidato oficial exportado: `{selected_config.get('source_candidate_name', '')}`.",
            f"- ROC-AUC holdout: `{selected_config.get('holdout_roc_auc', '')}`.",
            f"- Accuracy holdout: `{selected_config.get('holdout_accuracy', '')}`.",
            f"- Features finais: `{selected_config.get('features', [])}`.",
            "",
            "## Leitura curta para a reuniao",
            "",
            "A mensagem principal desta bateria final e: fizemos uma busca cega por variaveis em um dataset supervisionado pre-jogo, comparamos familias de modelos pedidas pelo professor e correlatas, e escolhemos o modelo final so depois da comparacao sob um protocolo temporal mais rigoroso.",
        ]
    )

    output_path = output_dir / "texto_base_modelagem_ptbr.md"
    output_path.write_text("\n".join(text_lines), encoding="utf-8")
    return output_path


def write_post_finalization(
    *,
    output_dir: Path,
    finalization: dict[str, object],
) -> dict[str, object]:
    integration_summary = dict(finalization["integration_summary"])
    integration_dir = output_dir / "official_integration"
    exported_config_path = Path(integration_summary["exported_config_path"])
    renamed_config_path = integration_dir / "config_modelo_principal.json"
    if exported_config_path.exists() and exported_config_path.resolve() != renamed_config_path.resolve():
        shutil.copy2(exported_config_path, renamed_config_path)

    commands_text = "\n".join(
        [
            "Comandos principais da bateria experimental final:",
            f'python "{ROOT / "run_experimental_battery.py"}"',
            f'python "{ROOT / "train_model.py"}" --training-config "{renamed_config_path}"',
            f'python "{ROOT / "run_pipeline.py"}" --training-config "{renamed_config_path}"',
            "",
            "Se quiser apenas refazer a finalizacao a partir de uma pasta pronta:",
            f'python "{ROOT / "run_experimental_battery.py"}" --finalize-only --existing-experiment-dir "{output_dir}"',
            "",
        ]
    )
    commands_path = integration_dir / "commands.txt"
    commands_path.write_text(commands_text, encoding="utf-8")

    note = {
        "generated_at": datetime.now().isoformat(),
        "runner": str(ROOT / "run_experimental_battery.py"),
        "exported_config_path_original": str(exported_config_path),
        "exported_config_path_final": str(renamed_config_path),
        "commands_path": str(commands_path),
    }
    dump_json(note, integration_dir / "post_finalization.json")
    return {
        "renamed_config_path": str(renamed_config_path),
        "commands_path": str(commands_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bateria experimental final, cega e explicavel.")
    parser.add_argument("--representation", choices=["blind49", "compact22"], default="blind49")
    parser.add_argument("--search-mode", choices=list(SEARCH_MODES), default="focused")
    parser.add_argument("--data-path", type=Path, default=None)
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--existing-experiment-dir", type=Path, default=None)
    parser.add_argument("--official-config-copy", type=Path, default=None)
    args = parser.parse_args()

    if args.finalize_only:
        if args.existing_experiment_dir is None:
            raise SystemExit("Informe --existing-experiment-dir ao usar --finalize-only.")
        existing_dir = args.existing_experiment_dir.resolve()
        log_path = existing_dir / "run.log"
        finalization = finalize_outputs(
            existing_dir,
            official_config_copy=args.official_config_copy.resolve() if args.official_config_copy else None,
            log_path=log_path,
        )
        post = write_post_finalization(output_dir=existing_dir, finalization=finalization)
        print(f"Config exportada em: {post['renamed_config_path']}")
        print(f"Assets finais em: {finalization['assets_summary']['assets_dir']}")
        return

    if args.data_path is not None:
        data_path = args.data_path.resolve()
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset supervisionado nao encontrado: {data_path}")
        representation_name = "custom_dataset"
        representation_description = f"Dataset customizado informado manualmente em `{data_path}`."
    else:
        representation_spec = resolve_representation_spec(args.representation)
        ensure_representation_dataset(representation_spec)
        data_path = Path(representation_spec["data_path"]).resolve()
        representation_name = str(representation_spec["representation_name"])
        representation_description = str(representation_spec["description"])

    smoke_test = bool(args.smoke_test)
    search_mode = str(args.search_mode).strip().lower()
    stochastic_seeds = parse_csv_ints(args.stochastic_seeds) or list(DEFAULT_STOCHASTIC_SEEDS)
    threshold_objectives = [item.strip() for item in str(args.threshold_objectives).split(",") if item.strip()] or list(DEFAULT_THRESHOLD_OBJECTIVES)
    if smoke_test:
        stochastic_seeds = stochastic_seeds[:1] or [DEFAULT_RANDOM_STATE]
        threshold_objectives = threshold_objectives[:1] or ["accuracy"]
    effective_bootstrap_runs = 40 if smoke_test else args.bootstrap_runs

    default_name = "bateria_experimental_smoke" if smoke_test else f"bateria_experimental_{timestamp_now()}"
    output_dir = resolve_output_dir(args.reports_root, args.experiment_name or default_name)
    ensure_dir(output_dir)
    log_path = output_dir / "run.log"

    dataset = load_dataset(data_path)
    feature_columns = [
        column
        for column in resolve_feature_columns(dataset, data_path)
        if not (
            column.startswith("interaction_")
            or column.startswith("delta_")
            or column.startswith("ratio_")
            or column.startswith("abs_")
        )
    ]
    train_df, holdout_df = split_train_holdout(dataset)
    temporal_folds = build_temporal_folds(train_df)
    feature_specs = build_feature_set_specs(feature_columns, smoke_test=smoke_test, search_mode=search_mode)
    model_specs = build_model_specs(smoke_test=smoke_test, search_mode=search_mode)
    candidates = build_candidate_specs(feature_specs, model_specs)
    feature_lookup = {spec.name: spec for spec in feature_specs}
    model_lookup = {spec.name: spec for spec in model_specs}
    refinement_meta = {"used": False, "refined_feature_set_count": 0, "refined_candidate_count": 0}

    audit = build_pipeline_audit(
        data_path=data_path,
        representation_name=representation_name,
        representation_description=representation_description,
        search_mode=search_mode,
        feature_columns=feature_columns,
        train_df=train_df,
        holdout_df=holdout_df,
        feature_specs=feature_specs,
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

    volume = estimate_experiment_volume(
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
    experiment_config = {
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
        "dataset_path": str(data_path),
        "search_mode": search_mode,
        "base_feature_count": int(len(feature_columns)),
        "search_feature_count": int(len(feature_columns)),
        "feature_set_count_stage1": int(len(feature_specs)),
        "candidate_count_stage1": int(len(candidates)),
        "predefined_reference_used": False,
        "representation_name": representation_name,
        "representation_description": representation_description,
        "subset_refinement_phase2": refinement_meta,
        "volume": volume,
    }
    dump_json(experiment_config, output_dir / "experiment_config.json")

    log(
        f"[setup] features_base={len(feature_columns)} feature_sets={len(feature_specs)} models={len(model_specs)} candidates={len(candidates)} workers={args.workers}",
        log_path,
    )
    log(
        f"[volume] ranking_fits={volume['estimated_feature_ranking_fits']} exploratory_fits={volume['estimated_exploratory_fits']} total_fits~={volume['estimated_total_model_fits']}",
        log_path,
    )

    fold_payloads = prepare_exploratory_folds(
        train_df=train_df,
        temporal_folds=temporal_folds,
        feature_columns=feature_columns,
        feature_specs=feature_specs,
        random_state=args.random_state,
        output_dir=output_dir / "feature_rankings",
        log_path=log_path,
        smoke_test=smoke_test,
        search_profile="focused" if search_mode == "focused" else "balanced",
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
        stage_label="exploratory",
    )
    if search_mode == "extended":
        refinement_result = run_local_subset_refinement_stage(
            output_dir=output_dir,
            train_df=train_df,
            feature_columns=feature_columns,
            feature_specs=feature_specs,
            feature_lookup=feature_lookup,
            model_lookup=model_lookup,
            fold_payloads=fold_payloads,
            exploratory_leaderboard=exploratory_leaderboard,
            random_state=args.random_state,
            workers=args.workers,
            batch_size=8 if smoke_test else args.batch_size,
            stochastic_seeds=stochastic_seeds,
            smoke_test=smoke_test,
            search_mode=search_mode,
            log_path=log_path,
        )
        refinement_meta = {
            "used": bool(refinement_result["used"]),
            "refined_feature_set_count": int(refinement_result["refined_feature_set_count"]),
            "refined_candidate_count": int(refinement_result["refined_candidate_count"]),
        }
        feature_specs = list(refinement_result["feature_specs"])
        feature_lookup = dict(refinement_result["feature_lookup"])
        exploratory_leaderboard = refinement_result["leaderboard"]
        write_csv(
            pd.DataFrame(
                [{"feature_set_name": spec.name, "family": spec.family, "params_json": json.dumps(params_to_dict(spec.params), ensure_ascii=False)} for spec in feature_specs]
            ),
            output_dir / "feature_set_spec_manifest.csv",
        )
        experiment_config["subset_refinement_phase2"] = refinement_meta
        experiment_config["feature_set_count_effective"] = int(len(feature_specs))
        experiment_config["candidate_count_effective"] = int(len(candidates) + refinement_meta["refined_candidate_count"])
        dump_json(experiment_config, output_dir / "experiment_config.json")

    shortlist = freeze_shortlist(
        exploratory_leaderboard,
        shortlist_size=4 if smoke_test else args.shortlist_size,
        per_model_family=1 if smoke_test else args.shortlist_per_model_family,
        output_path=output_dir / "shortlist.csv",
    )
    log(f"[shortlist] finalistas congelados={len(shortlist)}", log_path)

    confirmatory = run_confirmatory_phase(
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
        search_profile="focused" if search_mode == "focused" else "balanced",
        log_path=log_path,
    )

    summary = {
        "generated_at": datetime.now().isoformat(),
        "runner": "run_experimental_battery.py",
        "output_dir": str(output_dir),
        "dataset_path": str(data_path),
        "search_mode": search_mode,
        "representation_name": representation_name,
        "representation_description": representation_description,
        "base_feature_count": int(len(feature_columns)),
        "search_feature_count": int(len(feature_columns)),
        "feature_set_count": int(len(feature_specs)),
        "model_count": int(len(model_specs)),
        "candidate_count": int(len(candidates) + refinement_meta["refined_candidate_count"]),
        "candidate_count_stage1": int(len(candidates)),
        "shortlist_count": int(len(shortlist)),
        "subset_refinement_phase2_used": bool(refinement_meta["used"]),
        "subset_refinement_feature_set_count": int(refinement_meta["refined_feature_set_count"]),
        "subset_refinement_candidate_count": int(refinement_meta["refined_candidate_count"]),
        "predefined_reference_used": False,
        "threshold_objectives": threshold_objectives,
        "best_exploratory_candidate": exploratory_leaderboard.iloc[0]["candidate_name"] if not exploratory_leaderboard.empty else "",
        "best_holdout_candidate": confirmatory["holdout_leaderboard"].iloc[0]["candidate_name"] if not confirmatory["holdout_leaderboard"].empty else "",
        "best_tuned_threshold_candidate": confirmatory["holdout_tuned_leaderboard"].iloc[0]["candidate_name"] if not confirmatory["holdout_tuned_leaderboard"].empty else "",
    }

    finalization = finalize_outputs(
        output_dir,
        official_config_copy=args.official_config_copy.resolve() if args.official_config_copy else None,
        log_path=log_path,
    )
    post = write_post_finalization(output_dir=output_dir, finalization=finalization)

    summary.update(
        {
            "official_selected_candidate": finalization["integration_summary"]["selected_candidate_name"],
            "official_selected_model_family": finalization["integration_summary"]["selected_model_family"],
            "official_selected_config_path": post["renamed_config_path"],
            "dissertation_assets_dir": finalization["assets_summary"]["assets_dir"],
            "commands_path": post["commands_path"],
        }
    )
    dump_json(summary, output_dir / "summary.json")

    texto_base_path = write_texto_base(
        output_dir=output_dir,
        data_path=data_path,
        representation_name=representation_name,
        representation_description=representation_description,
        search_mode=search_mode,
        feature_columns=feature_columns,
        feature_specs=feature_specs,
        model_specs=model_specs,
        summary=summary,
        selected_config_path=Path(post["renamed_config_path"]),
    )
    log(f"[docs] texto base gerado em {texto_base_path}", log_path)


if __name__ == "__main__":
    main()
