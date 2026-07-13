#!/usr/bin/env python3
"""Orquestra o pipeline principal do TCC em um unico comando."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.cs2_pipeline import (
    DEFAULT_ELO_INITIAL_RATING,
    DEFAULT_ELO_K_FACTOR,
    DEFAULT_ELO_SCALE,
    DEFAULT_RECENT_MATCH_COUNT,
    save_processed_datasets,
)
from train_model import DEFAULT_TRAINING_CONFIG_PATH, run_training

ROOT = Path(__file__).resolve().parent

DEFAULT_PLAYERS_PATH = ROOT / "data" / "raw" / "core"
DEFAULT_MATCHES_PATH = ROOT / "data" / "raw" / "core" / "matches_top50_dated.csv"
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_MODELS_DIR = ROOT / "models"
DEFAULT_REPORTS_DIR = ROOT / "reports"


def resolve_path(raw_path: str, default_root: Path = ROOT) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (default_root / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players-file", default=str(DEFAULT_PLAYERS_PATH), help="Arquivo ou diretorio com os snapshots temporais de jogadores.")
    parser.add_argument("--matches-file", default=str(DEFAULT_MATCHES_PATH))
    parser.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument(
        "--training-config",
        default=str(DEFAULT_TRAINING_CONFIG_PATH),
        help="JSON de configuracao do modelo logistico oficial. Pode apontar para qualquer config exportada pela bateria experimental.",
    )
    parser.add_argument("--split-strategy", choices=["stratified", "temporal", "season_holdout"], default="season_holdout")
    parser.add_argument("--recent-match-count", type=int, default=DEFAULT_RECENT_MATCH_COUNT)
    elo_group = parser.add_mutually_exclusive_group()
    elo_group.add_argument(
        "--include-elo",
        dest="include_elo",
        action="store_true",
        help="Mantem habilitada a feature diff_elo_pre_match.",
    )
    elo_group.add_argument(
        "--no-elo",
        dest="include_elo",
        action="store_false",
        help="Desabilita a feature diff_elo_pre_match para comparacoes experimentais.",
    )
    parser.add_argument("--elo-initial-rating", type=float, default=DEFAULT_ELO_INITIAL_RATING)
    parser.add_argument("--elo-k-factor", type=float, default=DEFAULT_ELO_K_FACTOR)
    parser.add_argument("--elo-scale", type=float, default=DEFAULT_ELO_SCALE)
    parser.set_defaults(include_elo=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    players_path = resolve_path(args.players_file)
    matches_path = resolve_path(args.matches_file)
    processed_dir = resolve_path(args.processed_dir)
    models_dir = resolve_path(args.models_dir)
    reports_dir = resolve_path(args.reports_dir)
    training_config_path = resolve_path(args.training_config)

    snapshot, features = save_processed_datasets(
        ROOT,
        players_path=players_path,
        matches_path=matches_path,
        processed_dir=processed_dir,
        recent_match_count=args.recent_match_count,
        include_elo=args.include_elo,
        elo_initial_rating=args.elo_initial_rating,
        elo_k_factor=args.elo_k_factor,
        elo_scale=args.elo_scale,
    )

    metrics, registry = run_training(
        data_path=processed_dir / "match_feature_differences.csv",
        snapshot_path=processed_dir / "team_snapshot.csv",
        models_dir=models_dir,
        reports_dir=reports_dir,
        logreg_config_path=training_config_path,
        split_strategy=args.split_strategy,
        test_size=0.2,
    )

    best = metrics.sort_values("test_roc_auc", ascending=False).iloc[0]
    logistic = metrics.loc[metrics["model"] == "logistic_regression"].iloc[0]

    print(f"Arquivo de jogadores usado: {players_path}")
    print(f"Arquivo de partidas usado: {matches_path}")
    print(f"Times no snapshot: {snapshot['team_slug'].nunique()}")
    print(f"Confrontos validos: {len(features)}")
    print(f"Split: {args.split_strategy}")
    print(f"Janela de win rate recente: {args.recent_match_count} partidas")
    print(f"Elo pre-jogo: {'habilitado' if args.include_elo else 'desabilitado'}")
    print(f"Configuracao de treino: {training_config_path}")
    print()
    print(f"Melhor modelo de teste: {best['model']} (ROC-AUC {best['test_roc_auc']:.3f})")
    print(
        "Regressao logistica: "
        f"ROC-AUC {logistic['test_roc_auc']:.3f}, "
        f"log-loss {logistic['test_log_loss']:.3f}, "
        f"Brier {logistic['test_brier']:.3f}"
    )
    print(f"Modelos salvos em: {models_dir}")
    print(f"Relatorios atualizados em: {reports_dir}")
    if "test_period_start" in registry and "test_period_end" in registry:
        print(f"Periodo do holdout final: {registry['test_period_start']} ate {registry['test_period_end']}")


if __name__ == "__main__":
    main()
