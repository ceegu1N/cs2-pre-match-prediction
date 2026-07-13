#!/usr/bin/env python3
"""Regenera os datasets processados a partir dos CSVs brutos.

Fluxo oficial atual:
- snapshots temporais de jogadores por season
- contexto dinamico pre-jogo com recent win rate e Elo classico
- dataset supervisionado final em ``data/processed/match_feature_differences.csv``

Observacao:
- H2H e Elo com decay continuam disponiveis por compatibilidade e pesquisa
- essas opcoes sao experimentais e nao fazem parte do modelo principal oficial
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.cs2_pipeline import (
    DEFAULT_RECENT_MATCH_COUNT,
    DEFAULT_ELO_INITIAL_RATING,
    DEFAULT_ELO_K_FACTOR,
    DEFAULT_ELO_SCALE,
    DEFAULT_ELO_DECAY_RATE,
    DEFAULT_H2H_MIN_GAMES,
    load_matches,
    load_players,
    save_processed_datasets,
    get_extended_feature_columns,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_PLAYERS_PATH = ROOT / "data" / "raw" / "core"
DEFAULT_MATCHES_PATH = ROOT / "data" / "raw" / "core" / "matches_top50_dated.csv"
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"


def resolve_path(raw_path: str, default_root: Path = ROOT) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (default_root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players-file", default=str(DEFAULT_PLAYERS_PATH))
    parser.add_argument("--matches-file", default=str(DEFAULT_MATCHES_PATH))
    parser.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR))
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
    parser.add_argument("--elo-decay-rate", type=float, default=DEFAULT_ELO_DECAY_RATE)
    parser.add_argument("--h2h-min-games", type=int, default=DEFAULT_H2H_MIN_GAMES)
    parser.add_argument(
        "--include-elo-decay",
        action="store_true",
        help="Opcao experimental: adiciona diff_elo_decay_pre_match usando decay temporal no Elo.",
    )
    parser.add_argument(
        "--include-h2h",
        action="store_true",
        help="Opcao experimental: adiciona diff_h2h_win_rate e has_h2h_history, calculadas pre-jogo por Match_ID.",
    )
    parser.set_defaults(include_elo=True)
    args = parser.parse_args()

    players_path = resolve_path(args.players_file)
    matches_path = resolve_path(args.matches_file)
    processed_dir = resolve_path(args.processed_dir)

    players = load_players(players_path)
    matches = load_matches(matches_path)
    snapshot, features = save_processed_datasets(
        ROOT,
        players_path=players_path,
        matches_path=matches_path,
        processed_dir=processed_dir,
        recent_match_count=args.recent_match_count,
        include_elo=args.include_elo,
        include_elo_decay=args.include_elo_decay,
        include_h2h=args.include_h2h,
        elo_initial_rating=args.elo_initial_rating,
        elo_k_factor=args.elo_k_factor,
        elo_scale=args.elo_scale,
        elo_decay_rate=args.elo_decay_rate,
        h2h_min_games=args.h2h_min_games,
    )

    feature_columns = get_extended_feature_columns(
        include_elo=args.include_elo,
        include_elo_decay=args.include_elo_decay,
        include_h2h=args.include_h2h,
    )
    duplicate_pairings = int(features[["team_name", "opponent_name"]].duplicated().sum())
    duplicate_feature_vectors = int(features[feature_columns].duplicated().sum())
    nonstandard_team_counts = players.groupby(["season_label", "team_display_name"], sort=False).size()
    nonstandard_team_counts = nonstandard_team_counts.loc[nonstandard_team_counts != 5]

    print(f"Arquivo de jogadores: {players_path}")
    print(f"Arquivo de partidas: {matches_path}")
    print(f"Diretorio processado: {processed_dir}")
    print(f"Elo pre-jogo habilitado: {args.include_elo}")
    print(f"Elo com decay experimental habilitado: {args.include_elo_decay}")
    print(f"H2H experimental habilitado: {args.include_h2h}")
    print(f"Jogadores brutos: {len(players)} linhas")
    print(f"Seasons de jogadores: {sorted(players['season_label'].unique().tolist())}")
    print(f"Times unicos totais nos jogadores: {players['team_key'].nunique()}")
    if not nonstandard_team_counts.empty:
        print(f"Times com quantidade de jogadores diferente de 5: {nonstandard_team_counts.to_dict()}")

    print(f"Partidas brutas: {len(matches)} linhas")
    if "Match_Date" in matches.columns and matches["Match_Date"].notna().all():
        print(f"Periodo das partidas: {matches['Match_Date'].min().date()} ate {matches['Match_Date'].max().date()}")

    print(f"Snapshot por season salvo em: {processed_dir / 'team_snapshot_by_season.csv'}")
    print(f"Snapshot final salvo em: {processed_dir / 'team_snapshot.csv'}")
    print(f"Snapshot final: {snapshot['team_slug'].nunique()} times")
    print(f"Dataset supervisionado salvo em: {processed_dir / 'match_feature_differences.csv'}")
    print(f"Confrontos validos: {len(features)} linhas")
    if "match_date" in features.columns:
        print(f"Periodo do dataset supervisionado: {features['match_date'].min()} ate {features['match_date'].max()}")
    print(f"Confrontos repetidos entre os mesmos pares de times: {duplicate_pairings}")
    print(f"Vetores de atributos duplicados no dataset final: {duplicate_feature_vectors}")
    print(f"Features finais: {', '.join(feature_columns)}")
    print("Valores ausentes nas metricas individuais:")
    print(players[["rating", "impact", "kast", "adr", "kd", "kills_per_round", "deaths_per_round", "headshot_pct", "utility_damage_per_round", "flash_assists_per_round", "opening_kills_per_round", "opening_success", "trade_kills_per_round"]].isna().sum().to_string())


if __name__ == "__main__":
    main()
