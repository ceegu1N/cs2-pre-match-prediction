#!/usr/bin/env python3
"""Faz previsao para um confronto entre dois times.

Exemplo:
    python predict_match.py --team-a equipe-aurora --team-b equipe-horizonte
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.cs2_pipeline import (
    build_match_row,
    load_feature_columns,
    load_model_registry,
    resolve_model_name,
    slugify_team_name,
)

ROOT = Path(__file__).resolve().parent
DEMO_SNAPSHOT_PATH = ROOT / 'data' / 'sample' / 'team_snapshot_synthetic.csv'
FEATURES_PATH = ROOT / 'models' / 'feature_columns.json'
REGISTRY_PATH = ROOT / 'models' / 'model_registry.json'
SNAPSHOT_MODE_PATHS = {'demo': DEMO_SNAPSHOT_PATH}


def resolve_snapshot_path(snapshot_mode: str = 'demo', snapshot_file: str | None = None) -> Path:
    if snapshot_file:
        snapshot_path = Path(snapshot_file)
        if not snapshot_path.is_absolute():
            snapshot_path = (ROOT / snapshot_path).resolve()
        return snapshot_path
    try:
        return SNAPSHOT_MODE_PATHS[snapshot_mode]
    except KeyError as exc:
        valid_modes = ', '.join(sorted(SNAPSHOT_MODE_PATHS))
        raise ValueError(f'Snapshot mode invalido: {snapshot_mode}. Modos validos: {valid_modes}') from exc


def load_prediction_context(snapshot_mode: str = 'demo', snapshot_file: str | None = None) -> tuple[pd.DataFrame, list[str], dict[str, Any], Path]:
    snapshot_path = resolve_snapshot_path(snapshot_mode=snapshot_mode, snapshot_file=snapshot_file)
    snapshot = pd.read_csv(snapshot_path)
    feature_cols = load_feature_columns(FEATURES_PATH)
    registry = load_model_registry(REGISTRY_PATH)
    return snapshot, feature_cols, registry, snapshot_path


def list_available_teams(snapshot_mode: str = 'demo', snapshot_file: str | None = None) -> list[dict[str, str]]:
    snapshot, _, _, _ = load_prediction_context(snapshot_mode=snapshot_mode, snapshot_file=snapshot_file)
    teams = (
        snapshot[['team_display_name', 'team_slug']]
        .drop_duplicates()
        .sort_values('team_display_name', kind='stable')
        .reset_index(drop=True)
    )
    return teams.rename(columns={'team_display_name': 'display_name', 'team_slug': 'slug'}).to_dict(orient='records')


def predict_match_probability(
    team_a_name: str,
    team_b_name: str,
    model_selector: str = 'primary',
    snapshot_mode: str = 'demo',
    snapshot_file: str | None = None,
) -> dict[str, Any]:
    team_a = slugify_team_name(team_a_name)
    team_b = slugify_team_name(team_b_name)
    if team_a == team_b:
        raise ValueError('Escolha dois times diferentes.')

    snapshot, feature_cols, registry, snapshot_path = load_prediction_context(
        snapshot_mode=snapshot_mode,
        snapshot_file=snapshot_file,
    )
    available = set(snapshot['team_slug'])
    if team_a not in available or team_b not in available:
        raise ValueError(
            f'Um ou ambos os times nao foram encontrados em {snapshot_path.name}. '
            f'Times disponiveis: {", ".join(sorted(available))}'
        )

    try:
        model_name = resolve_model_name(registry, model_selector)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    model_path = ROOT / 'models' / f'{model_name}.joblib'
    if not model_path.exists():
        raise ValueError(f'Modelo nao encontrado: {model_path.name}')

    model = joblib.load(model_path)
    row = build_match_row(snapshot, team_a, team_b, feature_cols)
    reverse_row = build_match_row(snapshot, team_b, team_a, feature_cols)
    direct_proba_a = float(model.predict_proba(row)[0, 1])
    reverse_proba_b = float(model.predict_proba(reverse_row)[0, 1])
    proba_a = (direct_proba_a + (1.0 - reverse_proba_b)) / 2.0
    proba_b = 1.0 - proba_a

    display_a = snapshot.loc[snapshot['team_slug'] == team_a, 'team_display_name'].iloc[0]
    display_b = snapshot.loc[snapshot['team_slug'] == team_b, 'team_display_name'].iloc[0]

    return {
        'team_a_slug': team_a,
        'team_b_slug': team_b,
        'team_a_display_name': display_a,
        'team_b_display_name': display_b,
        'model_name': model_name,
        'model_selector': model_selector,
        'snapshot_mode': snapshot_mode,
        'snapshot_path': str(snapshot_path),
        'probability_team_a': proba_a,
        'probability_team_b': proba_b,
        'probability_team_a_direct': direct_proba_a,
        'probability_team_b_direct_reverse': reverse_proba_b,
        'probability_symmetrized': True,
        'favorite': display_a if proba_a >= 0.5 else display_b,
        'feature_values': {column: float(row.iloc[0][column]) for column in row.columns},
        'team_a_snapshot': snapshot.loc[snapshot['team_slug'] == team_a].iloc[0].to_dict(),
        'team_b_snapshot': snapshot.loc[snapshot['team_slug'] == team_b].iloc[0].to_dict(),
        'available_models': list(registry.get('available_models', [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--team-a', required=True, help='Slug do time A, ex.: equipe-aurora')
    parser.add_argument('--team-b', required=True, help='Slug do time B, ex.: equipe-horizonte')
    parser.add_argument(
        '--model',
        default='primary',
        help='Modelo a usar: primary, best, best_test, best_cv ou o nome exato salvo em models/.',
    )
    parser.add_argument(
        '--snapshot-mode',
        default='demo',
        choices=sorted(SNAPSHOT_MODE_PATHS),
        help='Usa a demonstracao com equipes e atributos sinteticos.',
    )
    parser.add_argument(
        '--snapshot-file',
        default=None,
        help='Caminho opcional para um snapshot customizado. Se informado, sobrescreve --snapshot-mode.',
    )
    args = parser.parse_args()

    try:
        result = predict_match_probability(
            args.team_a,
            args.team_b,
            model_selector=args.model,
            snapshot_mode=args.snapshot_mode,
            snapshot_file=args.snapshot_file,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"{result['team_a_display_name']} vs {result['team_b_display_name']}")
    print(f"Modelo utilizado: {result['model_name']}")
    print(f"Snapshot utilizado: {result['snapshot_path']}")
    print(f"Probabilidade de vitoria de {result['team_a_display_name']}: {result['probability_team_a']:.2%}")
    print(f"Probabilidade de vitoria de {result['team_b_display_name']}: {result['probability_team_b']:.2%}")
    print('Favorito:', result['favorite'])


if __name__ == '__main__':
    main()
