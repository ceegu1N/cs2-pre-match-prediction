"""Utilitarios compartilhados para o pipeline do TCC de CS2."""
from __future__ import annotations

import json
import math
import re
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

BASE_FEATURE_COLUMNS = [
    "diff_recent_win_rate",
    "diff_maps_played_mean_5",
    "diff_rounds_played_mean_5",
    "diff_total_kills_mean_5",
    "diff_rating_mean_5",
    "diff_adr_mean_5",
    "diff_impact_mean_5",
    "diff_kast_mean_5",
    "diff_kd_mean_5",
    "diff_kills_per_round_mean_5",
    "diff_deaths_per_round_mean_5",
    "diff_assists_per_round_mean_5",
    "diff_headshot_pct_mean_5",
    "diff_saved_by_teammate_per_round_mean_5",
    "diff_utility_damage_per_round_mean_5",
    "diff_flash_assists_per_round_mean_5",
    "diff_utility_kills_per_100_rounds_mean_5",
    "diff_time_opponent_flashed_per_round_mean_5",
    "diff_opening_kills_per_round_mean_5",
    "diff_opening_deaths_per_round_mean_5",
    "diff_opening_success_mean_5",
    "diff_win_after_opening_kill_pct_mean_5",
    "diff_trade_kills_per_round_mean_5",
    "diff_saved_teammate_per_round_mean_5",
    "diff_last_alive_pct_mean_5",
    "diff_one_on_one_win_pct_mean_5",
    "diff_time_alive_per_round_mean_5",
    "diff_rounds_with_a_kill_mean_5",
    "diff_kills_per_round_win_mean_5",
    "diff_damage_per_round_win_mean_5",
    "diff_pistol_round_rating_mean_5",
    "diff_players_count",
    "diff_missing_players",
    "context_is_lan",
    "context_is_bo1",
    "context_is_bo3",
    "context_is_bo5",
]

ELO_FEATURE_COLUMN = "diff_elo_pre_match"
# As colunas abaixo continuam disponiveis para pesquisa controlada, mas
# nao pertencem ao fluxo oficial atual do modelo principal do TCC.
ELO_DECAY_FEATURE_COLUMN = "diff_elo_decay_pre_match"
H2H_WIN_RATE_FEATURE_COLUMN = "diff_h2h_win_rate"
H2H_HISTORY_FEATURE_COLUMN = "has_h2h_history"
FEATURE_COLUMNS = list(BASE_FEATURE_COLUMNS)

DEFAULT_RECENT_MATCH_COUNT = 30
DEFAULT_ELO_INITIAL_RATING = 1500.0
DEFAULT_ELO_K_FACTOR = 32.0
DEFAULT_ELO_SCALE = 400.0
DEFAULT_ELO_DECAY_RATE = 0.05
DEFAULT_H2H_MIN_GAMES = 3
DEFAULT_HOLDOUT_SEASON_LABEL = "2026_s1"
TRAIN_SEASON_USAGE = "train"
HOLDOUT_SEASON_USAGE = "holdout"

PLAYER_RENAME_MAP = {
    "Team": "team_display_name",
    "Player": "player_name",
    "Maps Played": "maps_played",
    "Rounds Played": "rounds_played",
    "Total Kills": "total_kills",
    "Rating 2.0": "rating",
    "Rating 3.0": "rating",
    "Impact": "impact",
    "KAST": "kast",
    "ADR": "adr",
    "K/D Ratio": "kd",
    "Kills / Round": "kills_per_round",
    "Deaths / Round": "deaths_per_round",
    "Assists / Round": "assists_per_round",
    "Headshot %": "headshot_pct",
    "Saved By Teammate / Round": "saved_by_teammate_per_round",
    "Utility Damage / Round": "utility_damage_per_round",
    "Flash Assists / Round": "flash_assists_per_round",
    "Utility Kills / 100 Rounds": "utility_kills_per_100_rounds",
    "Time Opponent Flashed / Round": "time_opponent_flashed_per_round",
    "Opening Kills / Round": "opening_kills_per_round",
    "Opening Deaths / Round": "opening_deaths_per_round",
    "Opening Success": "opening_success",
    "Win% After Opening Kill": "win_after_opening_kill_pct",
    "Trade Kills / Round": "trade_kills_per_round",
    "Saved Teammate / Round": "saved_teammate_per_round",
    "Last Alive Percentage": "last_alive_pct",
    "1on1 Win Percentage": "one_on_one_win_pct",
    "Time Alive / Round": "time_alive_per_round",
    "Rounds With a Kill": "rounds_with_a_kill",
    "Kills / Round Win": "kills_per_round_win",
    "Damage / Round Win": "damage_per_round_win",
    "Pistol Round Rating": "pistol_round_rating",
}

PLAYER_NUMERIC_COLUMNS = [
    "maps_played",
    "rounds_played",
    "total_kills",
    "rating",
    "impact",
    "kast",
    "adr",
    "kd",
    "kills_per_round",
    "deaths_per_round",
    "assists_per_round",
    "headshot_pct",
    "saved_by_teammate_per_round",
    "utility_damage_per_round",
    "flash_assists_per_round",
    "utility_kills_per_100_rounds",
    "time_opponent_flashed_per_round",
    "opening_kills_per_round",
    "opening_deaths_per_round",
    "opening_success",
    "win_after_opening_kill_pct",
    "trade_kills_per_round",
    "saved_teammate_per_round",
    "last_alive_pct",
    "one_on_one_win_pct",
    "time_alive_per_round",
    "rounds_with_a_kill",
    "kills_per_round_win",
    "damage_per_round_win",
    "pistol_round_rating",
]

PLAYER_AGGREGATIONS = {
    "maps_played": "maps_played_mean_5",
    "rounds_played": "rounds_played_mean_5",
    "total_kills": "total_kills_mean_5",
    "rating": "rating_mean_5",
    "impact": "impact_mean_5",
    "kast": "kast_mean_5",
    "adr": "adr_mean_5",
    "kd": "kd_mean_5",
    "kills_per_round": "kills_per_round_mean_5",
    "deaths_per_round": "deaths_per_round_mean_5",
    "assists_per_round": "assists_per_round_mean_5",
    "headshot_pct": "headshot_pct_mean_5",
    "saved_by_teammate_per_round": "saved_by_teammate_per_round_mean_5",
    "utility_damage_per_round": "utility_damage_per_round_mean_5",
    "flash_assists_per_round": "flash_assists_per_round_mean_5",
    "utility_kills_per_100_rounds": "utility_kills_per_100_rounds_mean_5",
    "time_opponent_flashed_per_round": "time_opponent_flashed_per_round_mean_5",
    "opening_kills_per_round": "opening_kills_per_round_mean_5",
    "opening_deaths_per_round": "opening_deaths_per_round_mean_5",
    "opening_success": "opening_success_mean_5",
    "win_after_opening_kill_pct": "win_after_opening_kill_pct_mean_5",
    "trade_kills_per_round": "trade_kills_per_round_mean_5",
    "saved_teammate_per_round": "saved_teammate_per_round_mean_5",
    "last_alive_pct": "last_alive_pct_mean_5",
    "one_on_one_win_pct": "one_on_one_win_pct_mean_5",
    "time_alive_per_round": "time_alive_per_round_mean_5",
    "rounds_with_a_kill": "rounds_with_a_kill_mean_5",
    "kills_per_round_win": "kills_per_round_win_mean_5",
    "damage_per_round_win": "damage_per_round_win_mean_5",
    "pistol_round_rating": "pistol_round_rating_mean_5",
}

ROUNDING_BY_SNAPSHOT_COLUMN = {
    "maps_played_mean_5": 1,
    "rounds_played_mean_5": 1,
    "total_kills_mean_5": 1,
    "rating_mean_5": 3,
    "impact_mean_5": 3,
    "kast_mean_5": 2,
    "adr_mean_5": 2,
    "kd_mean_5": 3,
    "kills_per_round_mean_5": 3,
    "deaths_per_round_mean_5": 3,
    "assists_per_round_mean_5": 3,
    "headshot_pct_mean_5": 2,
    "saved_by_teammate_per_round_mean_5": 3,
    "utility_damage_per_round_mean_5": 3,
    "flash_assists_per_round_mean_5": 3,
    "utility_kills_per_100_rounds_mean_5": 3,
    "time_opponent_flashed_per_round_mean_5": 3,
    "opening_kills_per_round_mean_5": 3,
    "opening_deaths_per_round_mean_5": 3,
    "opening_success_mean_5": 2,
    "win_after_opening_kill_pct_mean_5": 2,
    "trade_kills_per_round_mean_5": 3,
    "saved_teammate_per_round_mean_5": 3,
    "last_alive_pct_mean_5": 2,
    "one_on_one_win_pct_mean_5": 2,
    "time_alive_per_round_mean_5": 2,
    "rounds_with_a_kill_mean_5": 2,
    "kills_per_round_win_mean_5": 3,
    "damage_per_round_win_mean_5": 2,
    "pistol_round_rating_mean_5": 3,
    "players_count": 0,
    "missing_players": 0,
    "recent_win_rate": 3,
    "elo_rating": 3,
}

SEASON_SNAPSHOT_COLUMNS = [
    "season_label",
    "season_usage",
    "team_key",
    "team_slug",
    "team_display_name",
    "players_count",
    "missing_players",
    *PLAYER_AGGREGATIONS.values(),
]

LATEST_SNAPSHOT_COLUMNS = [
    "team_slug",
    "team_display_name",
    "players_count",
    "missing_players",
    *PLAYER_AGGREGATIONS.values(),
    "recent_win_rate",
]

DEFAULT_PRIMARY_MODEL = "logistic_regression"
NEUTRAL_RECENT_WIN_RATE = 0.5
SEASON_FILE_RE = re.compile(r"players_top50_(20\d{2}_s[12])\.csv$", re.IGNORECASE)


def get_feature_columns(include_elo: bool = False) -> list[str]:
    columns = list(BASE_FEATURE_COLUMNS)
    if include_elo:
        columns.append(ELO_FEATURE_COLUMN)
    return columns


def get_extended_feature_columns(
    include_elo: bool = False,
    include_elo_decay: bool = False,
    include_h2h: bool = False,
) -> list[str]:
    """Retorna a lista de features ativas para o dataset supervisionado.

    Observacao:
    - o fluxo oficial atual usa BASE_FEATURE_COLUMNS + diff_elo_pre_match
    - H2H e Elo com decay ficam aqui apenas como extensoes experimentais
    """
    columns = get_feature_columns(include_elo=include_elo)
    if include_elo_decay:
        columns.append(ELO_DECAY_FEATURE_COLUMN)
    if include_h2h:
        columns.extend([H2H_WIN_RATE_FEATURE_COLUMN, H2H_HISTORY_FEATURE_COLUMN])
    return columns


def get_snapshot_columns(include_elo: bool = False, include_elo_decay: bool = False) -> list[str]:
    columns = list(LATEST_SNAPSHOT_COLUMNS)
    if include_elo:
        columns.append("elo_rating")
    if include_elo_decay:
        columns.append("elo_decay_rating")
    return columns


def canonical_team_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def slugify_team_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower().replace("_", "-"))
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized.strip("-")


def coerce_numeric(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    time_mask = raw.str.fullmatch(r"(?i)(?:\d+m\s*)?(?:\d+s)")
    cleaned = (
        raw
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"-": pd.NA, "": pd.NA, "nan": pd.NA, "None": pd.NA})
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")
    if time_mask.any():
        extracted = raw.loc[time_mask].str.extract(r"(?i)(?:(?P<minutes>\d+)m\s*)?(?:(?P<seconds>\d+)s)")
        minutes = pd.to_numeric(extracted["minutes"], errors="coerce").fillna(0)
        seconds = pd.to_numeric(extracted["seconds"], errors="coerce").fillna(0)
        numeric.loc[time_mask] = (minutes * 60 + seconds).astype(float).to_numpy()
    return numeric



def excel_friendly_value(value: object) -> object:
    if pd.isna(value):
        return ""
    text = str(value)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return text.replace(".", ",")
    return text


def write_excel_friendly_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df = df.copy()
    for column in export_df.columns:
        export_df[column] = export_df[column].map(excel_friendly_value)
    export_df.to_csv(output_path, index=False, sep=";", encoding="utf-8-sig")


def save_visual_processed_csvs(root: Path, snapshot_by_season: pd.DataFrame, snapshot: pd.DataFrame, match_features: pd.DataFrame) -> None:
    planilhas_dir = root / "Planilhas"
    write_excel_friendly_csv(snapshot_by_season, planilhas_dir / "team_snapshot_by_season.csv")
    write_excel_friendly_csv(snapshot, planilhas_dir / "team_snapshot.csv")
    write_excel_friendly_csv(match_features, planilhas_dir / "match_feature_differences.csv")


def infer_season_label_from_path(path: Path) -> str:
    match = SEASON_FILE_RE.search(path.name)
    if match:
        return match.group(1).lower()
    return "global"


def infer_season_usage(season_label: str) -> str:
    label = str(season_label).lower()
    if label == DEFAULT_HOLDOUT_SEASON_LABEL:
        return HOLDOUT_SEASON_USAGE
    if re.fullmatch(r"20\d{2}_s[12]", label):
        return TRAIN_SEASON_USAGE
    return TRAIN_SEASON_USAGE


def season_sort_key(label: str) -> tuple[int, int]:
    match = re.fullmatch(r"(20\d{2})_s([12])", str(label).lower())
    if not match:
        return (0, 0)
    return int(match.group(1)), int(match.group(2))


def resolve_player_files(players_path: Path) -> list[Path]:
    path = Path(players_path)
    if path.is_file():
        return [path]

    files = sorted(
        [candidate for candidate in path.glob("players_top50_20*_s*.csv") if "_audit" not in candidate.name and "_summary" not in candidate.name and "_recheck" not in candidate.name],
        key=lambda candidate: season_sort_key(infer_season_label_from_path(candidate)),
    )
    if not files:
        raise FileNotFoundError(
            "Nenhum arquivo temporal de jogadores foi encontrado. "
            f"Esperado em: {path}"
        )
    return files


def load_players(players_path: Path) -> pd.DataFrame:
    player_files = resolve_player_files(players_path)
    frames: list[pd.DataFrame] = []
    for file_path in player_files:
        players = pd.read_csv(file_path).rename(columns=PLAYER_RENAME_MAP)
        for column in PLAYER_NUMERIC_COLUMNS:
            if column not in players.columns:
                players[column] = pd.NA
            players[column] = coerce_numeric(players[column])

        players["season_label"] = infer_season_label_from_path(file_path)
        players["season_usage"] = players["season_label"].map(infer_season_usage)
        players["team_key"] = players["team_display_name"].map(canonical_team_key)
        players["team_slug"] = players["team_display_name"].map(slugify_team_name)
        frames.append(players)

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["season_label", "team_display_name", "player_name"], kind="stable").reset_index(drop=True)


def load_matches(matches_path: Path) -> pd.DataFrame:
    matches = pd.read_csv(matches_path).copy()
    matches["Win"] = pd.to_numeric(matches["Win"], errors="coerce").astype("Int64")
    if matches["Win"].isna().any():
        invalid_rows = int(matches["Win"].isna().sum())
        raise ValueError(f"A coluna Win contem valores ausentes ou invalidos em {invalid_rows} linhas.")
    matches["Win"] = matches["Win"].astype(int)
    if "Team_Score" in matches.columns:
        matches["Team_Score"] = pd.to_numeric(matches["Team_Score"], errors="coerce")
    if "Opponent_Score" in matches.columns:
        matches["Opponent_Score"] = pd.to_numeric(matches["Opponent_Score"], errors="coerce")
    if "Match_Date" in matches.columns:
        matches["Match_Date"] = pd.to_datetime(matches["Match_Date"], errors="coerce")
    if "Season_Label" in matches.columns:
        matches["Season_Label"] = matches["Season_Label"].astype(str).str.strip().str.lower()
    if "Season_Usage" in matches.columns:
        matches["Season_Usage"] = matches["Season_Usage"].astype(str).str.strip().str.lower()

    if "LAN_Online" not in matches.columns and "Event_Name" in matches.columns:
        context_path = Path(matches_path).with_name("event_context_hltv.csv")
        if context_path.exists():
            event_context = pd.read_csv(context_path, usecols=["Event_Name", "LAN_Online", "Event_ID", "Event_URL", "Event_Location"])
            matches = matches.merge(event_context, on="Event_Name", how="left")

    if "Best_Of" in matches.columns:
        matches["Best_Of"] = matches["Best_Of"].fillna("").astype(str).str.strip().str.lower()
    if "LAN_Online" in matches.columns:
        matches["LAN_Online"] = matches["LAN_Online"].fillna("").astype(str).str.strip().str.upper()
    if "Map_Indicator" in matches.columns:
        matches["Map_Indicator"] = matches["Map_Indicator"].fillna("").astype(str).str.strip().str.lower()

    matches["team_key"] = matches["Team_Name"].map(canonical_team_key)
    matches["opponent_key"] = matches["Opponent"].map(canonical_team_key)
    matches["team_slug"] = matches["Team_Name"].map(slugify_team_name)
    matches["opponent_slug"] = matches["Opponent"].map(slugify_team_name)
    return matches


def validate_matches_structure(matches: pd.DataFrame) -> None:
    if matches.empty:
        raise ValueError("O arquivo de historico de partidas foi carregado vazio.")

    invalid_targets = matches.loc[~matches["Win"].isin([0, 1]), "Win"].unique().tolist()
    if invalid_targets:
        raise ValueError(f"A coluna Win contem valores invalidos: {invalid_targets}")

    missing_team_names = int(matches["Team_Name"].isna().sum())
    missing_opponents = int(matches["Opponent"].isna().sum())
    if missing_team_names or missing_opponents:
        raise ValueError(
            "O arquivo de historico contem nomes ausentes em Team_Name ou Opponent. "
            f"Team_Name ausentes: {missing_team_names}; Opponent ausentes: {missing_opponents}."
        )

    required_columns = {"Season_Label", "Season_Usage", "Match_Date"}
    missing_columns = sorted(column for column in required_columns if column not in matches.columns)
    if missing_columns:
        raise ValueError(
            "O arquivo de historico nao contem as colunas obrigatorias para o pipeline temporal: "
            f"{missing_columns}"
        )

    missing_dates = int(matches["Match_Date"].isna().sum())
    if missing_dates:
        raise ValueError(
            "O arquivo de historico contem datas ausentes ou invalidas em Match_Date. "
            f"Total de linhas com problema: {missing_dates}."
        )


def validate_players_structure(
    players: pd.DataFrame,
    expected_team_count: int = 50,
    min_players_per_team: int = 3,
    max_players_per_team: int = 5,
) -> None:
    if players.empty:
        raise ValueError("O arquivo de jogadores foi carregado vazio.")

    team_counts = players.groupby(["season_label", "team_display_name"], sort=False).size()
    invalid_teams = team_counts.loc[(team_counts < min_players_per_team) | (team_counts > max_players_per_team)]
    if not invalid_teams.empty:
        raise ValueError(
            "Estrutura invalida no arquivo de jogadores. "
            f"Times com quantidade inesperada de jogadores: {invalid_teams.to_dict()}"
        )

    season_counts = players.groupby("season_label", sort=False)["team_key"].nunique()
    invalid_seasons = season_counts.loc[season_counts != expected_team_count]
    if not invalid_seasons.empty:
        raise ValueError(
            "Estrutura invalida no arquivo de jogadores. "
            f"Esperados {expected_team_count} times por season, encontrados: {invalid_seasons.to_dict()}"
        )


def validate_snapshot(snapshot: pd.DataFrame, expected_team_count: int = 50, include_elo: bool = False) -> None:
    if snapshot.empty:
        raise ValueError("team_snapshot.csv foi gerado vazio.")
    if int(snapshot["team_slug"].nunique()) != expected_team_count:
        raise ValueError(
            "team_snapshot.csv nao contem a quantidade esperada de times. "
            f"Esperados {expected_team_count}, encontrados {snapshot['team_slug'].nunique()}."
        )
    if snapshot["recent_win_rate"].isna().any():
        missing = snapshot.loc[snapshot["recent_win_rate"].isna(), "team_display_name"].tolist()
        raise ValueError(f"Recent win rate ausente para os seguintes times: {missing}")
    if include_elo and "elo_rating" in snapshot.columns and snapshot["elo_rating"].isna().any():
        missing = snapshot.loc[snapshot["elo_rating"].isna(), "team_display_name"].tolist()
        raise ValueError(f"Elo rating ausente para os seguintes times: {missing}")
    if "elo_decay_rating" in snapshot.columns and snapshot["elo_decay_rating"].isna().any():
        missing = snapshot.loc[snapshot["elo_decay_rating"].isna(), "team_display_name"].tolist()
        raise ValueError(f"Elo decay rating ausente para os seguintes times: {missing}")


def validate_season_snapshots(snapshot_by_season: pd.DataFrame, expected_team_count: int = 50) -> None:
    if snapshot_by_season.empty:
        raise ValueError("team_snapshot_by_season.csv foi gerado vazio.")
    season_counts = snapshot_by_season.groupby("season_label", sort=False)["team_key"].nunique()
    invalid = season_counts.loc[season_counts != expected_team_count]
    if not invalid.empty:
        raise ValueError(
            "team_snapshot_by_season.csv nao contem a quantidade esperada de times em todas as seasons. "
            f"Encontrado: {invalid.to_dict()}"
        )
    metric_columns = list(PLAYER_AGGREGATIONS.values())
    if snapshot_by_season[metric_columns].isna().any().any():
        raise ValueError("Ha valores ausentes nas metricas agregadas de team_snapshot_by_season.csv.")


def validate_match_features(match_features: pd.DataFrame, feature_columns: list[str] | None = None) -> None:
    if match_features.empty:
        raise ValueError("match_feature_differences.csv foi gerado vazio.")
    unique_targets = set(match_features["win_target"].dropna().astype(int).unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"win_target contem valores invalidos: {sorted(unique_targets)}")
    resolved_feature_columns = feature_columns or FEATURE_COLUMNS
    missing_feature_columns = [column for column in resolved_feature_columns if column not in match_features.columns]
    if missing_feature_columns:
        raise ValueError(f"Features obrigatorias ausentes: {missing_feature_columns}")
    if match_features[resolved_feature_columns].isna().any().any():
        raise ValueError("Ha valores ausentes nas features finais de match_feature_differences.csv.")
    if "match_date" in match_features.columns and match_features["match_date"].isna().any():
        raise ValueError("Ha valores ausentes em match_date no dataset supervisionado.")
    if "season_usage" in match_features.columns:
        invalid_usage = set(match_features["season_usage"].unique()) - {TRAIN_SEASON_USAGE, HOLDOUT_SEASON_USAGE}
        if invalid_usage:
            raise ValueError(f"Season_Usage invalido no dataset supervisionado: {sorted(invalid_usage)}")


def compute_snapshot_recent_win_rate(

    matches: pd.DataFrame,
    recent_match_count: int = DEFAULT_RECENT_MATCH_COUNT,
) -> pd.Series:
    if "Match_Date" in matches.columns and matches["Match_Date"].notna().all():
        recent_matches = (
            matches.sort_values(["team_key", "Match_Date"], ascending=[True, False], kind="stable")
            .groupby("team_key", sort=False)
            .head(recent_match_count)
        )
    else:
        recent_matches = matches.groupby("team_key", sort=False).head(recent_match_count)

    return recent_matches.groupby("team_key", sort=False)["Win"].mean().rename("recent_win_rate")


def attach_pre_match_recent_win_rates(
    matches: pd.DataFrame,
    recent_match_count: int = DEFAULT_RECENT_MATCH_COUNT,
    neutral_rate: float = NEUTRAL_RECENT_WIN_RATE,
) -> pd.DataFrame:
    enriched = matches.copy()

    if "Match_Date" not in enriched.columns or enriched["Match_Date"].isna().any():
        snapshot_recent_rates = compute_snapshot_recent_win_rate(enriched, recent_match_count=recent_match_count)
        enriched["team_recent_win_rate_pre_match"] = enriched["team_key"].map(snapshot_recent_rates).fillna(
            neutral_rate
        )
        enriched["opponent_recent_win_rate_pre_match"] = enriched["opponent_key"].map(snapshot_recent_rates).fillna(
            neutral_rate
        )
        return enriched

    enriched["_original_index"] = range(len(enriched))
    enriched = enriched.sort_values(["team_key", "Match_Date", "_original_index"], kind="stable").reset_index(
        drop=True
    )

    team_prior_rates = pd.Series(index=enriched.index, dtype=float)
    for _, team_group in enriched.groupby("team_key", sort=False):
        history: deque[int] = deque(maxlen=recent_match_count)
        for _, date_group in team_group.groupby("Match_Date", sort=False):
            prior_rate = float(sum(history) / len(history)) if history else neutral_rate
            team_prior_rates.loc[date_group.index] = prior_rate
            history.extend(int(value) for value in date_group["Win"].tolist())

    enriched["team_recent_win_rate_pre_match"] = team_prior_rates.round(3)

    opponent_history = (
        enriched[["team_key", "Match_Date", "team_recent_win_rate_pre_match"]]
        .drop_duplicates(subset=["team_key", "Match_Date"], keep="first")
        .sort_values(["Match_Date", "team_key"], kind="stable")
    )
    opponent_targets = (
        enriched[["_original_index", "opponent_key", "Match_Date"]]
        .rename(columns={"opponent_key": "team_key"})
        .sort_values(["Match_Date", "team_key", "_original_index"], kind="stable")
    )
    opponent_rates = pd.merge_asof(
        opponent_targets,
        opponent_history,
        on="Match_Date",
        by="team_key",
        direction="backward",
        allow_exact_matches=True,
    )
    opponent_rate_lookup = opponent_rates.set_index("_original_index")["team_recent_win_rate_pre_match"]
    enriched["opponent_recent_win_rate_pre_match"] = enriched["_original_index"].map(opponent_rate_lookup)
    enriched["opponent_recent_win_rate_pre_match"] = (
        enriched["opponent_recent_win_rate_pre_match"].fillna(neutral_rate).round(3)
    )

    return enriched.sort_values("_original_index", kind="stable").drop(columns=["_original_index"])


def prepare_actual_matches(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = matches.copy()
    working["_original_index"] = range(len(working))
    working["pair_key"] = working.apply(
        lambda row: "|".join(sorted([str(row["team_key"]), str(row["opponent_key"])])),
        axis=1,
    )
    working["winner_key"] = working.apply(
        lambda row: row["team_key"] if int(row["Win"]) == 1 else row["opponent_key"],
        axis=1,
    )
    working["loser_key"] = working.apply(
        lambda row: row["opponent_key"] if int(row["Win"]) == 1 else row["team_key"],
        axis=1,
    )
    working["winner_score"] = working[["Team_Score", "Opponent_Score"]].max(axis=1)
    working["loser_score"] = working[["Team_Score", "Opponent_Score"]].min(axis=1)
    working["ordered_pair"] = working["team_key"] + "|" + working["opponent_key"]

    if "Match_Date" in working.columns and working["Match_Date"].notna().all():
        working["match_order_date"] = working["Match_Date"]
    else:
        working["match_order_date"] = pd.Timestamp("1970-01-01")

    group_columns = [
        "match_order_date",
        "pair_key",
        "winner_key",
        "loser_key",
        "winner_score",
        "loser_score",
    ]
    working["_unique_orientations"] = working.groupby(group_columns)["ordered_pair"].transform("nunique")
    working["_orientation_instance"] = working.groupby(group_columns + ["ordered_pair"]).cumcount()
    working["_single_instance"] = working.groupby(group_columns).cumcount()
    working["_match_instance"] = working["_single_instance"]
    mirrored_mask = working["_unique_orientations"] > 1
    working.loc[mirrored_mask, "_match_instance"] = working.loc[mirrored_mask, "_orientation_instance"]

    actual_matches = (
        working.sort_values(["match_order_date", "_original_index"], kind="stable")
        .drop_duplicates(group_columns + ["_match_instance"], keep="first")
        .copy()
        .reset_index(drop=True)
    )
    actual_matches["actual_match_id"] = range(len(actual_matches))
    actual_matches["team_a_key"] = actual_matches["winner_key"]
    actual_matches["team_b_key"] = actual_matches["loser_key"]

    mapping_columns = group_columns + ["_match_instance", "actual_match_id"]
    working = working.merge(actual_matches[mapping_columns], on=group_columns + ["_match_instance"], how="left")
    return working, actual_matches


def attach_pre_match_elo_ratings(
    matches: pd.DataFrame,
    initial_rating: float = DEFAULT_ELO_INITIAL_RATING,
    k_factor: float = DEFAULT_ELO_K_FACTOR,
    scale: float = DEFAULT_ELO_SCALE,
    decay_rate: float = 0.0,
    team_pre_column: str = "team_elo_pre_match",
    opponent_pre_column: str = "opponent_elo_pre_match",
    final_series_name: str = "elo_rating",
) -> tuple[pd.DataFrame, pd.Series]:
    working, actual_matches = prepare_actual_matches(matches)
    actual_matches = actual_matches.sort_values(["match_order_date", "_original_index"], kind="stable").reset_index(
        drop=True
    )

    ratings: dict[str, float] = {}
    last_match_dates: dict[str, pd.Timestamp] = {}
    winner_pre_ratings: list[float] = []
    loser_pre_ratings: list[float] = []
    winner_post_ratings: list[float] = []
    loser_post_ratings: list[float] = []

    for row in actual_matches.itertuples(index=False):
        winner_key = str(row.winner_key)
        loser_key = str(row.loser_key)
        match_date = pd.Timestamp(row.match_order_date)

        winner_pre = float(ratings.get(winner_key, initial_rating))
        loser_pre = float(ratings.get(loser_key, initial_rating))

        if decay_rate > 0:
            if winner_key in last_match_dates:
                days_since_winner = max(0, int((match_date - last_match_dates[winner_key]).days))
                if days_since_winner > 0:
                    decay = math.exp(-float(decay_rate) * days_since_winner / 30.0)
                    winner_pre = float(initial_rating + (winner_pre - initial_rating) * decay)
            if loser_key in last_match_dates:
                days_since_loser = max(0, int((match_date - last_match_dates[loser_key]).days))
                if days_since_loser > 0:
                    decay = math.exp(-float(decay_rate) * days_since_loser / 30.0)
                    loser_pre = float(initial_rating + (loser_pre - initial_rating) * decay)

        expected_winner = 1.0 / (1.0 + 10.0 ** ((loser_pre - winner_pre) / scale))
        rating_delta = k_factor * (1.0 - expected_winner)

        winner_post = winner_pre + rating_delta
        loser_post = loser_pre - rating_delta
        ratings[winner_key] = winner_post
        ratings[loser_key] = loser_post

        winner_pre_ratings.append(round(winner_pre, 3))
        loser_pre_ratings.append(round(loser_pre, 3))
        winner_post_ratings.append(round(winner_post, 3))
        loser_post_ratings.append(round(loser_post, 3))
        last_match_dates[winner_key] = match_date
        last_match_dates[loser_key] = match_date

    actual_matches["winner_elo_pre_match"] = winner_pre_ratings
    actual_matches["loser_elo_pre_match"] = loser_pre_ratings
    actual_matches["winner_elo_post_match"] = winner_post_ratings
    actual_matches["loser_elo_post_match"] = loser_post_ratings

    elo_columns = [
        "actual_match_id",
        "winner_key",
        "loser_key",
        "winner_elo_pre_match",
        "loser_elo_pre_match",
    ]
    working = working.merge(actual_matches[elo_columns], on=["actual_match_id", "winner_key", "loser_key"], how="left")
    team_is_winner = working["team_key"] == working["winner_key"]
    working[team_pre_column] = working["winner_elo_pre_match"].where(team_is_winner, working["loser_elo_pre_match"])
    working[opponent_pre_column] = working["loser_elo_pre_match"].where(
        team_is_winner,
        working["winner_elo_pre_match"],
    )
    working[team_pre_column] = working[team_pre_column].fillna(initial_rating).round(3)
    working[opponent_pre_column] = working[opponent_pre_column].fillna(initial_rating).round(3)

    final_ratings = pd.Series(ratings, name=final_series_name, dtype=float)
    return working.sort_values("_original_index", kind="stable").drop(
        columns=[
            "_original_index",
            "pair_key",
            "winner_key",
            "loser_key",
            "winner_score",
            "loser_score",
            "ordered_pair",
            "match_order_date",
            "_unique_orientations",
            "_orientation_instance",
            "_single_instance",
            "_match_instance",
            "winner_elo_pre_match",
            "loser_elo_pre_match",
        ],
        errors="ignore",
    ), final_ratings


def attach_pre_match_h2h_features(
    matches: pd.DataFrame,
    min_games: int = DEFAULT_H2H_MIN_GAMES,
) -> pd.DataFrame:
    working, actual_matches = prepare_actual_matches(matches)
    actual_matches = actual_matches.sort_values(["match_order_date", "_original_index"], kind="stable").reset_index(
        drop=True
    )

    pair_history: dict[tuple[str, str], dict[str, int]] = {}
    pair_team_a_keys: list[str] = []
    pair_team_b_keys: list[str] = []
    pair_team_a_wins_pre: list[int] = []
    pair_team_b_wins_pre: list[int] = []
    pair_total_games_pre: list[int] = []

    for row in actual_matches.itertuples(index=False):
        winner_key = str(row.winner_key)
        loser_key = str(row.loser_key)
        pair_team_a_key, pair_team_b_key = sorted([winner_key, loser_key])
        pair_key = (pair_team_a_key, pair_team_b_key)
        history = pair_history.get(pair_key, {pair_team_a_key: 0, pair_team_b_key: 0})

        wins_a = int(history.get(pair_team_a_key, 0))
        wins_b = int(history.get(pair_team_b_key, 0))
        total = wins_a + wins_b

        pair_team_a_keys.append(pair_team_a_key)
        pair_team_b_keys.append(pair_team_b_key)
        pair_team_a_wins_pre.append(wins_a)
        pair_team_b_wins_pre.append(wins_b)
        pair_total_games_pre.append(total)

        if pair_key not in pair_history:
            pair_history[pair_key] = {pair_team_a_key: 0, pair_team_b_key: 0}
        pair_history[pair_key][winner_key] = pair_history[pair_key].get(winner_key, 0) + 1

    actual_matches["pair_team_a_key"] = pair_team_a_keys
    actual_matches["pair_team_b_key"] = pair_team_b_keys
    actual_matches["pair_team_a_wins_pre"] = pair_team_a_wins_pre
    actual_matches["pair_team_b_wins_pre"] = pair_team_b_wins_pre
    actual_matches["pair_total_games_pre"] = pair_total_games_pre

    h2h_columns = [
        "actual_match_id",
        "pair_team_a_key",
        "pair_team_b_key",
        "pair_team_a_wins_pre",
        "pair_team_b_wins_pre",
        "pair_total_games_pre",
    ]
    working = working.merge(actual_matches[h2h_columns], on="actual_match_id", how="left")

    team_is_pair_a = working["team_key"] == working["pair_team_a_key"]
    working["team_h2h_wins_pre"] = working["pair_team_a_wins_pre"].where(team_is_pair_a, working["pair_team_b_wins_pre"])
    working["opponent_h2h_wins_pre"] = working["pair_team_b_wins_pre"].where(
        team_is_pair_a,
        working["pair_team_a_wins_pre"],
    )
    working["pair_total_games_pre"] = working["pair_total_games_pre"].fillna(0).astype(int)
    sufficient_history = working["pair_total_games_pre"] >= int(min_games)
    team_h2h_wins_pre = pd.to_numeric(working["team_h2h_wins_pre"], errors="coerce").fillna(0.0)
    opponent_h2h_wins_pre = pd.to_numeric(working["opponent_h2h_wins_pre"], errors="coerce").fillna(0.0)
    safe_total = working["pair_total_games_pre"].replace(0, np.nan).astype(float)
    team_h2h_rates = np.divide(team_h2h_wins_pre.to_numpy(), safe_total.to_numpy(), out=np.full(len(working), 0.5), where=~np.isnan(safe_total.to_numpy()))
    opponent_h2h_rates = np.divide(opponent_h2h_wins_pre.to_numpy(), safe_total.to_numpy(), out=np.full(len(working), 0.5), where=~np.isnan(safe_total.to_numpy()))
    working["team_h2h_win_rate_pre_match"] = np.where(sufficient_history.to_numpy(), team_h2h_rates, 0.5)
    working["opponent_h2h_win_rate_pre_match"] = np.where(sufficient_history.to_numpy(), opponent_h2h_rates, 0.5)
    working[H2H_WIN_RATE_FEATURE_COLUMN] = (working["team_h2h_win_rate_pre_match"] - 0.5).round(3)
    working[H2H_HISTORY_FEATURE_COLUMN] = sufficient_history.astype(int)

    return working.sort_values("_original_index", kind="stable").drop(
        columns=[
            "_original_index",
            "pair_key",
            "winner_key",
            "loser_key",
            "winner_score",
            "loser_score",
            "ordered_pair",
            "match_order_date",
            "_unique_orientations",
            "_orientation_instance",
            "_single_instance",
            "_match_instance",
            "actual_match_id",
            "pair_team_a_key",
            "pair_team_b_key",
            "pair_team_a_wins_pre",
            "pair_team_b_wins_pre",
            "team_h2h_wins_pre",
            "opponent_h2h_wins_pre",
            "pair_total_games_pre",
            "team_h2h_win_rate_pre_match",
            "opponent_h2h_win_rate_pre_match",
        ],
        errors="ignore",
    )


def aggregate_player_snapshot(players: pd.DataFrame, grouping_columns: list[str]) -> pd.DataFrame:
    aggregations = {
        target_column: (source_column, "mean")
        for source_column, target_column in PLAYER_AGGREGATIONS.items()
    }
    aggregated = players.groupby(grouping_columns, sort=False, as_index=False).agg(**aggregations)
    counts = players.groupby(grouping_columns, sort=False, as_index=False).size().rename(columns={"size": "players_count"})
    aggregated = aggregated.merge(counts, on=grouping_columns, how="left")
    aggregated["players_count"] = aggregated["players_count"].fillna(0).astype(int)
    aggregated["missing_players"] = (5 - aggregated["players_count"]).clip(lower=0).astype(int)
    for column, decimals in ROUNDING_BY_SNAPSHOT_COLUMN.items():
        if column in aggregated.columns:
            aggregated[column] = aggregated[column].round(decimals)
    return aggregated


def build_team_snapshots_by_season(players: pd.DataFrame) -> pd.DataFrame:
    aggregated = aggregate_player_snapshot(
        players,
        ["season_label", "season_usage", "team_key", "team_slug", "team_display_name"],
    )
    return aggregated[SEASON_SNAPSHOT_COLUMNS].sort_values(["season_label", "team_display_name"], kind="stable").reset_index(drop=True)


def build_team_snapshot(
    players: pd.DataFrame,
    matches: pd.DataFrame,
    recent_match_count: int = DEFAULT_RECENT_MATCH_COUNT,
    elo_ratings: pd.Series | None = None,
    elo_decay_ratings: pd.Series | None = None,
    initial_elo_rating: float = DEFAULT_ELO_INITIAL_RATING,
    initial_elo_decay_rating: float = DEFAULT_ELO_INITIAL_RATING,
) -> pd.DataFrame:
    available_labels = sorted(players["season_label"].dropna().unique().tolist(), key=season_sort_key)
    latest_label = available_labels[-1] if available_labels else "global"
    latest_players = players.loc[players["season_label"] == latest_label].copy()

    aggregated = aggregate_player_snapshot(
        latest_players,
        ["team_key", "team_slug", "team_display_name"],
    )

    recent_win_rate = compute_snapshot_recent_win_rate(matches, recent_match_count=recent_match_count)
    aggregated = aggregated.merge(recent_win_rate, on="team_key", how="left")
    aggregated["recent_win_rate"] = aggregated["recent_win_rate"].fillna(NEUTRAL_RECENT_WIN_RATE).round(3)

    if elo_ratings is not None:
        aggregated["elo_rating"] = (
            aggregated["team_key"].map(elo_ratings).fillna(float(initial_elo_rating)).round(3)
        )
    if elo_decay_ratings is not None:
        aggregated["elo_decay_rating"] = (
            aggregated["team_key"].map(elo_decay_ratings).fillna(float(initial_elo_decay_rating)).round(3)
        )

    return aggregated[[
        "team_key",
        *get_snapshot_columns(include_elo=elo_ratings is not None, include_elo_decay=elo_decay_ratings is not None),
    ]].sort_values(
        "team_display_name",
        kind="stable",
    ).reset_index(drop=True)


def build_match_feature_differences(
    matches: pd.DataFrame,
    snapshot_by_season: pd.DataFrame,
    recent_match_count: int = DEFAULT_RECENT_MATCH_COUNT,
    deduplicate_mirrors: bool = True,
    include_elo: bool = False,
    include_elo_decay: bool = False,
    include_h2h: bool = False,
    enriched_matches: pd.DataFrame | None = None,
) -> pd.DataFrame:
    working_matches = enriched_matches
    if working_matches is None:
        working_matches = attach_pre_match_recent_win_rates(matches, recent_match_count=recent_match_count)

    valid_matches = working_matches.copy()
    valid_matches["season_label"] = valid_matches["Season_Label"].astype(str).str.strip().str.lower()
    valid_matches["season_usage"] = valid_matches["Season_Usage"].astype(str).str.strip().str.lower()
    valid_matches = valid_matches.loc[valid_matches["season_usage"].isin({TRAIN_SEASON_USAGE, HOLDOUT_SEASON_USAGE})].copy()

    snapshot_columns = ["players_count", "missing_players", *PLAYER_AGGREGATIONS.values()]
    team_snapshot = snapshot_by_season[["season_label", "team_key", *snapshot_columns]].rename(
        columns={column: f"team_{column}" for column in snapshot_columns}
    )
    opponent_snapshot = snapshot_by_season[["season_label", "team_key", *snapshot_columns]].rename(
        columns={"team_key": "opponent_key", **{column: f"opponent_{column}" for column in snapshot_columns}},
    )

    valid_matches = valid_matches.merge(team_snapshot, on=["season_label", "team_key"], how="inner")
    valid_matches = valid_matches.merge(opponent_snapshot, on=["season_label", "opponent_key"], how="inner")

    if deduplicate_mirrors:
        valid_matches = valid_matches.loc[valid_matches["team_key"] < valid_matches["opponent_key"]].copy()

    if "Match_Date" in valid_matches.columns:
        valid_matches = valid_matches.sort_values(["Match_Date", "season_label", "team_key", "opponent_key"], kind="stable")

    valid_matches["team_name"] = valid_matches["Team_Name"]
    valid_matches["opponent_name"] = valid_matches["Opponent"]
    valid_matches["win_target"] = valid_matches["Win"].astype(int)
    valid_matches["diff_recent_win_rate"] = (
        valid_matches["team_recent_win_rate_pre_match"].to_numpy()
        - valid_matches["opponent_recent_win_rate_pre_match"].to_numpy()
    ).round(3)

    for source_column, snapshot_column in PLAYER_AGGREGATIONS.items():
        valid_matches[f"diff_{snapshot_column}"] = (
            valid_matches[f"team_{snapshot_column}"].to_numpy() - valid_matches[f"opponent_{snapshot_column}"].to_numpy()
        ).round(ROUNDING_BY_SNAPSHOT_COLUMN.get(snapshot_column, 3))

    valid_matches["diff_players_count"] = (
        valid_matches["team_players_count"].to_numpy() - valid_matches["opponent_players_count"].to_numpy()
    ).astype(int)
    valid_matches["diff_missing_players"] = (
        valid_matches["team_missing_players"].to_numpy() - valid_matches["opponent_missing_players"].to_numpy()
    ).astype(int)

    if include_elo:
        valid_matches[ELO_FEATURE_COLUMN] = (
            valid_matches["team_elo_pre_match"].to_numpy() - valid_matches["opponent_elo_pre_match"].to_numpy()
        ).round(3)
    if include_elo_decay:
        valid_matches[ELO_DECAY_FEATURE_COLUMN] = (
            valid_matches["team_elo_decay_pre_match"].to_numpy() - valid_matches["opponent_elo_decay_pre_match"].to_numpy()
        ).round(3)
    if include_h2h:
        valid_matches[H2H_WIN_RATE_FEATURE_COLUMN] = valid_matches[H2H_WIN_RATE_FEATURE_COLUMN].fillna(0.0).round(3)
        valid_matches[H2H_HISTORY_FEATURE_COLUMN] = valid_matches[H2H_HISTORY_FEATURE_COLUMN].fillna(0).astype(int)

    best_of = valid_matches.get("Best_Of", pd.Series("", index=valid_matches.index)).fillna("").astype(str).str.strip().str.lower()
    lan_online = valid_matches.get("LAN_Online", pd.Series("", index=valid_matches.index)).fillna("").astype(str).str.strip().str.upper()
    valid_matches["context_is_lan"] = (lan_online == "LAN").astype(int)
    valid_matches["context_is_bo1"] = (best_of == "bo1").astype(int)
    valid_matches["context_is_bo3"] = (best_of == "bo3").astype(int)
    valid_matches["context_is_bo5"] = (best_of == "bo5").astype(int)

    feature_columns = get_extended_feature_columns(
        include_elo=include_elo,
        include_elo_decay=include_elo_decay,
        include_h2h=include_h2h,
    )
    output_columns = ["season_label", "season_usage", "team_name", "opponent_name", "win_target", *feature_columns]
    if "actual_match_id" in valid_matches.columns:
        output_columns = ["actual_match_id", *output_columns]
    if "Match_Date" in valid_matches.columns:
        valid_matches["match_date"] = valid_matches["Match_Date"].dt.strftime("%Y-%m-%d")
        output_columns = ["match_date", *output_columns]

    return valid_matches[output_columns].reset_index(drop=True)


def build_match_row(
    snapshot: pd.DataFrame,
    team_a_slug: str,
    team_b_slug: str,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    row_a = snapshot.loc[snapshot["team_slug"] == team_a_slug]
    row_b = snapshot.loc[snapshot["team_slug"] == team_b_slug]
    if row_a.empty or row_b.empty:
        raise KeyError("Um ou ambos os times nao foram encontrados no snapshot.")

    requested_columns = feature_cols or FEATURE_COLUMNS
    row_a = row_a.iloc[0]
    row_b = row_b.iloc[0]
    feature_map: dict[str, float | int] = {}
    special_differences = {
        "diff_recent_win_rate": "recent_win_rate",
        "diff_players_count": "players_count",
        "diff_missing_players": "missing_players",
    }
    snapshot_metrics = set(PLAYER_AGGREGATIONS.values())

    for feature in requested_columns:
        if feature in special_differences:
            column = special_differences[feature]
            feature_map[feature] = row_a[column] - row_b[column]
        elif feature.startswith("diff_") and feature.removeprefix("diff_") in snapshot_metrics:
            column = feature.removeprefix("diff_")
            feature_map[feature] = row_a[column] - row_b[column]

    if ELO_FEATURE_COLUMN in requested_columns:
        if "elo_rating" not in row_a.index or "elo_rating" not in row_b.index:
            raise KeyError("Snapshot nao contem elo_rating, mas o modelo exige diff_elo_pre_match.")
        feature_map[ELO_FEATURE_COLUMN] = row_a["elo_rating"] - row_b["elo_rating"]
    if ELO_DECAY_FEATURE_COLUMN in requested_columns:
        if "elo_decay_rating" not in row_a.index or "elo_decay_rating" not in row_b.index:
            raise KeyError("Snapshot nao contem elo_decay_rating, mas o modelo exige diff_elo_decay_pre_match.")
        feature_map[ELO_DECAY_FEATURE_COLUMN] = row_a["elo_decay_rating"] - row_b["elo_decay_rating"]
    if H2H_WIN_RATE_FEATURE_COLUMN in requested_columns:
        feature_map[H2H_WIN_RATE_FEATURE_COLUMN] = 0.0
    if H2H_HISTORY_FEATURE_COLUMN in requested_columns:
        feature_map[H2H_HISTORY_FEATURE_COLUMN] = 0
    context_defaults = {
        "context_is_lan": 0,
        "context_is_bo1": 0,
        "context_is_bo3": 1,
        "context_is_bo5": 0,
    }
    for feature, value in context_defaults.items():
        if feature in requested_columns:
            feature_map[feature] = value

    missing = [feature for feature in requested_columns if feature not in feature_map]
    if missing:
        raise KeyError(f"Nao foi possivel construir os atributos: {missing}")
    return pd.DataFrame([feature_map], columns=requested_columns)


def load_feature_columns(features_path: Path) -> list[str]:
    with features_path.open("r", encoding="utf-8") as file:
        feature_cols = json.load(file)
    if not isinstance(feature_cols, list) or not feature_cols:
        raise ValueError("Arquivo de features invalido. Nenhuma coluna foi encontrada.")
    allowed = set(BASE_FEATURE_COLUMNS)
    allowed.update([ELO_FEATURE_COLUMN, ELO_DECAY_FEATURE_COLUMN, H2H_WIN_RATE_FEATURE_COLUMN, H2H_HISTORY_FEATURE_COLUMN])
    invalid = [column for column in feature_cols if column not in allowed]
    if invalid:
        raise ValueError(f"Arquivo de features invalido. Colunas desconhecidas: {invalid}")
    return feature_cols


def load_model_registry(registry_path: Path) -> dict[str, object]:
    if not registry_path.exists():
        return {
            "primary_model": DEFAULT_PRIMARY_MODEL,
            "available_models": [DEFAULT_PRIMARY_MODEL],
        }
    with registry_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_model_name(registry: dict[str, object], selector: str = "primary") -> str:
    available_models = set(registry.get("available_models", []))
    if selector == "best":
        return str(registry.get("best_model_by_test_roc_auc") or registry.get("primary_model") or DEFAULT_PRIMARY_MODEL)
    if selector == "best_test":
        return str(registry.get("best_model_by_test_roc_auc") or registry.get("primary_model") or DEFAULT_PRIMARY_MODEL)
    if selector == "best_cv":
        return str(registry.get("best_model_by_cv_roc_auc") or registry.get("primary_model") or DEFAULT_PRIMARY_MODEL)
    if selector == "primary":
        return str(registry.get("primary_model") or DEFAULT_PRIMARY_MODEL)
    if available_models and selector not in available_models:
        available_text = ", ".join(sorted(available_models))
        raise KeyError(f"Modelo '{selector}' nao consta no registro. Modelos disponiveis: {available_text}")
    return selector


def save_processed_datasets(
    root: Path,
    players_path: Path | None = None,
    matches_path: Path | None = None,
    processed_dir: Path | None = None,
    recent_match_count: int = DEFAULT_RECENT_MATCH_COUNT,
    include_elo: bool = False,
    include_elo_decay: bool = False,
    include_h2h: bool = False,
    elo_initial_rating: float = DEFAULT_ELO_INITIAL_RATING,
    elo_k_factor: float = DEFAULT_ELO_K_FACTOR,
    elo_scale: float = DEFAULT_ELO_SCALE,
    elo_decay_rate: float = DEFAULT_ELO_DECAY_RATE,
    h2h_min_games: int = DEFAULT_H2H_MIN_GAMES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Constroi os CSVs processados principais do projeto.

    O fluxo oficial atual usa:
    - recent win rate pre-jogo
    - Elo classico pre-jogo
    - snapshots por season

    Os caminhos de H2H e Elo com decay permanecem acessiveis por parametro,
    mas sao experimentais e nao entram no modelo principal final.
    """
    resolved_players_path = players_path or (root / "data" / "raw" / "core")
    resolved_matches_path = matches_path or (root / "data" / "raw" / "core" / "matches_top50_dated.csv")
    resolved_processed_dir = processed_dir or (root / "data" / "processed")

    players = load_players(resolved_players_path)
    matches = load_matches(resolved_matches_path)
    validate_players_structure(players)
    validate_matches_structure(matches)

    resolved_processed_dir.mkdir(parents=True, exist_ok=True)

    enriched_matches = attach_pre_match_recent_win_rates(matches, recent_match_count=recent_match_count)
    elo_ratings: pd.Series | None = None
    elo_decay_ratings: pd.Series | None = None
    if include_elo:
        enriched_matches, elo_ratings = attach_pre_match_elo_ratings(
            enriched_matches,
            initial_rating=elo_initial_rating,
            k_factor=elo_k_factor,
            scale=elo_scale,
        )
    if include_elo_decay:
        enriched_matches, elo_decay_ratings = attach_pre_match_elo_ratings(
            enriched_matches,
            initial_rating=elo_initial_rating,
            k_factor=elo_k_factor,
            scale=elo_scale,
            decay_rate=elo_decay_rate,
            team_pre_column="team_elo_decay_pre_match",
            opponent_pre_column="opponent_elo_decay_pre_match",
            final_series_name="elo_decay_rating",
        )
    if include_h2h:
        enriched_matches = attach_pre_match_h2h_features(enriched_matches, min_games=h2h_min_games)

    feature_columns = get_extended_feature_columns(
        include_elo=include_elo,
        include_elo_decay=include_elo_decay,
        include_h2h=include_h2h,
    )
    snapshot_columns = get_snapshot_columns(include_elo=include_elo, include_elo_decay=include_elo_decay)

    snapshot_by_season = build_team_snapshots_by_season(players)
    snapshot = build_team_snapshot(
        players,
        matches,
        recent_match_count=recent_match_count,
        elo_ratings=elo_ratings,
        elo_decay_ratings=elo_decay_ratings,
        initial_elo_rating=elo_initial_rating,
        initial_elo_decay_rating=elo_initial_rating,
    )
    match_features = build_match_feature_differences(
        matches,
        snapshot_by_season,
        recent_match_count=recent_match_count,
        deduplicate_mirrors=True,
        include_elo=include_elo,
        include_elo_decay=include_elo_decay,
        include_h2h=include_h2h,
        enriched_matches=enriched_matches,
    )
    validate_season_snapshots(snapshot_by_season)
    validate_snapshot(snapshot, include_elo=include_elo)
    validate_match_features(match_features, feature_columns=feature_columns)

    snapshot_by_season.to_csv(resolved_processed_dir / "team_snapshot_by_season.csv", index=False)
    snapshot[["team_key", *snapshot_columns]].to_csv(resolved_processed_dir / "team_snapshot.csv", index=False)
    match_features.to_csv(resolved_processed_dir / "match_feature_differences.csv", index=False)
    save_visual_processed_csvs(
        root,
        snapshot_by_season,
        snapshot[["team_key", *snapshot_columns]],
        match_features,
    )

    metadata: dict[str, object] = {
        "players_path": str(resolved_players_path),
        "matches_path": str(resolved_matches_path),
        "recent_match_count": int(recent_match_count),
        "include_elo": bool(include_elo),
        "include_elo_decay": bool(include_elo_decay),
        "include_h2h": bool(include_h2h),
        "elo_initial_rating": float(elo_initial_rating),
        "elo_k_factor": float(elo_k_factor),
        "elo_scale": float(elo_scale),
        "elo_decay_rate": float(elo_decay_rate),
        "h2h_min_games": int(h2h_min_games),
        "snapshot_by_season_rows": int(len(snapshot_by_season)),
        "snapshot_rows": int(len(snapshot)),
        "feature_rows": int(len(match_features)),
        "feature_columns": feature_columns,
        "season_labels": sorted(snapshot_by_season["season_label"].unique().tolist(), key=season_sort_key),
        "train_rows": int((match_features["season_usage"] == TRAIN_SEASON_USAGE).sum()),
        "holdout_rows": int((match_features["season_usage"] == HOLDOUT_SEASON_USAGE).sum()),
    }
    if "match_date" in match_features.columns:
        metadata["match_date_min"] = str(match_features["match_date"].min())
        metadata["match_date_max"] = str(match_features["match_date"].max())

    with (resolved_processed_dir / "dataset_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    return snapshot, match_features
