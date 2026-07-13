from __future__ import annotations

import math

from predict_match import list_available_teams, predict_match_probability


def test_demo_lists_only_synthetic_teams() -> None:
    teams = list_available_teams(snapshot_mode="demo")
    assert [team["display_name"] for team in teams] == [
        "Equipe Aurora",
        "Equipe Horizonte",
    ]


def test_demo_prediction_is_symmetric_probability() -> None:
    direct = predict_match_probability(
        "Equipe Aurora", "Equipe Horizonte", snapshot_mode="demo"
    )
    reverse = predict_match_probability(
        "Equipe Horizonte", "Equipe Aurora", snapshot_mode="demo"
    )

    assert direct["snapshot_mode"] == "demo"
    assert direct["probability_symmetrized"] is True
    assert 0.0 <= direct["probability_team_a"] <= 1.0
    assert math.isclose(
        direct["probability_team_a"] + direct["probability_team_b"],
        1.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        direct["probability_team_a"],
        reverse["probability_team_b"],
        abs_tol=1e-12,
    )
