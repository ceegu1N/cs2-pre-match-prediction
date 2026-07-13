from __future__ import annotations

import json
import subprocess
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_forbidden_private_content_is_absent() -> None:
    forbidden = [
        ROOT / "data" / "raw",
        ROOT / "data" / "processed",
        ROOT / "data" / "private",
        ROOT / "collect_matches_dated.py",
        ROOT / "collect_player_stats_hltv.py",
    ]
    assert all(not path.exists() for path in forbidden)


def test_collection_dependencies_and_imports_are_absent() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "selenium" not in requirements
    assert "beautifulsoup" not in requirements

    forbidden_imports = ("import selenium", "from selenium", "from bs4", "import bs4")
    for path in [*ROOT.glob("*.py"), *ROOT.glob("*.pyw"), *ROOT.rglob("src/*.py")]:
        source = path.read_text(encoding="utf-8", errors="ignore").lower()
        assert not any(item in source for item in forbidden_imports), path


def test_json_files_are_valid() -> None:
    for path in ROOT.rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))


def test_model_artifact_loads() -> None:
    model = joblib.load(ROOT / "models" / "logistic_regression.joblib")
    assert list(model.named_steps) == ["imputer", "scaler", "clf"]


def test_demo_contains_only_two_fictional_teams() -> None:
    sample = pd.read_csv(ROOT / "data" / "sample" / "team_snapshot_synthetic.csv")
    assert sample["team_display_name"].tolist() == ["Equipe Aurora", "Equipe Horizonte"]
    assert len(sample) == 2


def test_no_personal_local_paths_in_text_files() -> None:
    suffixes = {".py", ".pyw", ".md", ".txt", ".json", ".csv", ".yml", ".yaml"}
    forbidden = (
        "c:" + "\\users\\",
        "c:" + "\\tccteste0",
        "c:" + "\\eutccfim",
    )
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8").split("\0")
    for relative_path in tracked:
        path = ROOT / relative_path
        if relative_path and path.suffix.lower() in suffixes:
            content = path.read_text(encoding="utf-8", errors="ignore").lower()
            assert not any(value in content for value in forbidden), path


def test_data_notice_does_not_offer_private_assets() -> None:
    notice = (ROOT / "DATA_NOTICE.md").read_text(encoding="utf-8").lower()
    assert "dados brutos e coletores não são fornecidos mediante solicitação" in notice
    assert "data/sample/" in notice


def test_prospective_reports_do_not_publish_match_level_predictions() -> None:
    prospect_dir = ROOT / "reports" / "prospectiva"
    assert not (prospect_dir / "analise_final_prospectiva_3_eventos_20260524.json").exists()
    assert not (prospect_dir / "analise_final_prospectiva_3_eventos_20260524.md").exists()

    forbidden_keys = {"high_confidence_errors_80plus", "partida", "favorito", "vencedor"}
    for path in prospect_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        pending = [payload]
        while pending:
            current = pending.pop()
            if isinstance(current, dict):
                assert forbidden_keys.isdisjoint(current), path
                pending.extend(current.values())
            elif isinstance(current, list):
                pending.extend(current)

    for path in prospect_dir.glob("*.csv"):
        columns = set(pd.read_csv(path, nrows=0).columns.str.lower())
        assert forbidden_keys.isdisjoint(columns), path
