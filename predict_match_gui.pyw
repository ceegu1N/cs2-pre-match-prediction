#!/usr/bin/env python3
"""Interface grafica para demonstrar a previsao pre-jogo de partidas de CS2."""
from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from predict_match import list_available_teams, load_prediction_context, predict_match_probability

ROOT = Path(__file__).resolve().parent
OFFICIAL_MODEL_CONFIG_PATH = ROOT / "reports" / "modelo_principal" / "config_modelo_principal.json"


def load_official_model_info() -> dict[str, object]:
    if not OFFICIAL_MODEL_CONFIG_PATH.exists():
        return {}
    return json.loads(OFFICIAL_MODEL_CONFIG_PATH.read_text(encoding="utf-8"))


def build_model_selector_options(registry: dict[str, object]) -> tuple[list[str], dict[str, str]]:
    labels: list[str] = []
    mapping: dict[str, str] = {}

    primary_model = str(registry.get("primary_model") or "logistic_regression")
    primary_label = f"Modelo final ({primary_model})"
    labels.append(primary_label)
    mapping[primary_label] = "primary"

    best_test = str(registry.get("best_model_by_test_roc_auc") or "")
    if best_test and best_test != primary_model:
        label = f"Melhor holdout salvo ({best_test})"
        labels.append(label)
        mapping[label] = "best_test"

    best_cv = str(registry.get("best_model_by_cv_roc_auc") or "")
    if best_cv and best_cv not in {primary_model, best_test}:
        label = f"Melhor CV salvo ({best_cv})"
        labels.append(label)
        mapping[label] = "best_cv"

    for model_name in registry.get("available_models", []):
        model_name = str(model_name)
        if model_name == primary_model:
            continue
        label = f"Modelo salvo ({model_name})"
        if label not in mapping:
            labels.append(label)
            mapping[label] = model_name

    return labels, mapping


def compute_alignment_summary(registry: dict[str, object], official_config: dict[str, object]) -> tuple[str, str]:
    registry_features = [str(item) for item in registry.get("feature_columns", [])]
    official_features = [str(item) for item in official_config.get("features", [])]
    if not official_features:
        return (
            "Nao foi possivel carregar a configuracao documentada do modelo principal.",
            "Sem informacao suficiente para conferir o modelo carregado.",
        )

    missing_from_runtime = [feature for feature in official_features if feature not in registry_features]
    temporal_features = [feature for feature in official_features if feature.startswith("diff_temporal_")]

    if not missing_from_runtime and registry_features == official_features:
        return (
            "O modelo carregado esta alinhado com a configuracao final documentada.",
            f"Atributos usados: {len(registry_features)}. Atributos temporais: {len(temporal_features)}.",
        )

    if missing_from_runtime:
        return (
            "O modelo salvo em models/ nao coincide totalmente com a configuracao final documentada.",
            f"Atributos ausentes no modelo carregado: {', '.join(missing_from_runtime)}.",
        )

    return (
        "A interface usa um modelo valido, mas diferente da configuracao principal documentada.",
        "Revise o artefato salvo em models/ para alinhar a demonstracao ao pacote final.",
    )


class PredictionApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Demonstracao de Previsao de Partidas - CS2")
        self.root.geometry("980x760")
        self.root.minsize(900, 700)

        self.snapshot_labels = {"Demonstracao com dados sinteticos": "demo"}
        self.snapshot_var = tk.StringVar(value="Demonstracao com dados sinteticos")

        teams = list_available_teams(snapshot_mode=self.snapshot_labels[self.snapshot_var.get()])
        _, _, registry, _ = load_prediction_context(snapshot_mode=self.snapshot_labels[self.snapshot_var.get()])
        official_config = load_official_model_info()

        self.slug_by_display = {team["display_name"]: team["slug"] for team in teams}
        self.display_names = sorted(self.slug_by_display)
        self.model_labels, self.model_selector_by_label = build_model_selector_options(registry)
        alignment_title, alignment_text = compute_alignment_summary(registry, official_config)

        default_a = self.display_names[0]
        default_b_index = 1 if len(self.display_names) > 1 else 0
        default_b = self.display_names[default_b_index]

        self.team_a_var = tk.StringVar(value=default_a)
        self.team_b_var = tk.StringVar(value=default_b)
        self.model_var = tk.StringVar(value=self.model_labels[0])
        self.favorite_var = tk.StringVar(value='Selecione os times e clique em "Calcular previsao".')
        self.proba_a_var = tk.StringVar(value='-')
        self.proba_b_var = tk.StringVar(value='-')
        self.elo_a_var = tk.StringVar(value='-')
        self.elo_b_var = tk.StringVar(value='-')
        self.model_used_var = tk.StringVar(value='-')
        self.snapshot_used_var = tk.StringVar(value='-')
        self.feature_count_var = tk.StringVar(value='-')
        self.alignment_title_var = tk.StringVar(value=alignment_title)
        self.alignment_text_var = tk.StringVar(value=alignment_text)
        self.summary_var = tk.StringVar(
            value=(
                "A interface usa duas equipes ficticias para demonstrar o modelo salvo em models/ "
                "e os diferenciais enviados ao classificador."
            )
        )

        self._build_layout()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)

        header = ttk.Frame(self.root, padding=(20, 18, 20, 10))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="Demonstracao de Predicao de Partidas - CS2", font=("Segoe UI", 18, "bold")).grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(
            header,
            text="Compare duas equipes ficticias e visualize os diferenciais enviados ao modelo final.",
            font=("Segoe UI", 10),
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        warning = ttk.LabelFrame(self.root, text="Status do modelo carregado", padding=16)
        warning.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 12))
        warning.columnconfigure(0, weight=1)
        ttk.Label(warning, textvariable=self.alignment_title_var, font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(warning, textvariable=self.alignment_text_var, wraplength=860, justify="left").grid(row=1, column=0, sticky="w", pady=(8, 0))

        controls = ttk.LabelFrame(self.root, text="Entradas", padding=16)
        controls.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 12))
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(3, weight=1)

        ttk.Label(controls, text="Snapshot").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 10))
        snapshot_combo = ttk.Combobox(
            controls,
            textvariable=self.snapshot_var,
            values=list(self.snapshot_labels.keys()),
            state="readonly",
            font=("Segoe UI", 10),
        )
        snapshot_combo.grid(row=0, column=1, columnspan=3, sticky="ew", pady=(0, 10))
        snapshot_combo.bind("<<ComboboxSelected>>", self.on_snapshot_changed)

        ttk.Label(controls, text="Time A").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(0, 10))
        self.team_a_combo = ttk.Combobox(
            controls,
            textvariable=self.team_a_var,
            values=self.display_names,
            state="readonly",
            font=("Segoe UI", 10),
        )
        self.team_a_combo.grid(row=1, column=1, sticky="ew", pady=(0, 10))

        ttk.Label(controls, text="Time B").grid(row=1, column=2, sticky="w", padx=(14, 8), pady=(0, 10))
        self.team_b_combo = ttk.Combobox(
            controls,
            textvariable=self.team_b_var,
            values=self.display_names,
            state="readonly",
            font=("Segoe UI", 10),
        )
        self.team_b_combo.grid(row=1, column=3, sticky="ew", pady=(0, 10))

        ttk.Label(controls, text="Modelo").grid(row=2, column=0, sticky="w", padx=(0, 8))
        ttk.Combobox(
            controls,
            textvariable=self.model_var,
            values=self.model_labels,
            state="readonly",
            font=("Segoe UI", 10),
        ).grid(row=2, column=1, columnspan=3, sticky="ew")

        button_row = ttk.Frame(controls)
        button_row.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(14, 0))
        ttk.Button(button_row, text="Inverter times", command=self.swap_teams).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(button_row, text="Calcular previsao", command=self.run_prediction).grid(row=0, column=1)

        results = ttk.Frame(self.root, padding=(20, 0, 20, 0))
        results.grid(row=3, column=0, sticky="nsew")
        results.columnconfigure(0, weight=1)
        results.columnconfigure(1, weight=1)
        results.rowconfigure(1, weight=1)

        summary_card = ttk.LabelFrame(results, text="Resultado", padding=16)
        summary_card.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        summary_card.columnconfigure(0, weight=1)
        ttk.Label(summary_card, textvariable=self.favorite_var, font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(summary_card, textvariable=self.summary_var, wraplength=820, justify="left").grid(row=1, column=0, sticky="w", pady=(8, 0))

        team_a_card = ttk.LabelFrame(results, text="Time A", padding=16)
        team_a_card.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        team_a_card.columnconfigure(0, weight=1)
        ttk.Label(team_a_card, textvariable=self.team_a_var, font=("Segoe UI", 13, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(team_a_card, text="Probabilidade de vitoria", font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w", pady=(10, 2))
        ttk.Label(team_a_card, textvariable=self.proba_a_var, font=("Segoe UI", 20, "bold")).grid(row=2, column=0, sticky="w")
        ttk.Label(team_a_card, text="Elo atual do time", font=("Segoe UI", 9)).grid(row=3, column=0, sticky="w", pady=(14, 2))
        ttk.Label(team_a_card, textvariable=self.elo_a_var, font=("Segoe UI", 14, "bold")).grid(row=4, column=0, sticky="w")

        team_b_card = ttk.LabelFrame(results, text="Time B", padding=16)
        team_b_card.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        team_b_card.columnconfigure(0, weight=1)
        ttk.Label(team_b_card, textvariable=self.team_b_var, font=("Segoe UI", 13, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(team_b_card, text="Probabilidade de vitoria", font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w", pady=(10, 2))
        ttk.Label(team_b_card, textvariable=self.proba_b_var, font=("Segoe UI", 20, "bold")).grid(row=2, column=0, sticky="w")
        ttk.Label(team_b_card, text="Elo atual do time", font=("Segoe UI", 9)).grid(row=3, column=0, sticky="w", pady=(14, 2))
        ttk.Label(team_b_card, textvariable=self.elo_b_var, font=("Segoe UI", 14, "bold")).grid(row=4, column=0, sticky="w")

        details = ttk.LabelFrame(self.root, text="Detalhes do confronto", padding=16)
        details.grid(row=4, column=0, sticky="nsew", padx=20, pady=(0, 20))
        details.columnconfigure(0, weight=1)

        ttk.Label(details, text="Modelo efetivamente usado:").grid(row=0, column=0, sticky="w")
        ttk.Label(details, textvariable=self.model_used_var, font=("Segoe UI", 10, "bold")).grid(row=1, column=0, sticky="w", pady=(2, 8))
        ttk.Label(details, text="Snapshot efetivamente usado:").grid(row=2, column=0, sticky="w")
        ttk.Label(details, textvariable=self.snapshot_used_var, font=("Segoe UI", 10, "bold")).grid(row=3, column=0, sticky="w", pady=(2, 8))
        ttk.Label(details, text="Numero de features usadas pelo modelo salvo:").grid(row=4, column=0, sticky="w")
        ttk.Label(details, textvariable=self.feature_count_var, font=("Segoe UI", 10, "bold")).grid(row=5, column=0, sticky="w", pady=(2, 10))

        self.details_text = tk.Text(details, height=12, wrap="word", font=("Consolas", 10))
        self.details_text.grid(row=6, column=0, sticky="nsew")
        self.details_text.insert(
            "1.0",
            "Os diferenciais efetivamente enviados ao modelo salvo em models/ aparecerao aqui depois do primeiro calculo.",
        )
        self.details_text.config(state="disabled")

    def swap_teams(self) -> None:
        team_a = self.team_a_var.get()
        team_b = self.team_b_var.get()
        self.team_a_var.set(team_b)
        self.team_b_var.set(team_a)

    def on_snapshot_changed(self, _event: object | None = None) -> None:
        snapshot_mode = self.snapshot_labels[self.snapshot_var.get()]
        teams = list_available_teams(snapshot_mode=snapshot_mode)
        self.slug_by_display = {team["display_name"]: team["slug"] for team in teams}
        self.display_names = sorted(self.slug_by_display)
        self.team_a_combo["values"] = self.display_names
        self.team_b_combo["values"] = self.display_names
        if not self.display_names:
            return
        if self.team_a_var.get() not in self.slug_by_display:
            self.team_a_var.set(self.display_names[0])
        if self.team_b_var.get() not in self.slug_by_display:
            fallback_index = 1 if len(self.display_names) > 1 else 0
            self.team_b_var.set(self.display_names[fallback_index])

    def run_prediction(self) -> None:
        try:
            result = predict_match_probability(
                self.slug_by_display[self.team_a_var.get()],
                self.slug_by_display[self.team_b_var.get()],
                model_selector=self.model_selector_by_label[self.model_var.get()],
                snapshot_mode=self.snapshot_labels[self.snapshot_var.get()],
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Erro na previsao", str(exc))
            return

        self.proba_a_var.set(f"{result['probability_team_a']:.2%}")
        self.proba_b_var.set(f"{result['probability_team_b']:.2%}")
        self.elo_a_var.set(f"{float(result['team_a_snapshot'].get('elo_rating', 0.0)):.3f}")
        self.elo_b_var.set(f"{float(result['team_b_snapshot'].get('elo_rating', 0.0)):.3f}")
        self.model_used_var.set(str(result["model_name"]))
        self.snapshot_used_var.set(str(result["snapshot_path"]))
        self.feature_count_var.set(str(len(result["feature_values"])))
        self.favorite_var.set(f"Favorito: {result['favorite']}")
        self.summary_var.set(
            f"{result['team_a_display_name']} vs {result['team_b_display_name']} usando {result['model_name']}. "
            f"Probabilidades: {result['team_a_display_name']} {result['probability_team_a']:.2%} e "
            f"{result['team_b_display_name']} {result['probability_team_b']:.2%}. "
            f"Snapshot selecionado: {self.snapshot_var.get()}. "
            "Probabilidade simetrizada para nao depender da ordem dos times."
        )
        self._update_details(result)

    def _update_details(self, result: dict[str, object]) -> None:
        team_a_snapshot = result["team_a_snapshot"]
        team_b_snapshot = result["team_b_snapshot"]
        feature_values = result["feature_values"]

        def as_float(value: object, default: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        lines = [
            f"Resumo do confronto: {result['team_a_display_name']} vs {result['team_b_display_name']}",
            "",
            f"Snapshot utilizado: {result['snapshot_path']}",
            f"Modelo carregado: {result['model_name']}",
            f"Probabilidade simetrizada: {'sim' if result.get('probability_symmetrized') else 'nao'}",
            "",
            "Probabilidades estimadas:",
            f"- {result['team_a_display_name']}: {result['probability_team_a']:.2%}",
            f"- {result['team_b_display_name']}: {result['probability_team_b']:.2%}",
            f"- Favorito: {result['favorite']}",
            "",
            "Resumo dos times no snapshot:",
            f"- Elo: {as_float(team_a_snapshot.get('elo_rating')):.3f} vs {as_float(team_b_snapshot.get('elo_rating')):.3f}",
            f"- Recent win rate: {as_float(team_a_snapshot.get('recent_win_rate')):.3f} vs {as_float(team_b_snapshot.get('recent_win_rate')):.3f}",
            f"- Rating medio dos 5 jogadores: {as_float(team_a_snapshot.get('rating_mean_5')):.3f} vs {as_float(team_b_snapshot.get('rating_mean_5')):.3f}",
            f"- ADR medio dos 5 jogadores: {as_float(team_a_snapshot.get('adr_mean_5')):.2f} vs {as_float(team_b_snapshot.get('adr_mean_5')):.2f}",
            "",
            "Diferenciais efetivamente enviados ao modelo:",
        ]

        for feature_name, feature_value in feature_values.items():
            marker = " (temporal)" if str(feature_name).startswith("diff_temporal_") else ""
            lines.append(f"- {feature_name}{marker} = {float(feature_value):.6f}")

        self.details_text.config(state="normal")
        self.details_text.delete("1.0", tk.END)
        self.details_text.insert("1.0", "\n".join(lines))
        self.details_text.config(state="disabled")

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    PredictionApp().run()
