import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    CHECKPOINTS_DIR,
    CHAMPION_METADATA_FILE,
    CHAMPION_MODEL_FILE,
    EXPERIMENT_COMPARISON_FILE,
    ensure_directories,
)


def to_float(value: Any) -> float:
    """
    Converte um valor para float de forma segura.
    """

    try:
        return float(str(value).strip())
    except Exception:
        return 0.0


def build_versioned_checkpoint_path(model_version: str) -> Path:
    """
    Monta o caminho esperado do checkpoint versionado.
    """

    model_version_file_stem = (
        model_version
        .replace(".", "_")
        .replace(" ", "_")
    )

    return CHECKPOINTS_DIR / f"{model_version_file_stem}_best.pth"


def load_comparison() -> pd.DataFrame:
    """
    Carrega o relatório de comparação de experimentos.
    """

    if not EXPERIMENT_COMPARISON_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo de comparação não encontrado: {EXPERIMENT_COMPARISON_FILE}. "
            "Execute primeiro: python -m src.compare_experiments"
        )

    comparison_df = pd.read_csv(
        EXPERIMENT_COMPARISON_FILE,
        dtype=str,
        keep_default_na=False,
    )

    if comparison_df.empty:
        raise ValueError("O arquivo de comparação está vazio.")

    return comparison_df


def select_best_experiment(comparison_df: pd.DataFrame) -> pd.Series:
    """
    Seleciona o melhor experimento com base nas métricas principais.

    Prioridade:
    1. external_accuracy
    2. test_accuracy
    3. best_validation_accuracy
    """

    comparison_df = comparison_df.copy()

    comparison_df["external_accuracy_float"] = comparison_df["external_accuracy"].apply(to_float)
    comparison_df["test_accuracy_float"] = comparison_df["test_accuracy"].apply(to_float)
    comparison_df["best_validation_accuracy_float"] = comparison_df[
        "best_validation_accuracy"
    ].apply(to_float)

    sorted_df = comparison_df.sort_values(
        by=[
            "external_accuracy_float",
            "test_accuracy_float",
            "best_validation_accuracy_float",
        ],
        ascending=False,
    )

    return sorted_df.iloc[0]


def find_source_checkpoint(best_row: pd.Series) -> Path:
    """
    Encontra o checkpoint do melhor experimento.

    Prioriza o checkpoint versionado baseado no model_version,
    pois arquivos antigos como models/astronomy_classifier_best.pth
    podem ter sido sobrescritos por treinos mais recentes.
    """

    model_version = str(best_row.get("model_version", "")).strip()

    versioned_path = build_versioned_checkpoint_path(model_version)

    if versioned_path.exists():
        return versioned_path

    registered_path = Path(str(best_row.get("best_model_file", "")))

    if registered_path.exists():
        return registered_path

    raise FileNotFoundError(
        "Não foi possível encontrar o checkpoint do melhor modelo.\n"
        f"Caminho versionado esperado: {versioned_path}\n"
        f"Caminho registrado no CSV: {registered_path}"
    )

def validate_checkpoint_metadata(
    checkpoint_path: Path,
    best_row: pd.Series,
) -> None:
    """
    Valida se o checkpoint encontrado corresponde ao experimento selecionado.
    """

    import torch

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    expected_model_version = str(best_row.get("model_version", "")).strip()
    expected_train_run_id = str(best_row.get("train_run_id", "")).strip()

    checkpoint_model_version = str(checkpoint.get("model_version", "")).strip()
    checkpoint_train_run_id = str(checkpoint.get("train_run_id", "")).strip()

    if checkpoint_model_version != expected_model_version:
        raise ValueError(
            "Checkpoint encontrado não corresponde ao model_version esperado.\n"
            f"Esperado: {expected_model_version}\n"
            f"Encontrado: {checkpoint_model_version}\n"
            f"Arquivo: {checkpoint_path}"
        )

    if checkpoint_train_run_id != expected_train_run_id:
        raise ValueError(
            "Checkpoint encontrado não corresponde ao train_run_id esperado.\n"
            f"Esperado: {expected_train_run_id}\n"
            f"Encontrado: {checkpoint_train_run_id}\n"
            f"Arquivo: {checkpoint_path}"
        )


def save_metadata(
    best_row: pd.Series,
    source_checkpoint: Path,
) -> None:
    """
    Salva metadados do modelo campeão.
    """

    metadata = {
        "model_version": str(best_row.get("model_version", "")),
        "train_run_id": str(best_row.get("train_run_id", "")),
        "source_checkpoint": str(source_checkpoint),
        "champion_model_file": str(CHAMPION_MODEL_FILE),
        "external_accuracy": to_float(best_row.get("external_accuracy", 0)),
        "test_accuracy": to_float(best_row.get("test_accuracy", 0)),
        "best_validation_accuracy": to_float(
            best_row.get("best_validation_accuracy", 0)
        ),
        "external_nebulosa_accuracy": str(
            best_row.get("external_nebulosa_accuracy", "")
        ),
        "external_galaxia_accuracy": str(
            best_row.get("external_galaxia_accuracy", "")
        ),
        "reason": (
            "Melhor experimento pelo ranking automático, priorizando "
            "acurácia externa, teste interno e validação."
        ),
    }

    CHAMPION_METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    CHAMPION_METADATA_FILE.write_text(
        json.dumps(metadata, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Metadados do campeão salvos em: {CHAMPION_METADATA_FILE}")


def promote_champion() -> None:
    """
    Promove o melhor experimento válido para modelo campeão oficial.

    Experimentos antigos podem ter logs, mas não possuir mais checkpoint
    correspondente. Nesse caso, eles são ignorados.
    """

    ensure_directories()

    comparison_df = load_comparison()

    comparison_df = comparison_df.copy()

    if "is_promotable" in comparison_df.columns:
        comparison_df = comparison_df[
            comparison_df["is_promotable"].astype(str).str.lower() == "true"
        ].copy()

        if comparison_df.empty:
            raise RuntimeError(
                "Nenhum experimento promovível encontrado. "
                "Execute um treino novo para gerar checkpoints por train_run_id."
            )

    comparison_df["external_accuracy_float"] = comparison_df["external_accuracy"].apply(to_float)
    comparison_df["test_accuracy_float"] = comparison_df["test_accuracy"].apply(to_float)
    comparison_df["best_validation_accuracy_float"] = comparison_df[
        "best_validation_accuracy"
    ].apply(to_float)

    ranked_df = comparison_df.sort_values(
        by=[
            "external_accuracy_float",
            "test_accuracy_float",
            "best_validation_accuracy_float",
        ],
        ascending=False,
    )

    selected_row = None
    selected_checkpoint = None

    for _, candidate_row in ranked_df.iterrows():
        try:
            source_checkpoint = find_source_checkpoint(candidate_row)

            validate_checkpoint_metadata(
                checkpoint_path=source_checkpoint,
                best_row=candidate_row,
            )

            selected_row = candidate_row
            selected_checkpoint = source_checkpoint
            break

        except Exception as error:
            print("-" * 80)
            print("Experimento ignorado por checkpoint incompatível ou ausente.")
            print(f"Modelo: {candidate_row.get('model_version', '')}")
            print(f"Train Run ID: {candidate_row.get('train_run_id', '')}")
            print(f"Motivo: {error}")

    if selected_row is None or selected_checkpoint is None:
        raise RuntimeError(
            "Nenhum experimento válido pôde ser promovido. "
            "Rode novamente um treino para gerar checkpoints por train_run_id."
        )

    CHAMPION_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        selected_checkpoint,
        CHAMPION_MODEL_FILE,
    )

    save_metadata(
        best_row=selected_row,
        source_checkpoint=selected_checkpoint,
    )

    print("-" * 80)
    print("Modelo campeão promovido com sucesso.")
    print(f"Modelo: {selected_row.get('model_version', '')}")
    print(f"Train Run ID: {selected_row.get('train_run_id', '')}")
    print(f"Origem: {selected_checkpoint}")
    print(f"Destino: {CHAMPION_MODEL_FILE}")
    print(f"External Accuracy: {to_float(selected_row.get('external_accuracy', 0)) * 100:.2f}%")
    print(f"Test Accuracy: {to_float(selected_row.get('test_accuracy', 0)) * 100:.2f}%")
    print(f"Best Validation Accuracy: {to_float(selected_row.get('best_validation_accuracy', 0)) * 100:.2f}%")

def main() -> None:
    """
    Executa a promoção do modelo campeão.
    """

    promote_champion()


if __name__ == "__main__":
    main()