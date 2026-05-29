from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.config import (
    TRAIN_RUNS_LOG_FILE,
    EVALUATE_RUNS_LOG_FILE,
    EXTERNAL_EVALUATE_RUNS_LOG_FILE,
    EXPERIMENT_COMPARISON_FILE,
    EXPERIMENT_COMPARISON_MARKDOWN_FILE,
    CHECKPOINTS_DIR,
    RUN_CHECKPOINTS_DIR,
    ensure_directories,
)

def read_csv_if_exists(path: Path) -> pd.DataFrame:
    """
    Lê um CSV se ele existir.
    Caso não exista, retorna DataFrame vazio.
    """

    if not path.exists():
        print(f"Arquivo não encontrado: {path}")
        return pd.DataFrame()

    try:
        return pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
        )

    except Exception as error:
        print(f"Erro ao ler {path}: {error}")
        return pd.DataFrame()


def to_float(value: Any) -> float:
    """
    Converte valores para float de forma segura.
    """

    try:
        if value is None:
            return 0.0

        text = str(value).strip()

        if text == "":
            return 0.0

        return float(text)

    except Exception:
        return 0.0


def to_int(value: Any) -> int:
    """
    Converte valores para int de forma segura.
    """

    try:
        if value is None:
            return 0

        text = str(value).strip()

        if text == "":
            return 0

        return int(float(text))

    except Exception:
        return 0


def parse_class_accuracy(
    class_accuracy_text: str,
    target_class: str,
) -> str:
    """
    Extrai a acurácia de uma classe específica do campo class_accuracy.

    Exemplo de entrada:
    aglomerado=1.000000(6/6); estrela=1.000000(6/6)
    """

    if not isinstance(class_accuracy_text, str):
        return ""

    parts = class_accuracy_text.split(";")

    for part in parts:
        cleaned = part.strip()

        if cleaned.startswith(f"{target_class}="):
            return cleaned.replace(f"{target_class}=", "")

    return ""

def build_model_version_stem(model_version: str) -> str:
    """
    Converte o nome da versão do modelo para um formato seguro de arquivo.

    Exemplo:
    AstroMindCNNV2.4 -> AstroMindCNNV2_4
    """

    return (
        model_version
        .replace(".", "_")
        .replace(" ", "_")
    )


def get_checkpoint_candidates(
    model_version: str,
    train_run_id: str,
    best_model_file: str,
) -> list[Path]:
    """
    Monta os possíveis caminhos de checkpoint para um experimento.

    Ordem:
    1. Caminho registrado no train_runs.csv.
    2. Caminho por execução: models/checkpoints/runs/{model}/{train_run_id}_best.pth.
    3. Caminho alias da versão: models/checkpoints/{model}_best.pth.
    """

    candidates: list[Path] = []

    if best_model_file:
        candidates.append(Path(best_model_file))

    model_version_stem = build_model_version_stem(model_version)

    run_checkpoint_path = (
        RUN_CHECKPOINTS_DIR
        / model_version_stem
        / f"{train_run_id}_best.pth"
    )

    candidates.append(run_checkpoint_path)

    version_checkpoint_path = CHECKPOINTS_DIR / f"{model_version_stem}_best.pth"

    candidates.append(version_checkpoint_path)

    unique_candidates: list[Path] = []

    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)

    return unique_candidates


def inspect_checkpoint_status(row: dict[str, Any]) -> dict[str, Any]:
    """
    Verifica se existe um checkpoint válido para o experimento.

    Um checkpoint é considerado promovível quando:
    - existe fisicamente;
    - contém o mesmo model_version registrado no log;
    - contém o mesmo train_run_id registrado no log.
    """

    expected_model_version = str(row.get("model_version", "")).strip()
    expected_train_run_id = str(row.get("train_run_id", "")).strip()
    best_model_file = str(row.get("best_model_file", "")).strip()

    candidates = get_checkpoint_candidates(
        model_version=expected_model_version,
        train_run_id=expected_train_run_id,
        best_model_file=best_model_file,
    )

    first_existing_checkpoint = ""
    first_checkpoint_model_version = ""
    first_checkpoint_train_run_id = ""
    first_error = ""

    for candidate in candidates:
        if not candidate.exists():
            continue

        first_existing_checkpoint = str(candidate)

        try:
            checkpoint = torch.load(
                candidate,
                map_location="cpu",
            )

            checkpoint_model_version = str(
                checkpoint.get("model_version", "")
            ).strip()

            checkpoint_train_run_id = str(
                checkpoint.get("train_run_id", "")
            ).strip()

            first_checkpoint_model_version = checkpoint_model_version
            first_checkpoint_train_run_id = checkpoint_train_run_id

            matches_model = checkpoint_model_version == expected_model_version
            matches_run = checkpoint_train_run_id == expected_train_run_id

            if matches_model and matches_run:
                return {
                    "checkpoint_path": str(candidate),
                    "checkpoint_exists": True,
                    "checkpoint_model_version": checkpoint_model_version,
                    "checkpoint_train_run_id": checkpoint_train_run_id,
                    "checkpoint_matches_train_run": True,
                    "is_promotable": True,
                    "checkpoint_status": "valid",
                }

        except Exception as error:
            first_error = str(error)

    if first_existing_checkpoint:
        return {
            "checkpoint_path": first_existing_checkpoint,
            "checkpoint_exists": True,
            "checkpoint_model_version": first_checkpoint_model_version,
            "checkpoint_train_run_id": first_checkpoint_train_run_id,
            "checkpoint_matches_train_run": False,
            "is_promotable": False,
            "checkpoint_status": (
                "checkpoint encontrado, mas metadata incompatível"
            ),
        }

    return {
        "checkpoint_path": "",
        "checkpoint_exists": False,
        "checkpoint_model_version": "",
        "checkpoint_train_run_id": "",
        "checkpoint_matches_train_run": False,
        "is_promotable": False,
        "checkpoint_status": (
            f"checkpoint não encontrado"
            if not first_error
            else f"erro ao ler checkpoint: {first_error}"
        ),
    }

def latest_by_train_run(
    df: pd.DataFrame,
    date_column: str = "created_at",
) -> pd.DataFrame:
    """
    Mantém a última linha registrada por train_run_id.

    Isso evita duplicidade quando você avalia o mesmo treino mais de uma vez.
    """

    if df.empty:
        return df

    if "train_run_id" not in df.columns:
        return df

    if date_column not in df.columns:
        return df.drop_duplicates(
            subset=["train_run_id"],
            keep="last",
        )

    sorted_df = df.sort_values(by=date_column)

    return sorted_df.drop_duplicates(
        subset=["train_run_id"],
        keep="last",
    )


def build_comparison_df(
    train_df: pd.DataFrame,
    evaluate_df: pd.DataFrame,
    external_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Junta logs de treino, avaliação interna e avaliação externa.
    """

    if train_df.empty:
        raise ValueError("Não há treinos registrados em train_runs.csv.")

    evaluate_df = latest_by_train_run(evaluate_df)
    external_df = latest_by_train_run(external_df)

    rows: list[dict[str, Any]] = []

    for _, train_row in train_df.iterrows():
        train_run_id = str(train_row.get("train_run_id", "")).strip()

        if not train_run_id:
            continue

        matching_eval = pd.DataFrame()

        if not evaluate_df.empty and "train_run_id" in evaluate_df.columns:
            matching_eval = evaluate_df[
                evaluate_df["train_run_id"].astype(str) == train_run_id
            ]

        matching_external = pd.DataFrame()

        if not external_df.empty and "train_run_id" in external_df.columns:
            matching_external = external_df[
                external_df["train_run_id"].astype(str) == train_run_id
            ]

        eval_row = matching_eval.iloc[-1] if not matching_eval.empty else {}
        external_row = matching_external.iloc[-1] if not matching_external.empty else {}

        internal_class_accuracy = str(
            eval_row.get("class_accuracy", "")
            if hasattr(eval_row, "get")
            else ""
        )

        external_class_accuracy = str(
            external_row.get("class_accuracy", "")
            if hasattr(external_row, "get")
            else ""
        )

        row = {
            "train_run_id": train_run_id,
            "created_at": train_row.get("created_at", ""),
            "model_version": train_row.get("model_version", ""),
            "train_size": to_int(train_row.get("train_size", 0)),
            "validation_size": to_int(train_row.get("validation_size", 0)),
            "epochs": to_int(train_row.get("epochs", 0)),
            "stopped_epoch": to_int(train_row.get("stopped_epoch", 0)),
            "best_epoch": to_int(train_row.get("best_epoch", 0)),
            "best_validation_loss": to_float(train_row.get("best_validation_loss", 0)),
            "best_validation_accuracy": to_float(
                train_row.get("best_validation_accuracy", 0)
            ),
            "final_train_accuracy": to_float(train_row.get("final_train_accuracy", 0)),
            "final_validation_accuracy": to_float(
                train_row.get("final_validation_accuracy", 0)
            ),
            "use_weighted_sampler": train_row.get("use_weighted_sampler", ""),
            "use_class_weights": train_row.get("use_class_weights", ""),
            "best_model_file": train_row.get("best_model_file", ""),
            "test_accuracy": to_float(
                eval_row.get("accuracy", 0)
                if hasattr(eval_row, "get")
                else 0
            ),
            "test_correct": to_int(
                eval_row.get("correct", 0)
                if hasattr(eval_row, "get")
                else 0
            ),
            "test_errors": to_int(
                eval_row.get("errors", 0)
                if hasattr(eval_row, "get")
                else 0
            ),
            "test_nebulosa_accuracy": parse_class_accuracy(
                internal_class_accuracy,
                "nebulosa",
            ),
            "test_galaxia_accuracy": parse_class_accuracy(
                internal_class_accuracy,
                "galaxia",
            ),
            "external_accuracy": to_float(
                external_row.get("accuracy", 0)
                if hasattr(external_row, "get")
                else 0
            ),
            "external_correct": to_int(
                external_row.get("correct", 0)
                if hasattr(external_row, "get")
                else 0
            ),
            "external_errors": to_int(
                external_row.get("errors", 0)
                if hasattr(external_row, "get")
                else 0
            ),
            "external_nebulosa_accuracy": parse_class_accuracy(
                external_class_accuracy,
                "nebulosa",
            ),
            "external_galaxia_accuracy": parse_class_accuracy(
                external_class_accuracy,
                "galaxia",
            ),
            "external_estrela_accuracy": parse_class_accuracy(
                external_class_accuracy,
                "estrela",
            ),
            "external_aglomerado_accuracy": parse_class_accuracy(
                external_class_accuracy,
                "aglomerado",
            ),
        }

        checkpoint_status = inspect_checkpoint_status(row)
        row.update(checkpoint_status)

        rows.append(row)

    comparison_df = pd.DataFrame(rows)

    if comparison_df.empty:
        raise ValueError("Nenhuma comparação pôde ser gerada.")

    comparison_df = comparison_df.sort_values(
        by=[
            "external_accuracy",
            "test_accuracy",
            "best_validation_accuracy",
        ],
        ascending=False,
    )

    return comparison_df


def save_comparison_csv(comparison_df: pd.DataFrame) -> None:
    """
    Salva comparação em CSV.
    """

    EXPERIMENT_COMPARISON_FILE.parent.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(
        EXPERIMENT_COMPARISON_FILE,
        index=False,
        encoding="utf-8",
    )

    print(f"Comparação salva em: {EXPERIMENT_COMPARISON_FILE}")


def format_percent(value: Any) -> str:
    """
    Formata número decimal como percentual.
    """

    number = to_float(value)

    return f"{number * 100:.2f}%"


def save_comparison_markdown(comparison_df: pd.DataFrame) -> None:
    """
    Salva comparação em Markdown.
    """

    lines: list[str] = []

    lines.append("# Comparação de Experimentos — AstroMind")
    lines.append("")
    lines.append("Este arquivo compara os experimentos registrados nos logs acumulativos.")
    lines.append("")
    lines.append("## Ranking geral")
    lines.append("")
    lines.append(
        "| Posição | Modelo | Train Run ID | Test Acc | External Acc | "
        "Best Val Acc | Nebulosa Externa | Galáxia Externa | Promovível |"
    )
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")

    for position, (_, row) in enumerate(comparison_df.iterrows(), start=1):
        lines.append(
            "| "
            f"{position} | "
            f"{row.get('model_version', '')} | "
            f"`{row.get('train_run_id', '')}` | "
            f"{format_percent(row.get('test_accuracy', 0))} | "
            f"{format_percent(row.get('external_accuracy', 0))} | "
            f"{format_percent(row.get('best_validation_accuracy', 0))} | "
            f"{row.get('external_nebulosa_accuracy', '')} | "
            f"{row.get('external_galaxia_accuracy', '')} | "
            f"{row.get('is_promotable', '')} |"
        )

    best_row = comparison_df.iloc[0]

    lines.append("")
    lines.append("## Melhor experimento até o momento")
    lines.append("")
    lines.append(f"- Modelo: **{best_row.get('model_version', '')}**")
    lines.append(f"- Train Run ID: `{best_row.get('train_run_id', '')}`")
    lines.append(f"- Test Accuracy: {format_percent(best_row.get('test_accuracy', 0))}")
    lines.append(
        f"- External Accuracy: {format_percent(best_row.get('external_accuracy', 0))}"
    )
    lines.append(
        f"- Best Validation Accuracy: "
        f"{format_percent(best_row.get('best_validation_accuracy', 0))}"
    )
    lines.append(
        f"- Nebulosa externa: {best_row.get('external_nebulosa_accuracy', '')}"
    )
    lines.append(
        f"- Galáxia externa: {best_row.get('external_galaxia_accuracy', '')}"
    )
    lines.append(
        f"- Promovível: {best_row.get('is_promotable', '')}"
    )

    promotable_df = comparison_df[
        comparison_df["is_promotable"].astype(str).str.lower() == "true"
    ]

    lines.append("")
    lines.append("## Melhor experimento promovível")
    lines.append("")

    if promotable_df.empty:
        lines.append("Nenhum experimento com checkpoint válido foi encontrado.")
    else:
        best_promotable_row = promotable_df.iloc[0]

        lines.append(f"- Modelo: **{best_promotable_row.get('model_version', '')}**")
        lines.append(f"- Train Run ID: `{best_promotable_row.get('train_run_id', '')}`")
        lines.append(
            f"- Test Accuracy: "
            f"{format_percent(best_promotable_row.get('test_accuracy', 0))}"
        )
        lines.append(
            f"- External Accuracy: "
            f"{format_percent(best_promotable_row.get('external_accuracy', 0))}"
        )
        lines.append(
            f"- Best Validation Accuracy: "
            f"{format_percent(best_promotable_row.get('best_validation_accuracy', 0))}"
        )
        lines.append(
            f"- Nebulosa externa: "
            f"{best_promotable_row.get('external_nebulosa_accuracy', '')}"
        )
        lines.append(
            f"- Galáxia externa: "
            f"{best_promotable_row.get('external_galaxia_accuracy', '')}"
        )
        lines.append(
            f"- Checkpoint: `{best_promotable_row.get('checkpoint_path', '')}`"
        )
    lines.append("")
    lines.append("## Observações")
    lines.append("")
    lines.append(
        "- O ranking prioriza a acurácia externa, pois ela mede melhor a generalização."
    )
    lines.append(
        "- Em caso de empate, considera-se acurácia de teste interno e depois validação."
    )
    lines.append(
        "- Resultados com poucos exemplos externos devem ser interpretados com cautela."
    )

    EXPERIMENT_COMPARISON_MARKDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)

    EXPERIMENT_COMPARISON_MARKDOWN_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Comparação Markdown salva em: {EXPERIMENT_COMPARISON_MARKDOWN_FILE}")


def print_summary(comparison_df: pd.DataFrame) -> None:
    """
    Exibe resumo no terminal.
    """

    print("-" * 80)
    print("Ranking de experimentos")

    for position, (_, row) in enumerate(comparison_df.iterrows(), start=1):
        print(
            f"{position}. {row.get('model_version', '')} | "
            f"External Acc: {format_percent(row.get('external_accuracy', 0))} | "
            f"Test Acc: {format_percent(row.get('test_accuracy', 0))} | "
            f"Best Val Acc: {format_percent(row.get('best_validation_accuracy', 0))} | "
            f"Promovível: {row.get('is_promotable', '')} | "
            f"Train Run: {row.get('train_run_id', '')}"
        )

    best_row = comparison_df.iloc[0]

    print("-" * 80)
    print("Melhor experimento histórico")
    print(f"Modelo: {best_row.get('model_version', '')}")
    print(f"Train Run ID: {best_row.get('train_run_id', '')}")
    print(f"External Accuracy: {format_percent(best_row.get('external_accuracy', 0))}")
    print(f"Test Accuracy: {format_percent(best_row.get('test_accuracy', 0))}")
    print(
        f"Best Validation Accuracy: "
        f"{format_percent(best_row.get('best_validation_accuracy', 0))}"
    )
    print(f"Nebulosa externa: {best_row.get('external_nebulosa_accuracy', '')}")
    print(f"Galáxia externa: {best_row.get('external_galaxia_accuracy', '')}")
    print(f"Promovível: {best_row.get('is_promotable', '')}")

    promotable_df = comparison_df[
        comparison_df["is_promotable"].astype(str).str.lower() == "true"
    ]

    print("-" * 80)
    print("Melhor experimento promovível")

    if promotable_df.empty:
        print("Nenhum experimento promovível encontrado.")
        return

    best_promotable_row = promotable_df.iloc[0]

    print(f"Modelo: {best_promotable_row.get('model_version', '')}")
    print(f"Train Run ID: {best_promotable_row.get('train_run_id', '')}")
    print(
        f"External Accuracy: "
        f"{format_percent(best_promotable_row.get('external_accuracy', 0))}"
    )
    print(
        f"Test Accuracy: "
        f"{format_percent(best_promotable_row.get('test_accuracy', 0))}"
    )
    print(
        f"Best Validation Accuracy: "
        f"{format_percent(best_promotable_row.get('best_validation_accuracy', 0))}"
    )
    print(f"Nebulosa externa: {best_promotable_row.get('external_nebulosa_accuracy', '')}")
    print(f"Galáxia externa: {best_promotable_row.get('external_galaxia_accuracy', '')}")
    print(f"Checkpoint: {best_promotable_row.get('checkpoint_path', '')}")


def main() -> None:
    """
    Executa a comparação dos experimentos.
    """

    ensure_directories()

    train_df = read_csv_if_exists(TRAIN_RUNS_LOG_FILE)
    evaluate_df = read_csv_if_exists(EVALUATE_RUNS_LOG_FILE)
    external_df = read_csv_if_exists(EXTERNAL_EVALUATE_RUNS_LOG_FILE)

    comparison_df = build_comparison_df(
        train_df=train_df,
        evaluate_df=evaluate_df,
        external_df=external_df,
    )

    save_comparison_csv(comparison_df)
    save_comparison_markdown(comparison_df)
    print_summary(comparison_df)

    print("-" * 80)
    print("Comparação de experimentos finalizada.")


if __name__ == "__main__":
    main()