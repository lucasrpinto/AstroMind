from typing import Any, cast

import pandas as pd
import torch
from torch import Tensor

from src.config import (
    MODEL_FILE,
    EVALUATION_REPORT_FILE,
    CONFUSION_MATRIX_FILE,
    TEST_LABELS_FILE,
    EVALUATE_RUNS_LOG_FILE,
    TEST_LABELS_FILE,
    ensure_directories,
)

from src.experiment_logger import (
    append_csv_row,
    compact_dict,
    create_run_id,
    format_float,
    now_iso,
)

from src.dataset import AstronomyImageDataset, get_eval_transforms

from torch import nn

from src.model_registry import build_model
from src.train import get_device


def load_checkpoint(device: torch.device) -> dict[str, Any]:
    """
    Carrega o modelo salvo no treino.
    """

    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_FILE}")

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
    )

    return cast(dict[str, Any], checkpoint)


def load_model(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """
    Recria a CNN com base na versão salva no checkpoint e carrega os pesos treinados.
    """

    num_classes = int(checkpoint["num_classes"])
    model_version = str(checkpoint.get("model_version", "AstroMindCNNV1"))

    model = build_model(
        num_classes=num_classes,
        model_version=model_version,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Versão do modelo carregado: {model_version}")

    return model


def get_class_name(
    idx_to_class: dict[Any, Any],
    index: int,
) -> str:
    """
    Retorna o nome da classe usando índice inteiro ou string.

    Dependendo de como o checkpoint foi salvo/carregado,
    as chaves podem voltar como int ou str.
    """

    class_name = idx_to_class.get(
        index,
        idx_to_class.get(str(index), f"classe_{index}"),
    )

    return str(class_name)


@torch.no_grad()
def predict_tensor(
    model: nn.Module,
    image_tensor: Tensor,
    device: torch.device,
) -> tuple[int, float, Tensor]:
    """
    Faz a predição de uma imagem e retorna índice, confiança e probabilidades.
    """

    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)

    predicted_index = int(torch.argmax(probabilities, dim=1).item())
    confidence = float(probabilities[0][predicted_index].item())

    return predicted_index, confidence, probabilities[0].cpu()


def evaluate_model() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Avalia o modelo em todas as imagens do dataset aprovado.
    """

    ensure_directories()

    device = get_device()

    print(f"Dispositivo usado: {device}")

    checkpoint = load_checkpoint(device)
    model = load_model(checkpoint, device)

    idx_to_class = checkpoint["idx_to_class"]

    dataset = AstronomyImageDataset(
        labels_file=TEST_LABELS_FILE,
        transform=get_eval_transforms(),
    )

    print(f"Total de imagens avaliadas: {len(dataset)}")
    print(f"Classes: {dataset.classes}")

    results: list[dict[str, Any]] = []

    correct_predictions = 0

    for index in range(len(dataset)):
        image_tensor, label_tensor = dataset[index]

        filename = str(dataset.data.iloc[index]["filename"])

        true_index = int(label_tensor.item())
        true_class = dataset.idx_to_class[true_index]

        predicted_index, confidence, probabilities = predict_tensor(
            model=model,
            image_tensor=image_tensor,
            device=device,
        )

        predicted_class = get_class_name(
            idx_to_class=idx_to_class,
            index=predicted_index,
        )

        is_correct = true_class == predicted_class

        if is_correct:
            correct_predictions += 1

        row: dict[str, Any] = {
            "filename": filename,
            "true_class": true_class,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 6),
            "is_correct": is_correct,
        }

        for class_index, probability in enumerate(probabilities):
            class_name = get_class_name(
                idx_to_class=idx_to_class,
                index=class_index,
            )

            row[f"prob_{class_name}"] = round(float(probability.item()), 6)

        results.append(row)

    accuracy = correct_predictions / len(dataset)

    print("-" * 80)
    print(f"Acurácia geral: {accuracy * 100:.2f}%")
    print(f"Acertos: {correct_predictions}/{len(dataset)}")

    return results, checkpoint


def save_evaluation_report(results: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Salva o relatório completo de avaliação por imagem.
    """

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        EVALUATION_REPORT_FILE,
        index=False,
        encoding="utf-8",
    )

    print(f"Relatório de avaliação salvo em: {EVALUATION_REPORT_FILE}")

    return results_df


def save_confusion_matrix(results_df: pd.DataFrame) -> None:
    """
    Salva a matriz de confusão em CSV.

    Linhas: classe real.
    Colunas: classe prevista.
    """

    confusion_matrix_df = pd.crosstab(
        results_df["true_class"],
        results_df["predicted_class"],
        rownames=["classe_real"],
        colnames=["classe_prevista"],
        dropna=False,
    )

    confusion_matrix_df.to_csv(
        CONFUSION_MATRIX_FILE,
        encoding="utf-8",
    )

    print(f"Matriz de confusão salva em: {CONFUSION_MATRIX_FILE}")

    print("-" * 80)
    print("Matriz de confusão:")
    print(confusion_matrix_df)


def print_class_accuracy(results_df: pd.DataFrame) -> None:
    """
    Exibe a acurácia por classe.
    """

    print("-" * 80)
    print("Acurácia por classe:")

    grouped = results_df.groupby("true_class")

    for class_name, group in grouped:
        total = len(group)
        correct = int(group["is_correct"].sum())
        accuracy = correct / total

        print(f"{class_name}: {accuracy * 100:.2f}% ({correct}/{total})")


def print_errors(results_df: pd.DataFrame) -> None:
    """
    Exibe os erros do modelo.
    """

    errors_df = results_df[results_df["is_correct"] == False]

    print("-" * 80)
    print(f"Total de erros: {len(errors_df)}")

    if errors_df.empty:
        print("Nenhum erro encontrado.")
        return

    for _, row in errors_df.iterrows():
        print(
            f"{row['filename']} | "
            f"Real: {row['true_class']} | "
            f"Previsto: {row['predicted_class']} | "
            f"Confiança: {float(row['confidence']) * 100:.2f}%"
        )


def append_evaluate_run_log(
    evaluate_run_id: str,
    checkpoint: dict[str, Any],
    results_df: pd.DataFrame,
) -> None:
    """
    Registra uma linha acumulativa com o resultado da avaliação.
    """

    total = len(results_df)
    correct = int(results_df["is_correct"].sum())
    errors = total - correct
    accuracy = correct / total if total > 0 else 0.0

    class_accuracy: dict[str, str] = {}

    grouped = results_df.groupby("true_class")

    for class_name, group in grouped:
        class_total = len(group)
        class_correct = int(group["is_correct"].sum())
        class_acc = class_correct / class_total if class_total > 0 else 0.0

        class_accuracy[str(class_name)] = (
            f"{format_float(class_acc)}({class_correct}/{class_total})"
        )

    fieldnames = [
        "external_evaluate_run_id",
        "train_run_id",
        "created_at",
        "model_version",
        "model_file",
        "model_epoch",
        "total_images",
        "labeled_images",
        "accuracy",
        "correct",
        "errors",
        "class_accuracy",
        "external_labels_file",
        "external_report_file",
        "external_confusion_matrix_file",
    ]

    row = {
        "evaluate_run_id": evaluate_run_id,
        "train_run_id": checkpoint.get("train_run_id", ""),
        "model_version": checkpoint.get("model_version", ""),
        "created_at": now_iso(),
        "model_file": str(MODEL_FILE),
        "model_epoch": checkpoint.get("epoch", ""),
        "dataset_file": str(TEST_LABELS_FILE),
        "dataset_size": total,
        "num_classes": len(checkpoint.get("classes", [])),
        "classes": "|".join(checkpoint.get("classes", [])),
        "accuracy": format_float(accuracy),
        "correct": correct,
        "errors": errors,
        "class_accuracy": compact_dict(class_accuracy),
        "evaluation_report_file": str(EVALUATION_REPORT_FILE),
        "confusion_matrix_file": str(CONFUSION_MATRIX_FILE),
    }

    append_csv_row(
        file_path=EVALUATE_RUNS_LOG_FILE,
        fieldnames=fieldnames,
        row=row,
    )

    print(f"Log acumulativo da avaliação salvo em: {EVALUATE_RUNS_LOG_FILE}")

def main() -> None:
    """
    Executa a avaliação completa do modelo.
    """

    evaluate_run_id = create_run_id("evaluate")
    print(f"Evaluate Run ID: {evaluate_run_id}")

    results, checkpoint = evaluate_model()

    results_df = save_evaluation_report(results)

    save_confusion_matrix(results_df)
    print_class_accuracy(results_df)
    print_errors(results_df)

    append_evaluate_run_log(
        evaluate_run_id=evaluate_run_id,
        checkpoint=checkpoint,
        results_df=results_df,
    )

    print("Avaliação finalizada.")


if __name__ == "__main__":
    main()