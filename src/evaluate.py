from typing import Any, cast

import pandas as pd
import torch
from torch import Tensor

from src.config import (
    MODEL_FILE,
    EVALUATION_REPORT_FILE,
    CONFUSION_MATRIX_FILE,
    TEST_LABELS_FILE,
    ensure_directories,
)
from src.dataset import AstronomyImageDataset, get_eval_transforms
from src.train import SimpleAstronomyCNN, get_device


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
) -> SimpleAstronomyCNN:
    """
    Recria a CNN e carrega os pesos treinados.
    """

    num_classes = int(checkpoint["num_classes"])

    model = SimpleAstronomyCNN(num_classes=num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

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
    model: SimpleAstronomyCNN,
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


def evaluate_model() -> list[dict[str, Any]]:
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

    return results


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


def main() -> None:
    """
    Executa a avaliação completa do modelo.
    """

    results = evaluate_model()

    results_df = save_evaluation_report(results)

    save_confusion_matrix(results_df)
    print_class_accuracy(results_df)
    print_errors(results_df)

    print("Avaliação finalizada.")


if __name__ == "__main__":
    main()