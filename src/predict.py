import argparse
from pathlib import Path
from typing import Any, cast
import pandas as pd

import torch
from PIL import Image
from torch import Tensor

from src.config import (
    MODEL_FILE,
    PROCESSED_IMAGES_DIR,
    PREDICT_RUNS_LOG_FILE,
    LABELS_FILE,
)

from src.experiment_logger import (
    append_csv_row,
    compact_dict,
    create_run_id,
    format_float,
    now_iso,
)

from src.dataset import get_default_transforms
from src.train import SimpleAstronomyCNN, get_device


def load_checkpoint(model_path: Path, device: torch.device) -> dict[str, Any]:
    """
    Carrega o checkpoint salvo durante o treino.
    """

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    return cast(dict[str, Any], checkpoint)


def load_model(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> SimpleAstronomyCNN:
    """
    Recria a arquitetura do modelo e carrega os pesos treinados.
    """

    num_classes = int(checkpoint["num_classes"])

    model = SimpleAstronomyCNN(num_classes=num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def load_image(image_path: Path) -> Tensor:
    """
    Carrega uma imagem PNG/JPG e aplica as mesmas transformações usadas no treino.
    """

    if not image_path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    image = Image.open(image_path).convert("RGB")

    transform = get_default_transforms()
    image_tensor = transform(image)

    # Adiciona a dimensão do batch.
    # Antes: [3, 224, 224]
    # Depois: [1, 3, 224, 224]
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


@torch.no_grad()
def predict_image(
    model: SimpleAstronomyCNN,
    image_tensor: Tensor,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[str, float, dict[str, float]]:
    """
    Executa a predição da imagem e retorna a classe prevista.
    """

    image_tensor = image_tensor.to(device)

    outputs = model(image_tensor)

    probabilities = torch.softmax(outputs, dim=1)

    predicted_index = int(torch.argmax(probabilities, dim=1).item())
    confidence = float(probabilities[0][predicted_index].item())

    idx_to_class = checkpoint["idx_to_class"]

    # Em alguns casos, o dicionário salvo pode voltar com chave int ou string.
    predicted_class = idx_to_class.get(
        predicted_index,
        idx_to_class.get(str(predicted_index), "classe_desconhecida"),
    )

    all_probabilities: dict[str, float] = {}

    for index, probability in enumerate(probabilities[0]):
        class_name = idx_to_class.get(
            index,
            idx_to_class.get(str(index), f"classe_{index}"),
        )

        all_probabilities[str(class_name)] = float(probability.item())

    return str(predicted_class), confidence, all_probabilities


def get_default_image_path() -> Path:
    """
    Busca automaticamente a primeira imagem processada caso o usuário
    não informe uma imagem manualmente.
    """

    images = sorted(PROCESSED_IMAGES_DIR.glob("*.png"))

    if not images:
        raise FileNotFoundError(
            f"Nenhuma imagem .png encontrada em: {PROCESSED_IMAGES_DIR}"
        )

    return images[0]


def parse_args() -> argparse.Namespace:
    """
    Lê os argumentos enviados pelo terminal.
    """

    parser = argparse.ArgumentParser(
        description="Predição de imagens astronômicas usando modelo treinado."
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Caminho da imagem para predição. Se não informado, usa a primeira imagem processada.",
    )

    return parser.parse_args()

def infer_true_label_from_labels_file(image_path: Path) -> str:
    """
    Tenta descobrir a classe real da imagem usando o labels.csv.

    Se a imagem não estiver no labels.csv, retorna vazio.
    """

    if not LABELS_FILE.exists():
        return ""

    labels_df = pd.read_csv(LABELS_FILE)

    filename = image_path.name

    matches = labels_df[labels_df["filename"] == filename]

    if matches.empty:
        return ""

    return str(matches.iloc[0]["label"])


def append_predict_run_log(
    predict_run_id: str,
    checkpoint: dict[str, Any],
    image_path: Path,
    predicted_class: str,
    confidence: float,
    all_probabilities: dict[str, float],
    true_class: str,
) -> None:
    """
    Registra uma linha acumulativa com o resultado da predição.
    """

    is_correct = ""

    if true_class:
        is_correct = str(true_class == predicted_class)

    probabilities_formatted = {
        class_name: format_float(probability)
        for class_name, probability in all_probabilities.items()
    }

    fieldnames = [
        "predict_run_id",
        "train_run_id",
        "created_at",
        "model_file",
        "model_epoch",
        "image_path",
        "true_class",
        "predicted_class",
        "confidence",
        "is_correct",
        "probabilities",
    ]

    row = {
        "predict_run_id": predict_run_id,
        "train_run_id": checkpoint.get("train_run_id", ""),
        "created_at": now_iso(),
        "model_file": str(MODEL_FILE),
        "model_epoch": checkpoint.get("epoch", ""),
        "image_path": str(image_path),
        "true_class": true_class,
        "predicted_class": predicted_class,
        "confidence": format_float(confidence),
        "is_correct": is_correct,
        "probabilities": compact_dict(probabilities_formatted),
    }

    append_csv_row(
        file_path=PREDICT_RUNS_LOG_FILE,
        fieldnames=fieldnames,
        row=row,
    )

    print(f"Log acumulativo da predição salvo em: {PREDICT_RUNS_LOG_FILE}")

def main() -> None:
    """
    Fluxo principal de predição.
    """

    args = parse_args()

    predict_run_id = create_run_id("predict")
    print(f"Predict Run ID: {predict_run_id}")

    device = get_device()

    image_path = Path(args.image) if args.image else get_default_image_path()

    print(f"Imagem analisada: {image_path}")
    print(f"Dispositivo usado: {device}")

    checkpoint = load_checkpoint(MODEL_FILE, device)
    model = load_model(checkpoint, device)

    image_tensor = load_image(image_path)

    predicted_class, confidence, all_probabilities = predict_image(
        model=model,
        image_tensor=image_tensor,
        checkpoint=checkpoint,
        device=device,
    )

    true_class = infer_true_label_from_labels_file(image_path)

    if true_class:
        print(f"Classe real encontrada no labels.csv: {true_class}")
        print(f"Acertou: {true_class == predicted_class}")

    print(f"Classe prevista: {predicted_class}")
    print(f"Confiança: {confidence * 100:.2f}%")

    print("Probabilidades por classe:")

    append_predict_run_log(
        predict_run_id=predict_run_id,
        checkpoint=checkpoint,
        image_path=image_path,
        predicted_class=predicted_class,
        confidence=confidence,
        all_probabilities=all_probabilities,
        true_class=true_class,
    )

    for class_name, probability in all_probabilities.items():
        print(f"- {class_name}: {probability * 100:.2f}%")


if __name__ == "__main__":
    main()