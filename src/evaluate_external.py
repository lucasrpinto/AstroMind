from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
from PIL import Image
from torch import Tensor

import numpy as np
import cv2
from astropy.io import fits

from src.config import (
    MODEL_FILE,
    EXTERNAL_IMAGES_DIR,
    EXTERNAL_LABELS_FILE,
    EXTERNAL_EVALUATION_REPORT_FILE,
    EXTERNAL_CONFUSION_MATRIX_FILE,
    EXTERNAL_EVALUATE_RUNS_LOG_FILE,
    DEFAULT_IMAGE_SIZE,
    PREPROCESS_PERCENTILE_LOW,
    PREPROCESS_PERCENTILE_HIGH,
    ensure_directories,
)
from src.dataset import get_eval_transforms
from src.experiment_logger import (
    append_csv_row,
    compact_dict,
    create_run_id,
    format_float,
    now_iso,
)

from torch import nn

from src.model_registry import build_model
from src.train import get_device


def load_checkpoint(device: torch.device) -> dict[str, Any]:
    """
    Carrega o melhor modelo salvo.
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
    Recria o modelo com base na versão salva no checkpoint.
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
    Retorna o nome da classe a partir do índice.
    """

    class_name = idx_to_class.get(
        index,
        idx_to_class.get(str(index), f"classe_{index}"),
    )

    return str(class_name)


def load_external_labels() -> pd.DataFrame:
    """
    Carrega o CSV com as imagens externas e seus rótulos reais.
    """

    if not EXTERNAL_LABELS_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {EXTERNAL_LABELS_FILE}. "
            "Crie o arquivo com as colunas filename,label."
        )

    labels_df = pd.read_csv(EXTERNAL_LABELS_FILE)

    if labels_df.empty:
        raise ValueError("O external_labels.csv está vazio.")

    if "filename" not in labels_df.columns:
        raise ValueError("O external_labels.csv precisa conter a coluna filename.")

    if "label" not in labels_df.columns:
        labels_df["label"] = ""

    labels_df["label"] = labels_df["label"].fillna("")

    return labels_df


def normalize_fits_image(
    image: np.ndarray,
    low_percentile: int = PREPROCESS_PERCENTILE_LOW,
    high_percentile: int = PREPROCESS_PERCENTILE_HIGH,
) -> np.ndarray:
    """
    Normaliza uma imagem FITS para 0-255.
    """

    image = image.astype(np.float32)

    image[~np.isfinite(image)] = np.nan

    if np.isnan(image).all():
        raise ValueError("A imagem FITS contém somente valores inválidos.")

    median_value = float(np.nanmedian(image))
    image = np.nan_to_num(image, nan=median_value)

    low_value = float(np.percentile(image, low_percentile))
    high_value = float(np.percentile(image, high_percentile))

    if high_value <= low_value:
        low_value = float(np.min(image))
        high_value = float(np.max(image))

    if high_value <= low_value:
        raise ValueError("Não foi possível normalizar a imagem FITS.")

    image = np.clip(image, low_value, high_value)
    image = (image - low_value) / (high_value - low_value)
    image = image * 255.0

    return image.astype(np.uint8)


def load_fits_as_rgb_image(image_path: Path) -> Image.Image:
    """
    Carrega um FITS externo e transforma em imagem RGB para o modelo.
    """

    with fits.open(image_path) as hdul:
        image_data = None

        for hdu in hdul:
            data = hdu.data

            if data is None:
                continue

            if not isinstance(data, np.ndarray):
                continue

            data = np.squeeze(data)

            while data.ndim > 2:
                data = data[0]

            if data.ndim == 2:
                image_data = data.astype(np.float32)
                break

        if image_data is None:
            raise ValueError("Nenhuma imagem 2D encontrada no FITS externo.")

    normalized = normalize_fits_image(image_data)

    resized = cv2.resize(
        normalized,
        (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(rgb)


def load_image_tensor(image_path: Path) -> Tensor:
    """
    Carrega uma imagem externa comum ou FITS e aplica as transformações de avaliação.
    """

    if not image_path.exists():
        raise FileNotFoundError(f"Imagem externa não encontrada: {image_path}")

    suffix = image_path.suffix.lower()

    if suffix == ".fits":
        image = load_fits_as_rgb_image(image_path)
    else:
        image = Image.open(image_path).convert("RGB")

    transform = get_eval_transforms()
    image_tensor = transform(image)

    return image_tensor.unsqueeze(0)

@torch.no_grad()
def predict_external_image(
    model: nn.Module,
    image_tensor: Tensor,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[str, float, dict[str, float]]:
    """
    Executa a predição de uma imagem externa.
    """

    image_tensor = image_tensor.to(device)

    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)

    predicted_index = int(torch.argmax(probabilities, dim=1).item())
    confidence = float(probabilities[0][predicted_index].item())

    idx_to_class = checkpoint["idx_to_class"]

    predicted_class = get_class_name(
        idx_to_class=idx_to_class,
        index=predicted_index,
    )

    all_probabilities: dict[str, float] = {}

    for index, probability in enumerate(probabilities[0]):
        class_name = get_class_name(
            idx_to_class=idx_to_class,
            index=index,
        )

        all_probabilities[class_name] = float(probability.item())

    return predicted_class, confidence, all_probabilities


def evaluate_external_images() -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Avalia o modelo nas imagens externas informadas no external_labels.csv.
    """

    ensure_directories()

    external_run_id = create_run_id("external_evaluate")
    print(f"External Evaluate Run ID: {external_run_id}")

    device = get_device()
    print(f"Dispositivo usado: {device}")

    checkpoint = load_checkpoint(device)
    model = load_model(checkpoint, device)

    labels_df = load_external_labels()

    results: list[dict[str, Any]] = []

    for _, row in labels_df.iterrows():
        filename = str(row["filename"])
        true_class = str(row["label"]).strip()

        image_path = EXTERNAL_IMAGES_DIR / filename

        try:
            image_tensor = load_image_tensor(image_path)

            predicted_class, confidence, probabilities = predict_external_image(
                model=model,
                image_tensor=image_tensor,
                checkpoint=checkpoint,
                device=device,
            )

            is_correct = ""

            if true_class:
                is_correct = true_class == predicted_class

            result_row: dict[str, Any] = {
                "external_run_id": external_run_id,
                "filename": filename,
                "image_path": str(image_path),
                "true_class": true_class,
                "predicted_class": predicted_class,
                "confidence": round(confidence, 6),
                "is_correct": is_correct,
                "status": "success",
                "error": "",
            }

            for class_name, probability in probabilities.items():
                result_row[f"prob_{class_name}"] = round(probability, 6)

            results.append(result_row)

            print(
                f"{filename} | "
                f"Real: {true_class if true_class else 'não informado'} | "
                f"Previsto: {predicted_class} | "
                f"Confiança: {confidence * 100:.2f}%"
            )

        except Exception as error:
            results.append(
                {
                    "external_run_id": external_run_id,
                    "filename": filename,
                    "image_path": str(image_path),
                    "true_class": true_class,
                    "predicted_class": "",
                    "confidence": "",
                    "is_correct": "",
                    "status": "error",
                    "error": str(error),
                }
            )

            print(f"Erro ao avaliar {filename}: {error}")

    results_df = pd.DataFrame(results)

    return results_df, checkpoint


def save_external_report(results_df: pd.DataFrame) -> None:
    """
    Salva o relatório detalhado das imagens externas.
    """

    results_df.to_csv(
        EXTERNAL_EVALUATION_REPORT_FILE,
        index=False,
        encoding="utf-8",
    )

    print(f"Relatório externo salvo em: {EXTERNAL_EVALUATION_REPORT_FILE}")


def save_external_confusion_matrix(results_df: pd.DataFrame) -> None:
    """
    Salva a matriz de confusão quando houver classe real informada.
    """

    valid_df = results_df[
        (results_df["status"] == "success")
        & (results_df["true_class"].astype(str).str.strip() != "")
    ]

    if valid_df.empty:
        print("Nenhuma classe real informada. Matriz de confusão não gerada.")
        return

    confusion_matrix_df = pd.crosstab(
        valid_df["true_class"],
        valid_df["predicted_class"],
        rownames=["classe_real"],
        colnames=["classe_prevista"],
        dropna=False,
    )

    confusion_matrix_df.to_csv(
        EXTERNAL_CONFUSION_MATRIX_FILE,
        encoding="utf-8",
    )

    print(f"Matriz de confusão externa salva em: {EXTERNAL_CONFUSION_MATRIX_FILE}")
    print("-" * 80)
    print("Matriz de confusão externa:")
    print(confusion_matrix_df)


def append_external_evaluate_run_log(
    results_df: pd.DataFrame,
    checkpoint: dict[str, Any],
) -> None:
    """
    Registra uma linha acumulativa da avaliação externa.
    """

    success_df = results_df[results_df["status"] == "success"]

    labeled_df = success_df[
        success_df["true_class"].astype(str).str.strip() != ""
    ]

    total = len(success_df)
    labeled_total = len(labeled_df)

    correct = 0
    accuracy = 0.0

    if labeled_total > 0:
        correct = int(labeled_df["is_correct"].sum())
        accuracy = correct / labeled_total

    class_accuracy: dict[str, str] = {}

    if labeled_total > 0:
        grouped = labeled_df.groupby("true_class")

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

    external_run_id = str(results_df.iloc[0]["external_run_id"]) if not results_df.empty else ""

    row = {
        "external_evaluate_run_id": external_run_id,
        "train_run_id": checkpoint.get("train_run_id", ""),
        "created_at": now_iso(),
        "model_file": str(MODEL_FILE),
        "model_epoch": checkpoint.get("epoch", ""),
        "total_images": total,
        "labeled_images": labeled_total,
        "accuracy": format_float(accuracy),
        "correct": correct,
        "errors": labeled_total - correct,
        "class_accuracy": compact_dict(class_accuracy),
        "external_labels_file": str(EXTERNAL_LABELS_FILE),
        "external_report_file": str(EXTERNAL_EVALUATION_REPORT_FILE),
        "external_confusion_matrix_file": str(EXTERNAL_CONFUSION_MATRIX_FILE),
    }

    append_csv_row(
        file_path=EXTERNAL_EVALUATE_RUNS_LOG_FILE,
        fieldnames=fieldnames,
        row=row,
    )

    print(f"Log acumulativo externo salvo em: {EXTERNAL_EVALUATE_RUNS_LOG_FILE}")


def print_external_summary(results_df: pd.DataFrame) -> None:
    """
    Exibe resumo da avaliação externa.
    """

    success_df = results_df[results_df["status"] == "success"]

    labeled_df = success_df[
        success_df["true_class"].astype(str).str.strip() != ""
    ]

    print("-" * 80)
    print("Resumo da avaliação externa")
    print(f"Total de imagens avaliadas com sucesso: {len(success_df)}")
    print(f"Total com classe real informada: {len(labeled_df)}")

    if not labeled_df.empty:
        correct = int(labeled_df["is_correct"].sum())
        total = len(labeled_df)
        accuracy = correct / total

        print(f"Acurácia externa: {accuracy * 100:.2f}%")
        print(f"Acertos: {correct}/{total}")

        print("-" * 80)
        print("Acurácia externa por classe:")

        grouped = labeled_df.groupby("true_class")

        for class_name, group in grouped:
            class_total = len(group)
            class_correct = int(group["is_correct"].sum())
            class_acc = class_correct / class_total

            print(f"{class_name}: {class_acc * 100:.2f}% ({class_correct}/{class_total})")


def main() -> None:
    """
    Executa a avaliação externa.
    """

    results_df, checkpoint = evaluate_external_images()

    save_external_report(results_df)
    save_external_confusion_matrix(results_df)
    print_external_summary(results_df)

    append_external_evaluate_run_log(
        results_df=results_df,
        checkpoint=checkpoint,
    )

    print("Avaliação externa finalizada.")


if __name__ == "__main__":
    main()