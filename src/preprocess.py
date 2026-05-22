import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from astropy.io import fits

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_IMAGES_DIR,
    PROCESSED_ARRAYS_DIR,
    REJECTED_IMAGES_DIR,
    OUTPUT_REPORTS_DIR,
    LABELS_FILE,
    DEFAULT_IMAGE_SIZE,
    PREPROCESS_PERCENTILE_LOW,
    PREPROCESS_PERCENTILE_HIGH,
    MIN_IMAGE_STD,
    MAX_BLACK_PIXEL_RATIO,
    MAX_WHITE_PIXEL_RATIO,
    MAX_EXTREME_PIXEL_RATIO,
    BLACK_PIXEL_THRESHOLD,
    WHITE_PIXEL_THRESHOLD,
    ensure_directories,
    REJECT_FILENAME_PATTERNS,
    BRIGHT_OBJECT_PERCENTILE,
    MAX_BRIGHT_CENTROID_DISTANCE,
)


def sanitize_name(value: str) -> str:
    """
    Ajusta nomes para serem usados em arquivos.
    """

    return (
        value.strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )

def validate_filename_pattern(fits_path: Path) -> tuple[bool, str]:
    """
    Rejeita arquivos pelo padrão do nome.

    Alguns produtos FITS podem até conter dados numéricos,
    mas não são bons como imagem para treino visual.
    """

    file_stem = fits_path.stem.lower()

    for pattern in REJECT_FILENAME_PATTERNS:
        if pattern.lower() in file_stem:
            return False, f"padrao_arquivo_indesejado_{pattern.replace('_', '')}"

    return True, ""

def get_label_from_path(fits_path: Path) -> str | None:
    """
    Extrai a classe com base na estrutura data/raw/classe/alvo/arquivo.fits.

    Exemplo:
    data/raw/galaxia/M51/arquivo.fits

    Classe retornada:
    galaxia
    """

    relative_path = fits_path.relative_to(RAW_DATA_DIR)
    parts = relative_path.parts

    if len(parts) < 3:
        return None

    return parts[0]


def get_target_from_path(fits_path: Path) -> str:
    """
    Extrai o alvo com base na estrutura data/raw/classe/alvo/arquivo.fits.
    """

    relative_path = fits_path.relative_to(RAW_DATA_DIR)
    parts = relative_path.parts

    if len(parts) < 3:
        return "unknown_target"

    return parts[1]


def find_image_data(fits_path: Path) -> tuple[np.ndarray, int]:
    """
    Procura dentro do arquivo FITS o primeiro HDU que contenha dados de imagem.
    """

    with fits.open(fits_path) as hdul:
        for index, hdu in enumerate(hdul):
            data: Any = hdu.data

            if data is None:
                continue

            if not isinstance(data, np.ndarray):
                continue

            if not np.issubdtype(data.dtype, np.number):
                continue

            data = np.squeeze(data)

            # Se vier imagem 3D ou maior, pegamos a primeira fatia 2D.
            while data.ndim > 2:
                data = data[0]

            if data.ndim == 2:
                return data.astype(np.float32), index

    raise ValueError("Nenhuma imagem 2D numérica encontrada no arquivo FITS.")


def normalize_image(
    image: np.ndarray,
    low_percentile: int = PREPROCESS_PERCENTILE_LOW,
    high_percentile: int = PREPROCESS_PERCENTILE_HIGH,
) -> np.ndarray:
    """
    Normaliza a imagem para valores entre 0 e 255.

    O uso de percentis reduz o impacto de pixels extremos,
    algo comum em imagens astronômicas.
    """

    image = image.astype(np.float32)

    image[~np.isfinite(image)] = np.nan

    if np.isnan(image).all():
        raise ValueError("A imagem contém somente valores inválidos.")

    median_value = float(np.nanmedian(image))
    image = np.nan_to_num(image, nan=median_value)

    low_value = float(np.percentile(image, low_percentile))
    high_value = float(np.percentile(image, high_percentile))

    if high_value <= low_value:
        low_value = float(np.min(image))
        high_value = float(np.max(image))

    if high_value <= low_value:
        raise ValueError("Não foi possível normalizar a imagem.")

    image = np.clip(image, low_value, high_value)
    image = (image - low_value) / (high_value - low_value)
    image = image * 255.0

    return image.astype(np.uint8)


def resize_image(image: np.ndarray, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """
    Redimensiona a imagem para o tamanho padrão do modelo.
    """

    resized = cv2.resize(
        image,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )

    return resized


def calculate_image_quality_metrics(image: np.ndarray) -> dict[str, float]:
    """
    Calcula métricas simples para identificar imagens ruins.

    Essas métricas ajudam a detectar imagens muito vazias,
    muito brancas, muito pretas ou com pouco contraste.
    """

    image_float = image.astype(np.float32)

    total_pixels = float(image_float.size)

    black_pixels = float(np.sum(image_float <= BLACK_PIXEL_THRESHOLD))
    white_pixels = float(np.sum(image_float >= WHITE_PIXEL_THRESHOLD))

    black_ratio = black_pixels / total_pixels
    white_ratio = white_pixels / total_pixels
    extreme_ratio = black_ratio + white_ratio

    mean_value = float(np.mean(image_float))
    std_value = float(np.std(image_float))
    min_value = float(np.min(image_float))
    max_value = float(np.max(image_float))

    return {
        "mean": mean_value,
        "std": std_value,
        "min": min_value,
        "max": max_value,
        "black_ratio": black_ratio,
        "white_ratio": white_ratio,
        "extreme_ratio": extreme_ratio,
    }


def validate_image_quality(metrics: dict[str, float]) -> tuple[bool, str]:
    """
    Decide se uma imagem será aceita ou rejeitada para treino.
    """

    if metrics["std"] < MIN_IMAGE_STD:
        return False, "baixo_contraste"

    if metrics["black_ratio"] > MAX_BLACK_PIXEL_RATIO:
        return False, "muitos_pixels_pretos"

    if metrics["white_ratio"] > MAX_WHITE_PIXEL_RATIO:
        return False, "muitos_pixels_brancos"

    if metrics["extreme_ratio"] > MAX_EXTREME_PIXEL_RATIO:
        return False, "muitos_pixels_extremos"

    return True, ""

def calculate_bright_centroid_distance(image: np.ndarray) -> float:
    """
    Calcula a distância do centroide dos pixels mais brilhantes até o centro da imagem.

    Retorna um valor normalizado:
    - perto de 0: brilho principal está no centro;
    - perto de 1: brilho principal está muito deslocado para a borda.
    """

    image_float = image.astype(np.float32)

    threshold = float(np.percentile(image_float, BRIGHT_OBJECT_PERCENTILE))

    bright_mask = image_float >= threshold

    if not np.any(bright_mask):
        return 1.0

    y_positions, x_positions = np.where(bright_mask)

    centroid_y = float(np.mean(y_positions))
    centroid_x = float(np.mean(x_positions))

    height, width = image_float.shape

    center_y = height / 2.0
    center_x = width / 2.0

    distance = float(
        np.sqrt(
            ((centroid_y - center_y) ** 2) + ((centroid_x - center_x) ** 2)
        )
    )

    max_distance = float(
        np.sqrt(
            (center_y**2) + (center_x**2)
        )
    )

    return distance / max_distance


def validate_bright_object_position(
    image: np.ndarray,
    label: str,
) -> tuple[bool, str, float]:
    """
    Rejeita imagens de estrela quando o objeto mais brilhante está muito fora do centro.

    Isso ajuda a remover imagens com estrela cortada, no canto,
    ou com muitos artefatos de detector.
    """

    if label != "estrela":
        return True, "", 0.0

    distance = calculate_bright_centroid_distance(image)

    if distance > MAX_BRIGHT_CENTROID_DISTANCE:
        return False, "estrela_muito_fora_do_centro", distance

    return True, "", distance

def build_output_filename(fits_path: Path, label: str, target: str) -> str:
    """
    Cria um nome único para a imagem processada.
    """

    safe_label = sanitize_name(label)
    safe_target = sanitize_name(target)
    safe_stem = sanitize_name(fits_path.stem)

    return f"{safe_label}_{safe_target}_{safe_stem}"


def build_base_result(
    fits_path: Path,
    label: str,
    target: str,
    processed_filename: str,
    hdu_index: int,
    original_shape: tuple[int, ...],
    metrics: dict[str, float],
    status: str,
    rejection_reason: str,
    png_path: Path | None,
    npy_path: Path | None,
    error: str = "",
) -> dict[str, str]:
    """
    Monta uma linha padronizada para o relatório CSV.
    """

    return {
        "file_name": fits_path.name,
        "label": label,
        "target": target,
        "processed_filename": processed_filename,
        "hdu_index": str(hdu_index),
        "original_shape": str(original_shape),
        "mean": f"{metrics.get('mean', 0.0):.4f}",
        "std": f"{metrics.get('std', 0.0):.4f}",
        "black_ratio": f"{metrics.get('black_ratio', 0.0):.4f}",
        "white_ratio": f"{metrics.get('white_ratio', 0.0):.4f}",
        "extreme_ratio": f"{metrics.get('extreme_ratio', 0.0):.4f}",
        "bright_centroid_distance": f"{metrics.get('bright_centroid_distance', 0.0):.4f}",
        "png_path": str(png_path) if png_path else "",
        "npy_path": str(npy_path) if npy_path else "",
        "status": status,
        "rejection_reason": rejection_reason,
        "error": error,
    }


def process_fits_file(fits_path: Path) -> dict[str, str]:
    """
    Processa um único arquivo FITS.

    Se a imagem for aprovada, salva em data/processed/images e entra no labels.csv.
    Se for rejeitada, salva em data/processed/rejected e não entra no labels.csv.
    """

    label = get_label_from_path(fits_path)

    if label is None:
        raise ValueError(
            "Arquivo FITS ignorado porque não está em data/raw/classe/alvo/"
        )

    target = get_target_from_path(fits_path)

    # Cria o nome de saída logo no início para evitar erro de variável não definida
    output_name = build_output_filename(
        fits_path=fits_path,
        label=label,
        target=target,
    )

    png_filename = f"{output_name}.png"
    npy_filename = f"{output_name}.npy"

    # Primeiro filtro: rejeição por padrão no nome do arquivo
    is_filename_valid, filename_rejection_reason = validate_filename_pattern(fits_path)

    if not is_filename_valid:
        return build_base_result(
            fits_path=fits_path,
            label=label,
            target=target,
            processed_filename=png_filename,
            hdu_index=-1,
            original_shape=(),
            metrics={},
            status="rejected",
            rejection_reason=filename_rejection_reason,
            png_path=None,
            npy_path=None,
        )

    image_data, hdu_index = find_image_data(fits_path)

    original_shape = image_data.shape

    normalized_image = normalize_image(image_data)
    resized_image = resize_image(normalized_image)

    metrics = calculate_image_quality_metrics(resized_image)

    # Segundo filtro: qualidade geral da imagem
    is_valid, rejection_reason = validate_image_quality(metrics)

    # Terceiro filtro: posição do objeto brilhante para a classe estrela
    is_position_valid, position_rejection_reason, bright_centroid_distance = (
        validate_bright_object_position(
            image=resized_image,
            label=label,
        )
    )

    metrics["bright_centroid_distance"] = bright_centroid_distance

    if is_valid and not is_position_valid:
        is_valid = False
        rejection_reason = position_rejection_reason

    if is_valid:
        png_path = PROCESSED_IMAGES_DIR / png_filename
        npy_path = PROCESSED_ARRAYS_DIR / npy_filename

        cv2.imwrite(str(png_path), resized_image)
        np.save(npy_path, resized_image)

        return build_base_result(
            fits_path=fits_path,
            label=label,
            target=target,
            processed_filename=png_filename,
            hdu_index=hdu_index,
            original_shape=original_shape,
            metrics=metrics,
            status="success",
            rejection_reason="",
            png_path=png_path,
            npy_path=npy_path,
        )

    rejected_path = REJECTED_IMAGES_DIR / png_filename

    cv2.imwrite(str(rejected_path), resized_image)

    return build_base_result(
        fits_path=fits_path,
        label=label,
        target=target,
        processed_filename=png_filename,
        hdu_index=hdu_index,
        original_shape=original_shape,
        metrics=metrics,
        status="rejected",
        rejection_reason=rejection_reason,
        png_path=rejected_path,
        npy_path=None,
    )


def save_preprocess_report(rows: list[dict[str, str]]) -> None:
    """
    Salva um relatório CSV com o resultado do pré-processamento.
    """

    report_path = OUTPUT_REPORTS_DIR / "preprocess_report.csv"

    fieldnames = [
        "file_name",
        "label",
        "target",
        "processed_filename",
        "hdu_index",
        "original_shape",
        "mean",
        "std",
        "black_ratio",
        "white_ratio",
        "extreme_ratio",
        "bright_centroid_distance",
        "png_path",
        "npy_path",
        "status",
        "rejection_reason",
        "error",
    ]

    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Relatório salvo em: {report_path}")


def save_labels_file(rows: list[dict[str, str]]) -> None:
    """
    Gera o labels.csv automaticamente somente com imagens aprovadas.
    """

    success_rows = [
        row
        for row in rows
        if row["status"] == "success"
    ]

    with LABELS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "label"])

        for row in success_rows:
            writer.writerow(
                [
                    row["processed_filename"],
                    row["label"],
                ]
            )

    print(f"Labels salvos em: {LABELS_FILE}")
    print(f"Total de imagens aprovadas e rotuladas: {len(success_rows)}")


def print_summary(rows: list[dict[str, str]]) -> None:
    """
    Exibe um resumo do processamento.
    """

    total = len(rows)
    success = len([row for row in rows if row["status"] == "success"])
    rejected = len([row for row in rows if row["status"] == "rejected"])
    error = len([row for row in rows if row["status"] == "error"])

    print("-" * 80)
    print("Resumo do pré-processamento")
    print(f"Total analisado: {total}")
    print(f"Aprovadas: {success}")
    print(f"Rejeitadas: {rejected}")
    print(f"Erros: {error}")

    rejection_reasons: dict[str, int] = {}

    for row in rows:
        reason = row.get("rejection_reason", "")

        if not reason:
            continue

        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    if rejection_reasons:
        print("Motivos de rejeição:")

        for reason, count in rejection_reasons.items():
            print(f"- {reason}: {count}")


def main() -> None:
    """
    Executa o pré-processamento dos arquivos FITS em data/raw/classe/alvo.
    """

    ensure_directories()

    fits_files = sorted(RAW_DATA_DIR.rglob("*.fits"))

    if not fits_files:
        raise FileNotFoundError("Nenhum arquivo .fits encontrado em data/raw.")

    print(f"Arquivos FITS encontrados: {len(fits_files)}")

    report_rows: list[dict[str, str]] = []

    for fits_path in fits_files:
        print(f"Processando: {fits_path}")

        try:
            result = process_fits_file(fits_path)
            report_rows.append(result)

            status = result["status"]

            if status == "success":
                print(f"Aprovado: {fits_path.name}")

            elif status == "rejected":
                print(
                    f"Rejeitado: {fits_path.name} | "
                    f"Motivo: {result['rejection_reason']}"
                )

        except Exception as error:
            report_rows.append(
                {
                    "file_name": fits_path.name,
                    "label": "",
                    "target": "",
                    "processed_filename": "",
                    "hdu_index": "",
                    "original_shape": "",
                    "mean": "",
                    "std": "",
                    "black_ratio": "",
                    "white_ratio": "",
                    "extreme_ratio": "",
                    "bright_centroid_distance": "",
                    "png_path": "",
                    "npy_path": "",
                    "status": "error",
                    "rejection_reason": "",
                    "error": str(error),
                }
            )

            print(f"Erro ao processar {fits_path.name}: {error}")

    save_preprocess_report(report_rows)
    save_labels_file(report_rows)
    print_summary(report_rows)

    print("Pré-processamento finalizado.")


if __name__ == "__main__":
    main()