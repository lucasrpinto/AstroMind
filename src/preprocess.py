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
    OUTPUT_REPORTS_DIR,
    LABELS_FILE,
    DEFAULT_IMAGE_SIZE,
    PREPROCESS_PERCENTILE_LOW,
    PREPROCESS_PERCENTILE_HIGH,
    ensure_directories,
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


def get_label_from_path(fits_path: Path) -> str | None:
    """
    Extrai a classe com base na estrutura data/raw/classe/alvo/arquivo.fits.

    Exemplo:
    data/raw/galaxia/M51/arquivo.fits
    Classe retornada: galaxia
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


def build_output_filename(fits_path: Path, label: str, target: str) -> str:
    """
    Cria um nome único para a imagem processada.
    """

    safe_label = sanitize_name(label)
    safe_target = sanitize_name(target)
    safe_stem = sanitize_name(fits_path.stem)

    return f"{safe_label}_{safe_target}_{safe_stem}"


def process_fits_file(fits_path: Path) -> dict[str, str]:
    """
    Processa um único arquivo FITS e salva PNG e NPY.
    """

    label = get_label_from_path(fits_path)

    if label is None:
        raise ValueError(
            "Arquivo FITS ignorado porque não está em data/raw/classe/alvo/"
        )

    target = get_target_from_path(fits_path)

    image_data, hdu_index = find_image_data(fits_path)

    original_shape = image_data.shape

    normalized_image = normalize_image(image_data)
    resized_image = resize_image(normalized_image)

    output_name = build_output_filename(
        fits_path=fits_path,
        label=label,
        target=target,
    )

    png_filename = f"{output_name}.png"
    npy_filename = f"{output_name}.npy"

    png_path = PROCESSED_IMAGES_DIR / png_filename
    npy_path = PROCESSED_ARRAYS_DIR / npy_filename

    cv2.imwrite(str(png_path), resized_image)
    np.save(npy_path, resized_image)

    return {
        "file_name": fits_path.name,
        "label": label,
        "target": target,
        "processed_filename": png_filename,
        "hdu_index": str(hdu_index),
        "original_shape": str(original_shape),
        "png_path": str(png_path),
        "npy_path": str(npy_path),
        "status": "success",
        "error": "",
    }


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
        "png_path",
        "npy_path",
        "status",
        "error",
    ]

    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Relatório salvo em: {report_path}")


def save_labels_file(rows: list[dict[str, str]]) -> None:
    """
    Gera o labels.csv automaticamente a partir dos arquivos processados com sucesso.
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
    print(f"Total de imagens rotuladas: {len(success_rows)}")


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

            print(f"Finalizado: {fits_path.name}")

        except Exception as error:
            report_rows.append(
                {
                    "file_name": fits_path.name,
                    "label": "",
                    "target": "",
                    "processed_filename": "",
                    "hdu_index": "",
                    "original_shape": "",
                    "png_path": "",
                    "npy_path": "",
                    "status": "error",
                    "error": str(error),
                }
            )

            print(f"Erro ao processar {fits_path.name}: {error}")

    save_preprocess_report(report_rows)
    save_labels_file(report_rows)

    print("Pré-processamento finalizado.")


if __name__ == "__main__":
    main()