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
    DEFAULT_IMAGE_SIZE,
    PREPROCESS_PERCENTILE_LOW,
    PREPROCESS_PERCENTILE_HIGH,
    ensure_directories,
)


def find_image_data(fits_path: Path) -> tuple[np.ndarray, int]:
    """
    Procura dentro do arquivo FITS o primeiro HDU que contenha dados de imagem.

    Um arquivo FITS pode ter várias extensões internas.
    Algumas têm metadados, outras têm tabelas e outras têm imagens.
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

            # Caso venha uma imagem 3D ou mais, pega a primeira fatia 2D.
            # Isso é suficiente para o primeiro processamento.
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

    Usa percentis para reduzir o impacto de pixels muito extremos,
    muito comuns em imagens astronômicas.
    """

    image = image.astype(np.float32)

    # Remove infinitos, mantendo como NaN para tratamento posterior
    image[~np.isfinite(image)] = np.nan

    if np.isnan(image).all():
        raise ValueError("A imagem contém somente valores inválidos.")

    median_value = float(np.nanmedian(image))
    image = np.nan_to_num(image, nan=median_value)

    low_value = np.percentile(image, low_percentile)
    high_value = np.percentile(image, high_percentile)

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

    No início usa 224x224 porque é um tamanho comum em modelos
    de visão computacional.
    """

    resized = cv2.resize(
        image,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )

    return resized


def process_fits_file(fits_path: Path) -> dict[str, str]:
    """
    Processa um único arquivo FITS e salva as versões PNG e NPY.
    """

    image_data, hdu_index = find_image_data(fits_path)

    original_shape = image_data.shape

    normalized_image = normalize_image(image_data)
    resized_image = resize_image(normalized_image)

    output_stem = fits_path.stem

    png_path = PROCESSED_IMAGES_DIR / f"{output_stem}.png"
    npy_path = PROCESSED_ARRAYS_DIR / f"{output_stem}.npy"

    # Salva imagem para visualização
    cv2.imwrite(str(png_path), resized_image)

    # Salva array numérico para uso futuro no treinamento
    np.save(npy_path, resized_image)

    return {
        "file_name": fits_path.name,
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


def main() -> None:
    """
    Executa o pré-processamento de todos os arquivos FITS em data/raw.
    """

    ensure_directories()

    fits_files = sorted(RAW_DATA_DIR.glob("*.fits"))

    if not fits_files:
        raise FileNotFoundError("Nenhum arquivo .fits encontrado em data/raw.")

    print(f"Arquivos FITS encontrados: {len(fits_files)}")

    report_rows: list[dict[str, str]] = []

    for fits_path in fits_files:
        print(f"Processando: {fits_path.name}")

        try:
            result = process_fits_file(fits_path)
            report_rows.append(result)
            print(f"Finalizado: {fits_path.name}")

        except Exception as error:
            report_rows.append(
                {
                    "file_name": fits_path.name,
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

    print("Pré-processamento finalizado.")


if __name__ == "__main__":
    main()