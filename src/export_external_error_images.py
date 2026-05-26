from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

from src.config import (
    EXTERNAL_IMAGES_DIR,
    EXTERNAL_ERRORS_REPORT_FILE,
    EXTERNAL_ERROR_IMAGES_DIR,
    DEFAULT_IMAGE_SIZE,
    PREPROCESS_PERCENTILE_LOW,
    PREPROCESS_PERCENTILE_HIGH,
    ensure_directories,
)


def normalize_fits_image(
    image: np.ndarray,
    low_percentile: int = PREPROCESS_PERCENTILE_LOW,
    high_percentile: int = PREPROCESS_PERCENTILE_HIGH,
) -> np.ndarray:
    """
    Normaliza uma imagem FITS para valores entre 0 e 255.
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


def load_fits_image(fits_path: Path) -> np.ndarray:
    """
    Carrega a primeira imagem 2D encontrada em um arquivo FITS.
    """

    with fits.open(fits_path) as hdul:
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
                return data.astype(np.float32)

    raise ValueError(f"Nenhuma imagem 2D encontrada em: {fits_path}")


def convert_error_image(row: pd.Series) -> dict[str, str]:
    """
    Converte uma imagem externa com erro para PNG.
    """

    filename = str(row["filename"])
    true_class = str(row["true_class"])
    predicted_class = str(row["predicted_class"])
    confidence = float(row["confidence"])

    image_path = EXTERNAL_IMAGES_DIR / filename

    if not image_path.exists():
        raise FileNotFoundError(f"Imagem externa não encontrada: {image_path}")

    image_data = load_fits_image(image_path)
    normalized = normalize_fits_image(image_data)

    resized = cv2.resize(
        normalized,
        (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    output_filename = (
        f"error_real_{true_class}_pred_{predicted_class}_"
        f"conf_{confidence:.2f}_{Path(filename).stem}.png"
    )

    output_path = EXTERNAL_ERROR_IMAGES_DIR / output_filename

    cv2.imwrite(str(output_path), resized)

    return {
        "filename": filename,
        "true_class": true_class,
        "predicted_class": predicted_class,
        "confidence": f"{confidence:.6f}",
        "png_path": str(output_path),
    }


def create_contact_sheet(converted_rows: list[dict[str, str]]) -> None:
    """
    Gera uma imagem única com todos os erros externos lado a lado.
    """

    if not converted_rows:
        print("Nenhuma imagem para gerar contact sheet.")
        return

    total_images = len(converted_rows)
    columns = 2
    rows = int(np.ceil(total_images / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 5, rows * 5),
    )

    if total_images == 1:
        axes = np.array([axes])

    axes = np.array(axes).reshape(-1)

    for axis in axes:
        axis.axis("off")

    for index, row in enumerate(converted_rows):
        png_path = Path(row["png_path"])
        image = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)

        axis = axes[index]
        axis.imshow(image, cmap="gray")
        axis.axis("off")

        title = (
            f"Real: {row['true_class']}\n"
            f"Previsto: {row['predicted_class']}\n"
            f"Conf: {float(row['confidence']) * 100:.2f}%"
        )

        axis.set_title(title)

    contact_sheet_path = EXTERNAL_ERROR_IMAGES_DIR / "external_errors_contact_sheet.png"

    plt.tight_layout()
    plt.savefig(contact_sheet_path, bbox_inches="tight")
    plt.close()

    print(f"Contact sheet salvo em: {contact_sheet_path}")


def main() -> None:
    """
    Exporta imagens dos erros externos para análise visual.
    """

    ensure_directories()

    if not EXTERNAL_ERRORS_REPORT_FILE.exists():
        raise FileNotFoundError(
            f"Relatório de erros externos não encontrado: {EXTERNAL_ERRORS_REPORT_FILE}"
        )

    errors_df = pd.read_csv(EXTERNAL_ERRORS_REPORT_FILE)

    if errors_df.empty:
        print("Nenhum erro externo encontrado.")
        return

    converted_rows: list[dict[str, str]] = []

    for _, row in errors_df.iterrows():
        try:
            converted_row = convert_error_image(row)
            converted_rows.append(converted_row)

            print(f"Imagem convertida: {converted_row['png_path']}")

        except Exception as error:
            print(f"Erro ao converter {row.get('filename', '')}: {error}")

    create_contact_sheet(converted_rows)

    print("-" * 80)
    print("Exportação visual dos erros externos finalizada.")


if __name__ == "__main__":
    main()