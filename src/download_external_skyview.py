import csv
from pathlib import Path
from typing import Any, cast

from astropy import units as u
from astroquery.skyview import SkyView

from src.config import (
    EXTERNAL_IMAGES_DIR,
    EXTERNAL_LABELS_FILE,
    EXTERNAL_SKYVIEW_TARGETS,
    OUTPUT_REPORTS_DIR,
    ensure_directories,
)


SKYVIEW: Any = cast(Any, SkyView)

EXTERNAL_SKYVIEW_DOWNLOAD_REPORT_FILE = (
    OUTPUT_REPORTS_DIR / "external_skyview_download_report.csv"
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
        .replace("*", "")
        .replace("?", "")
        .replace('"', "")
        .replace("<", "")
        .replace(">", "")
        .replace("|", "")
    )


def build_external_image_filename(
    label: str,
    target: str,
    survey: str,
    index: int,
) -> str:
    """
    Cria o nome da imagem externa.
    """

    safe_label = sanitize_name(label)
    safe_target = sanitize_name(target)
    safe_survey = sanitize_name(survey)

    return f"external_{safe_label}_{safe_target}_{safe_survey}_{index}.fits"


def download_one_external_image(
    label: str,
    target: str,
    survey: str,
    width_degrees: float,
    pixels: int,
) -> list[dict[str, str]]:
    """
    Baixa uma imagem externa do SkyView.
    """

    print("-" * 80)
    print(f"Classe externa: {label}")
    print(f"Alvo externo: {target}")
    print(f"Survey: {survey}")

    rows: list[dict[str, str]] = []

    try:
        images = SKYVIEW.get_images(
            position=target,
            survey=[survey],
            pixels=str(pixels),
            width=width_degrees * u.deg,
            height=width_degrees * u.deg,
            coordinates="J2000",
            projection="Tan",
            cache=True,
            show_progress=False,
        )

        if not images:
            rows.append(
                {
                    "filename": "",
                    "label": label,
                    "target": target,
                    "survey": survey,
                    "status": "empty",
                    "error": "Nenhuma imagem retornada pelo SkyView.",
                }
            )

            print("Nenhuma imagem retornada.")
            return rows

        for index, hdu_list in enumerate(images):
            filename = build_external_image_filename(
                label=label,
                target=target,
                survey=survey,
                index=index,
            )

            output_path = EXTERNAL_IMAGES_DIR / filename

            try:
                hdu_list.writeto(
                    output_path,
                    overwrite=True,
                )

                rows.append(
                    {
                        "filename": filename,
                        "label": label,
                        "target": target,
                        "survey": survey,
                        "status": "success",
                        "error": "",
                    }
                )

                print(f"Arquivo externo salvo em: {output_path}")

            finally:
                hdu_list.close()

    except Exception as error:
        rows.append(
            {
                "filename": "",
                "label": label,
                "target": target,
                "survey": survey,
                "status": "error",
                "error": str(error),
            }
        )

        print(f"Erro ao baixar {target} - {survey}: {error}")

    return rows


def process_external_target(target_config: dict[str, Any]) -> list[dict[str, str]]:
    """
    Processa um item da lista EXTERNAL_SKYVIEW_TARGETS.
    """

    label = str(target_config["label"])
    target = str(target_config["target"])
    surveys = target_config["surveys"]
    width_degrees = float(target_config["width_degrees"])
    pixels = int(target_config["pixels"])

    all_rows: list[dict[str, str]] = []

    for survey in surveys:
        rows = download_one_external_image(
            label=label,
            target=target,
            survey=str(survey),
            width_degrees=width_degrees,
            pixels=pixels,
        )

        all_rows.extend(rows)

    return all_rows


def save_download_report(rows: list[dict[str, str]]) -> None:
    """
    Salva relatório dos downloads externos.
    """

    fieldnames = [
        "filename",
        "label",
        "target",
        "survey",
        "status",
        "error",
    ]

    with EXTERNAL_SKYVIEW_DOWNLOAD_REPORT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Relatório salvo em: {EXTERNAL_SKYVIEW_DOWNLOAD_REPORT_FILE}")


def save_external_labels(rows: list[dict[str, str]]) -> None:
    """
    Gera o external_labels.csv somente com downloads bem-sucedidos.
    """

    success_rows = [
        row
        for row in rows
        if row["status"] == "success"
    ]

    with EXTERNAL_LABELS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "label"])

        for row in success_rows:
            writer.writerow(
                [
                    row["filename"],
                    row["label"],
                ]
            )

    print(f"Labels externos salvos em: {EXTERNAL_LABELS_FILE}")
    print(f"Total de imagens externas rotuladas: {len(success_rows)}")


def print_summary(rows: list[dict[str, str]]) -> None:
    """
    Exibe resumo do download externo.
    """

    total = len(rows)
    success = len([row for row in rows if row["status"] == "success"])
    empty = len([row for row in rows if row["status"] == "empty"])
    error = len([row for row in rows if row["status"] == "error"])

    print("-" * 80)
    print("Resumo do download externo SkyView")
    print(f"Total de tentativas: {total}")
    print(f"Sucesso: {success}")
    print(f"Sem retorno: {empty}")
    print(f"Erro: {error}")


def main() -> None:
    """
    Baixa imagens externas para teste de generalização do modelo.
    """

    ensure_directories()

    all_rows: list[dict[str, str]] = []

    for target_config in EXTERNAL_SKYVIEW_TARGETS:
        rows = process_external_target(target_config)
        all_rows.extend(rows)

    save_download_report(all_rows)
    save_external_labels(all_rows)
    print_summary(all_rows)

    print("Download externo finalizado.")


if __name__ == "__main__":
    main()