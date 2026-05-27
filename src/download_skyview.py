import csv
from pathlib import Path
from typing import Any, cast

from astropy import units as u
from astroquery.skyview import SkyView

from src.config import (
    RAW_DATA_DIR,
    SKYVIEW_TARGETS,
    SKYVIEW_DOWNLOAD_REPORT_FILE,
    ensure_directories,
)


# Evita falsos erros do Pylance com métodos dinâmicos do astroquery
SKYVIEW: Any = cast(Any, SkyView)


def sanitize_name(value: str) -> str:
    """
    Ajusta nomes para serem usados em pastas e arquivos.
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


def build_output_path(
    label: str,
    target: str,
    survey: str,
    width_degrees: float,
    index: int,
) -> Path:
    """
    Monta o caminho final do arquivo FITS baixado via SkyView.

    Inclui o tamanho do recorte no nome para evitar sobrescrever arquivos
    quando usamos múltiplos width_degrees para o mesmo alvo.
    """

    safe_label = sanitize_name(label)
    safe_target = sanitize_name(target)
    safe_survey = sanitize_name(survey)
    safe_width = str(width_degrees).replace(".", "p")

    target_dir = RAW_DATA_DIR / safe_label / safe_target
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"skyview_{safe_label}_{safe_target}_"
        f"{safe_survey}_{safe_width}deg_{index}.fits"
    )

    return target_dir / filename


def download_one_skyview_image(
    label: str,
    target: str,
    survey: str,
    width_degrees: float,
    pixels: int,
) -> list[dict[str, str]]:
    """
    Baixa uma ou mais imagens FITS do SkyView para um alvo e survey.
    """

    print("-" * 80)
    print(f"Classe: {label}")
    print(f"Alvo: {target}")
    print(f"Survey: {survey}")
    print(f"Largura do recorte: {width_degrees} graus")
    print(f"Pixels: {pixels}")

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
                    "label": label,
                    "target": target,
                    "survey": survey,
                    "file_path": "",
                    "status": "empty",
                    "error": "Nenhuma imagem retornada pelo SkyView.",
                }
            )

            print("Nenhuma imagem retornada.")
            return rows

        for index, hdu_list in enumerate(images):
            output_path = build_output_path(
                label=label,
                target=target,
                survey=survey,
                width_degrees=width_degrees,
                index=index,
            )

            try:
                hdu_list.writeto(
                    output_path,
                    overwrite=True,
                )

                rows.append(
                    {
                        "label": label,
                        "target": target,
                        "survey": survey,
                        "file_path": str(output_path),
                        "status": "success",
                        "error": "",
                    }
                )

                print(f"Arquivo salvo em: {output_path}")

            finally:
                hdu_list.close()

    except Exception as error:
        rows.append(
            {
                "label": label,
                "target": target,
                "survey": survey,
                "file_path": "",
                "status": "error",
                "error": str(error),
            }
        )

        print(f"Erro ao baixar {target} - {survey}: {error}")

    return rows


def process_skyview_target(target_config: dict[str, Any]) -> list[dict[str, str]]:
    """
    Processa um item do SKYVIEW_TARGETS.

    Suporta dois formatos:
    - width_degrees: float
    - width_degrees_list: list[float]
    """

    label = str(target_config["label"])
    target = str(target_config["target"])
    surveys = target_config["surveys"]
    pixels = int(target_config["pixels"])

    if "width_degrees_list" in target_config:
        width_degrees_values = [
            float(value)
            for value in target_config["width_degrees_list"]
        ]
    else:
        width_degrees_values = [
            float(target_config["width_degrees"])
        ]

    all_rows: list[dict[str, str]] = []

    for width_degrees in width_degrees_values:
        for survey in surveys:
            rows = download_one_skyview_image(
                label=label,
                target=target,
                survey=str(survey),
                width_degrees=width_degrees,
                pixels=pixels,
            )

            all_rows.extend(rows)

    return all_rows


def save_report(rows: list[dict[str, str]]) -> None:
    """
    Salva o relatório dos downloads do SkyView.
    """

    fieldnames = [
        "label",
        "target",
        "survey",
        "file_path",
        "status",
        "error",
    ]

    with SKYVIEW_DOWNLOAD_REPORT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("-" * 80)
    print(f"Relatório salvo em: {SKYVIEW_DOWNLOAD_REPORT_FILE}")


def print_summary(rows: list[dict[str, str]]) -> None:
    """
    Exibe resumo dos downloads.
    """

    total = len(rows)
    success = len([row for row in rows if row["status"] == "success"])
    empty = len([row for row in rows if row["status"] == "empty"])
    error = len([row for row in rows if row["status"] == "error"])

    print("-" * 80)
    print("Resumo do download SkyView")
    print(f"Total de tentativas: {total}")
    print(f"Sucesso: {success}")
    print(f"Sem retorno: {empty}")
    print(f"Erro: {error}")


def main() -> None:
    """
    Baixa imagens complementares do SkyView.
    """

    ensure_directories()

    all_rows: list[dict[str, str]] = []

    for target_config in SKYVIEW_TARGETS:
        rows = process_skyview_target(target_config)
        all_rows.extend(rows)

    save_report(all_rows)
    print_summary(all_rows)

    print("Download SkyView finalizado.")


if __name__ == "__main__":
    main()