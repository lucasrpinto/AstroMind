from pathlib import Path
from typing import Any, cast

from astroquery.mast import Observations

from src.config import (
    DATASET_TARGETS,
    RAW_DATA_DIR,
    OUTPUT_REPORTS_DIR,
    ensure_directories,
)


# Evita erro falso do Pylance com métodos dinâmicos do astroquery
MAST: Any = cast(Any, Observations)


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
    )


def save_table_report(table: Any, filename: str) -> None:
    """
    Salva uma tabela retornada pelo astroquery em CSV.
    """

    output_path = OUTPUT_REPORTS_DIR / filename

    table.write(
        output_path,
        format="csv",
        overwrite=True,
    )

    print(f"Relatório salvo em: {output_path}")


def query_target_observations(
    target: str,
    mission: str,
    radius: str,
    limit_observations: int,
):
    """
    Busca observações de um alvo específico no MAST.
    """

    print(f"Buscando observações para: {target}")
    print(f"Missão: {mission}")

    observations = MAST.query_object(
        target,
        radius=radius,
    )

    if len(observations) == 0:
        print(f"Nenhuma observação encontrada para {target}.")
        return None

    if "obs_collection" in observations.colnames:
        observations = observations[
            observations["obs_collection"] == mission.upper()
        ]

    if "dataproduct_type" in observations.colnames:
        observations = observations[
            observations["dataproduct_type"] == "image"
        ]

    if "intentType" in observations.colnames:
        observations = observations[
            observations["intentType"] == "science"
        ]

    if len(observations) == 0:
        print(f"Nenhuma observação compatível encontrada para {target}.")
        return None

    observations = observations[:limit_observations]

    print(f"Observações selecionadas para {target}: {len(observations)}")

    return observations


def get_target_products(
    observations,
    limit_products: int,
):
    """
    Busca produtos FITS científicos para as observações encontradas.
    """

    products = MAST.get_product_list(observations)

    if len(products) == 0:
        print("Nenhum produto encontrado.")
        return None

    filtered_products = MAST.filter_products(
        products,
        productType="SCIENCE",
        extension="fits",
    )

    if len(filtered_products) == 0:
        print("Nenhum produto FITS científico encontrado.")
        return None

    filtered_products = filtered_products[:limit_products]

    print(f"Produtos FITS selecionados: {len(filtered_products)}")

    return filtered_products


def download_target_products(
    products,
    label: str,
    target: str,
) -> None:
    """
    Baixa produtos FITS dentro da estrutura data/raw/classe/alvo.
    """

    safe_label = sanitize_name(label)
    safe_target = sanitize_name(target)

    target_dir = RAW_DATA_DIR / safe_label / safe_target
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Baixando arquivos em: {target_dir}")

    manifest = MAST.download_products(
        products,
        download_dir=str(target_dir),
        flat=True,
        cache=True,
    )

    manifest_filename = f"download_manifest_{safe_label}_{safe_target}.csv"
    save_table_report(manifest, manifest_filename)


def process_dataset_target(target_config: dict[str, Any]) -> None:
    """
    Executa o fluxo de download para um item do DATASET_TARGETS.
    """

    label = str(target_config["label"])
    target = str(target_config["target"])
    mission = str(target_config["mission"])
    radius = str(target_config["radius"])
    limit_observations = int(target_config["limit_observations"])
    limit_products = int(target_config["limit_products"])

    safe_label = sanitize_name(label)
    safe_target = sanitize_name(target)

    print("-" * 80)
    print(f"Classe: {label}")
    print(f"Alvo: {target}")

    observations = query_target_observations(
        target=target,
        mission=mission,
        radius=radius,
        limit_observations=limit_observations,
    )

    if observations is None:
        return

    observations_filename = f"observations_{safe_label}_{safe_target}.csv"
    save_table_report(observations, observations_filename)

    products = get_target_products(
        observations=observations,
        limit_products=limit_products,
    )

    if products is None:
        return

    products_filename = f"products_{safe_label}_{safe_target}.csv"
    save_table_report(products, products_filename)

    download_target_products(
        products=products,
        label=label,
        target=target,
    )


def main() -> None:
    """
    Baixa dados para múltiplas classes astronômicas.
    """

    ensure_directories()

    for target_config in DATASET_TARGETS:
        try:
            process_dataset_target(target_config)

        except Exception as error:
            print(f"Erro ao processar alvo {target_config}: {error}")

    print("-" * 80)
    print("Download do dataset finalizado.")


if __name__ == "__main__":
    main()