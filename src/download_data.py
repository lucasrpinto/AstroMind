from typing import Any, cast

from astroquery.mast import Observations

from src.config import (
    RAW_DATA_DIR,
    OUTPUT_REPORTS_DIR,
    DEFAULT_MISSION,
    DEFAULT_TARGET,
    DEFAULT_RADIUS,
    DEFAULT_LIMIT_OBSERVATIONS,
    DEFAULT_LIMIT_PRODUCTS,
    ensure_directories,
)


# O astroquery usa métodos que o Pylance nem sempre reconhece corretamente.
# Por isso fiz esse cast para evitar erro falso de análise estática.
MAST: Any = cast(Any, Observations)


def query_observations(
    target: str = DEFAULT_TARGET,
    mission: str = DEFAULT_MISSION,
    radius: str = DEFAULT_RADIUS,
    limit_observations: int = DEFAULT_LIMIT_OBSERVATIONS,
):
    """
    Busca observações públicas no MAST para um alvo astronômico.
    """

    print(f"Buscando observações para: {target}")
    print(f"Missão escolhida: {mission}")

    observations = MAST.query_object(
        target,
        radius=radius,
    )

    if len(observations) == 0:
        raise ValueError("Nenhuma observação encontrada para esse alvo.")

    # Filtra pela missão escolhida: HST ou JWST
    if "obs_collection" in observations.colnames:
        observations = observations[
            observations["obs_collection"] == mission.upper()
        ]

    # Filtra somente produtos de imagem
    if "dataproduct_type" in observations.colnames:
        observations = observations[
            observations["dataproduct_type"] == "image"
        ]

    # Filtra somente dados científicos
    if "intentType" in observations.colnames:
        observations = observations[
            observations["intentType"] == "science"
        ]

    if len(observations) == 0:
        raise ValueError("Nenhuma observação compatível encontrada após os filtros.")

    observations = observations[:limit_observations]

    print(f"Observações encontradas após filtro: {len(observations)}")

    return observations


def get_fits_products(
    observations,
    limit_products: int = DEFAULT_LIMIT_PRODUCTS,
):
    """
    Busca produtos FITS relacionados às observações encontradas.
    """

    print("Buscando produtos das observações...")

    products = MAST.get_product_list(observations)

    if len(products) == 0:
        raise ValueError("Nenhum produto encontrado para essas observações.")

    # Filtra somente arquivos científicos em formato FITS
    filtered_products = MAST.filter_products(
        products,
        productType="SCIENCE",
        extension="fits",
    )

    if len(filtered_products) == 0:
        raise ValueError("Nenhum arquivo FITS científico encontrado.")

    filtered_products = filtered_products[:limit_products]

    print(f"Produtos FITS selecionados: {len(filtered_products)}")

    return filtered_products


def save_report(table, filename: str) -> None:
    """
    Salva uma tabela do Astropy em CSV para conferência.
    """

    output_path = OUTPUT_REPORTS_DIR / filename

    table.write(
        output_path,
        format="csv",
        overwrite=True,
    )

    print(f"Relatório salvo em: {output_path}")


def download_products(products) -> None:
    """
    Baixa os arquivos selecionados para a pasta data/raw.
    """

    print("Iniciando download dos arquivos FITS...")

    manifest = MAST.download_products(
        products,
        download_dir=str(RAW_DATA_DIR),
        flat=True,
        cache=True,
    )

    save_report(manifest, "download_manifest.csv")

    print("Download finalizado.")


def main() -> None:
    """
    Fluxo principal do script.
    """

    ensure_directories()

    observations = query_observations()
    save_report(observations, "observations_found.csv")

    products = get_fits_products(observations)
    save_report(products, "products_selected.csv")

    download_products(products)


if __name__ == "__main__":
    main()