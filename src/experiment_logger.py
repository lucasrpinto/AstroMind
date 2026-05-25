# src/experiment_logger.py

import csv
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    """
    Retorna a data/hora atual em formato simples para registro.
    """

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_run_id(prefix: str) -> str:
    """
    Cria um identificador único para cada execução.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]

    return f"{prefix}_{timestamp}_{short_uuid}"


def append_csv_row(
    file_path: Path,
    fieldnames: list[str],
    row: dict[str, Any],
) -> None:
    """
    Adiciona uma linha em um CSV acumulativo.

    Se o arquivo ainda não existir, cria o arquivo e escreve o cabeçalho.
    """

    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = file_path.exists()

    with file_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def format_float(value: float) -> str:
    """
    Padroniza números decimais nos logs.
    """

    return f"{value:.6f}"


def compact_dict(data: dict[str, Any]) -> str:
    """
    Converte um dicionário em texto compacto para armazenar no CSV.
    """

    return "; ".join(
        f"{key}={value}"
        for key, value in data.items()
    )