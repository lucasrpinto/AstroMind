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


def read_existing_csv_rows(file_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """
    Lê um CSV existente e retorna o cabeçalho e as linhas.

    Isso permite evoluir o schema do CSV quando novas colunas forem adicionadas.
    """

    if not file_path.exists():
        return [], []

    with file_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        existing_fieldnames = list(reader.fieldnames or [])
        existing_rows = list(reader)

    return existing_fieldnames, existing_rows


def append_csv_row(
    file_path: Path,
    fieldnames: list[str],
    row: dict[str, Any],
) -> None:
    """
    Adiciona uma linha em um CSV acumulativo.

    Se o arquivo ainda não existir, cria o arquivo e escreve o cabeçalho.
    Se novas colunas forem adicionadas no futuro, o arquivo é reescrito
    preservando os registros antigos e incluindo o novo cabeçalho.
    """

    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Garante que todas as chaves da linha existam no cabeçalho.
    new_fieldnames = list(fieldnames)

    for key in row.keys():
        if key not in new_fieldnames:
            new_fieldnames.append(key)

    existing_fieldnames, existing_rows = read_existing_csv_rows(file_path)

    # Une cabeçalho antigo + cabeçalho novo, preservando a ordem.
    final_fieldnames: list[str] = []

    for fieldname in existing_fieldnames + new_fieldnames:
        if fieldname not in final_fieldnames:
            final_fieldnames.append(fieldname)

    normalized_row = {
        fieldname: row.get(fieldname, "")
        for fieldname in final_fieldnames
    }

    normalized_existing_rows = []

    for existing_row in existing_rows:
        normalized_existing_rows.append(
            {
                fieldname: existing_row.get(fieldname, "")
                for fieldname in final_fieldnames
            }
        )

    all_rows = normalized_existing_rows + [normalized_row]

    with file_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=final_fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(all_rows)


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