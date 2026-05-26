from pathlib import Path

import pandas as pd

from src.config import (
    TRAIN_RUNS_LOG_FILE,
    EVALUATE_RUNS_LOG_FILE,
    PREDICT_RUNS_LOG_FILE,
    EXPERIMENT_HISTORY_MARKDOWN_FILE,
    ensure_directories,
)

try:
    from src.config import EXTERNAL_EVALUATE_RUNS_LOG_FILE
except ImportError:
    EXTERNAL_EVALUATE_RUNS_LOG_FILE = None


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    """
    Lê um CSV se ele existir.

    Caso o CSV tenha linhas antigas com quantidade diferente de colunas,
    faz uma leitura mais tolerante para preservar o histórico.
    """

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
        )

    except Exception as error:
        print(f"Aviso: leitura padrão falhou para {path}")
        print(f"Motivo: {error}")
        print("Tentando leitura tolerante...")

        import csv

        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            rows = list(reader)

        if not rows:
            return pd.DataFrame()

        header = rows[0]
        data_rows = rows[1:]

        max_columns = max(len(row) for row in rows)

        normalized_header = list(header)

        for index in range(len(normalized_header), max_columns):
            normalized_header.append(f"extra_column_{index + 1}")

        normalized_rows = []

        for row in data_rows:
            normalized_row = list(row)

            while len(normalized_row) < max_columns:
                normalized_row.append("")

            if len(normalized_row) > max_columns:
                normalized_row = normalized_row[:max_columns]

            normalized_rows.append(normalized_row)

        return pd.DataFrame(
            normalized_rows,
            columns=normalized_header,
        )

def format_percent(value) -> str:
    """
    Formata valores de acurácia em percentual.
    """

    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return ""


def add_section(lines: list[str], title: str) -> None:
    """
    Adiciona um título de seção ao Markdown.
    """

    lines.append("")
    lines.append(f"## {title}")
    lines.append("")


def summarize_train_runs(lines: list[str], train_df: pd.DataFrame) -> None:
    """
    Resume as rodadas de treino.
    """

    add_section(lines, "1. Histórico de treinos")

    if train_df.empty:
        lines.append("Nenhum treino registrado até o momento.")
        return

    lines.append(f"Total de treinos registrados: **{len(train_df)}**")
    lines.append("")

    columns = [
        "train_run_id",
        "created_at",
        "train_size",
        "validation_size",
        "epochs",
        "best_epoch",
        "best_validation_loss",
        "best_validation_accuracy",
        "final_train_accuracy",
        "final_validation_accuracy",
    ]

    available_columns = [
        column
        for column in columns
        if column in train_df.columns
    ]

    lines.append("| Run ID | Data | Modelo | Treino | Validação | Épocas | Melhor época | Best Val Loss | Best Val Acc | Final Train Acc | Final Val Acc |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for _, row in train_df.iterrows():
        lines.append(
            "| "
            f"{row.get('train_run_id', '')} | "
            f"{row.get('created_at', '')} | "
            f"{row.get('model_version', '')} | "
            f"{row.get('train_size', '')} | "
            f"{row.get('validation_size', '')} | "
            f"{row.get('epochs', '')} | "
            f"{row.get('best_epoch', '')} | "
            f"{row.get('best_validation_loss', '')} | "
            f"{format_percent(row.get('best_validation_accuracy', ''))} | "
            f"{format_percent(row.get('final_train_accuracy', ''))} | "
            f"{format_percent(row.get('final_validation_accuracy', ''))} |"
        )

    if "best_validation_accuracy" in train_df.columns:
        best_row = train_df.sort_values(
            by="best_validation_accuracy",
            ascending=False,
        ).iloc[0]

        lines.append("")
        lines.append("**Melhor treino registrado:**")
        lines.append("")
        lines.append(f"- Run ID: `{best_row.get('train_run_id', '')}`")
        lines.append(f"- Data: {best_row.get('created_at', '')}")
        lines.append(f"- Modelo: {best_row.get('model_version', '')}")
        lines.append(f"- Melhor época: {best_row.get('best_epoch', '')}")
        lines.append(f"- Melhor validation accuracy: {format_percent(best_row.get('best_validation_accuracy', ''))}")
        lines.append(f"- Melhor validation loss: {best_row.get('best_validation_loss', '')}")


def summarize_evaluate_runs(lines: list[str], evaluate_df: pd.DataFrame) -> None:
    """
    Resume as avaliações em conjunto de teste.
    """

    add_section(lines, "2. Histórico de avaliações no conjunto de teste")

    if evaluate_df.empty:
        lines.append("Nenhuma avaliação registrada até o momento.")
        return

    lines.append(f"Total de avaliações registradas: **{len(evaluate_df)}**")
    lines.append("")

    lines.append("| Evaluate Run ID | Train Run ID | Data | Dataset | Acurácia | Acertos | Erros |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    for _, row in evaluate_df.iterrows():
        lines.append(
            "| "
            f"{row.get('evaluate_run_id', '')} | "
            f"{row.get('train_run_id', '')} | "
            f"{row.get('created_at', '')} | "
            f"{row.get('dataset_size', '')} | "
            f"{format_percent(row.get('accuracy', ''))} | "
            f"{row.get('correct', '')} | "
            f"{row.get('errors', '')} |"
        )

    if "accuracy" in evaluate_df.columns:
        best_row = evaluate_df.sort_values(
            by="accuracy",
            ascending=False,
        ).iloc[0]

        lines.append("")
        lines.append("**Melhor avaliação registrada:**")
        lines.append("")
        lines.append(f"- Evaluate Run ID: `{best_row.get('evaluate_run_id', '')}`")
        lines.append(f"- Train Run ID: `{best_row.get('train_run_id', '')}`")
        lines.append(f"- Acurácia: {format_percent(best_row.get('accuracy', ''))}")
        lines.append(f"- Acertos: {best_row.get('correct', '')}")
        lines.append(f"- Erros: {best_row.get('errors', '')}")


def summarize_external_runs(lines: list[str], external_df: pd.DataFrame) -> None:
    """
    Resume as avaliações externas.
    """

    add_section(lines, "3. Histórico de avaliações externas")

    if external_df.empty:
        lines.append("Nenhuma avaliação externa registrada até o momento.")
        return

    lines.append(f"Total de avaliações externas registradas: **{len(external_df)}**")
    lines.append("")

    lines.append("| External Run ID | Train Run ID | Data | Imagens | Imagens rotuladas | Acurácia externa | Acertos | Erros |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

    for _, row in external_df.iterrows():
        lines.append(
            "| "
            f"{row.get('external_evaluate_run_id', '')} | "
            f"{row.get('train_run_id', '')} | "
            f"{row.get('created_at', '')} | "
            f"{row.get('total_images', '')} | "
            f"{row.get('labeled_images', '')} | "
            f"{format_percent(row.get('accuracy', ''))} | "
            f"{row.get('correct', '')} | "
            f"{row.get('errors', '')} |"
        )


def summarize_predict_runs(lines: list[str], predict_df: pd.DataFrame) -> None:
    """
    Resume as predições individuais.
    """

    add_section(lines, "4. Histórico de predições individuais")

    if predict_df.empty:
        lines.append("Nenhuma predição registrada até o momento.")
        return

    lines.append(f"Total de predições registradas: **{len(predict_df)}**")
    lines.append("")

    last_predictions = predict_df.tail(10)

    lines.append("Últimas 10 predições:")
    lines.append("")
    lines.append("| Predict Run ID | Data | Classe real | Previsto | Confiança | Acertou |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for _, row in last_predictions.iterrows():
        lines.append(
            "| "
            f"{row.get('predict_run_id', '')} | "
            f"{row.get('created_at', '')} | "
            f"{row.get('true_class', '')} | "
            f"{row.get('predicted_class', '')} | "
            f"{format_percent(row.get('confidence', ''))} | "
            f"{row.get('is_correct', '')} |"
        )


def generate_markdown_report() -> None:
    """
    Gera um relatório Markdown consolidado dos experimentos.
    """

    ensure_directories()

    train_df = read_csv_if_exists(TRAIN_RUNS_LOG_FILE)
    evaluate_df = read_csv_if_exists(EVALUATE_RUNS_LOG_FILE)
    predict_df = read_csv_if_exists(PREDICT_RUNS_LOG_FILE)

    external_df = pd.DataFrame()

    if EXTERNAL_EVALUATE_RUNS_LOG_FILE is not None:
        external_df = read_csv_if_exists(EXTERNAL_EVALUATE_RUNS_LOG_FILE)

    lines: list[str] = []

    lines.append("# Histórico de Experimentos — AstroMind")
    lines.append("")
    lines.append("Este documento é gerado automaticamente a partir dos logs acumulativos do projeto.")
    lines.append("")
    lines.append("Arquivos de origem:")
    lines.append("")
    lines.append(f"- `{TRAIN_RUNS_LOG_FILE}`")
    lines.append(f"- `{EVALUATE_RUNS_LOG_FILE}`")
    lines.append(f"- `{PREDICT_RUNS_LOG_FILE}`")

    if EXTERNAL_EVALUATE_RUNS_LOG_FILE is not None:
        lines.append(f"- `{EXTERNAL_EVALUATE_RUNS_LOG_FILE}`")

    summarize_train_runs(lines, train_df)
    summarize_evaluate_runs(lines, evaluate_df)
    summarize_external_runs(lines, external_df)
    summarize_predict_runs(lines, predict_df)

    lines.append("")
    lines.append("## 5. Observações")
    lines.append("")
    lines.append("- As métricas de validação acompanham o desempenho durante o treino.")
    lines.append("- As métricas de teste medem o desempenho em imagens separadas do treino.")
    lines.append("- As avaliações externas medem a capacidade de generalização em imagens fora do dataset principal.")
    lines.append("- Resultados com poucos exemplos devem ser interpretados com cautela.")

    EXPERIMENT_HISTORY_MARKDOWN_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Histórico de experimentos salvo em: {EXPERIMENT_HISTORY_MARKDOWN_FILE}")


def main() -> None:
    """
    Executa a geração do relatório.
    """

    generate_markdown_report()


if __name__ == "__main__":
    main()