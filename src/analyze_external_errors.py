import pandas as pd

from src.config import (
    EXTERNAL_EVALUATION_REPORT_FILE,
    EXTERNAL_ERRORS_REPORT_FILE,
)


def load_external_evaluation() -> pd.DataFrame:
    """
    Carrega o relatório de avaliação externa.
    """

    if not EXTERNAL_EVALUATION_REPORT_FILE.exists():
        raise FileNotFoundError(
            f"Relatório externo não encontrado: {EXTERNAL_EVALUATION_REPORT_FILE}"
        )

    evaluation_df = pd.read_csv(EXTERNAL_EVALUATION_REPORT_FILE)

    if evaluation_df.empty:
        raise ValueError("O relatório de avaliação externa está vazio.")

    required_columns = {
        "filename",
        "true_class",
        "predicted_class",
        "confidence",
        "is_correct",
        "status",
    }

    missing_columns = required_columns - set(evaluation_df.columns)

    if missing_columns:
        raise ValueError(
            f"O relatório externo não possui as colunas obrigatórias: {missing_columns}"
        )

    return evaluation_df


def get_probability_columns(evaluation_df: pd.DataFrame) -> list[str]:
    """
    Identifica as colunas de probabilidade por classe.

    Exemplo:
    prob_aglomerado
    prob_estrela
    prob_galaxia
    prob_nebulosa
    """

    return [
        column
        for column in evaluation_df.columns
        if column.startswith("prob_")
    ]


def filter_errors(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra somente registros com erro de classificação.
    """

    success_df = evaluation_df[evaluation_df["status"] == "success"].copy()

    labeled_df = success_df[
        success_df["true_class"].astype(str).str.strip() != ""
    ].copy()

    errors_df = labeled_df[
        labeled_df["is_correct"].astype(str).str.lower() == "false"
    ].copy()

    return errors_df


def save_errors_report(errors_df: pd.DataFrame) -> None:
    """
    Salva o relatório apenas com os erros externos.
    """

    errors_df.to_csv(
        EXTERNAL_ERRORS_REPORT_FILE,
        index=False,
        encoding="utf-8",
    )

    print(f"Relatório de erros externos salvo em: {EXTERNAL_ERRORS_REPORT_FILE}")


def print_general_summary(evaluation_df: pd.DataFrame, errors_df: pd.DataFrame) -> None:
    """
    Exibe resumo geral da avaliação externa.
    """

    success_df = evaluation_df[evaluation_df["status"] == "success"].copy()

    labeled_df = success_df[
        success_df["true_class"].astype(str).str.strip() != ""
    ].copy()

    total = len(labeled_df)
    errors = len(errors_df)
    correct = total - errors
    accuracy = correct / total if total > 0 else 0.0

    print("-" * 80)
    print("Resumo geral da avaliação externa")
    print(f"Total de imagens rotuladas avaliadas: {total}")
    print(f"Acertos: {correct}")
    print(f"Erros: {errors}")
    print(f"Acurácia externa: {accuracy * 100:.2f}%")


def print_error_summary_by_class(errors_df: pd.DataFrame) -> None:
    """
    Exibe quantos erros ocorreram por classe real e por classe prevista.
    """

    print("-" * 80)
    print("Erros por classe real")

    if errors_df.empty:
        print("Nenhum erro encontrado.")
        return

    true_counts = errors_df["true_class"].value_counts()

    for class_name, count in true_counts.items():
        print(f"{class_name}: {count}")

    print("-" * 80)
    print("Classes previstas nos erros")

    predicted_counts = errors_df["predicted_class"].value_counts()

    for class_name, count in predicted_counts.items():
        print(f"{class_name}: {count}")


def print_detailed_errors(
    errors_df: pd.DataFrame,
    probability_columns: list[str],
) -> None:
    """
    Exibe os erros detalhados com confiança e probabilidades.
    """

    print("-" * 80)
    print("Detalhamento dos erros externos")

    if errors_df.empty:
        print("Nenhum erro encontrado.")
        return

    for _, row in errors_df.iterrows():
        print("-" * 80)
        print(f"Arquivo: {row['filename']}")
        print(f"Classe real: {row['true_class']}")
        print(f"Classe prevista: {row['predicted_class']}")
        print(f"Confiança: {float(row['confidence']) * 100:.2f}%")

        print("Probabilidades:")

        for column in probability_columns:
            class_name = column.replace("prob_", "")
            probability = float(row[column])

            print(f"- {class_name}: {probability * 100:.2f}%")


def print_nebulosa_focus(errors_df: pd.DataFrame) -> None:
    """
    Mostra análise específica dos erros da classe nebulosa.
    """

    nebulosa_errors_df = errors_df[
        errors_df["true_class"].astype(str) == "nebulosa"
    ]

    print("-" * 80)
    print("Foco: erros da classe nebulosa")

    if nebulosa_errors_df.empty:
        print("Nenhum erro externo da classe nebulosa encontrado.")
        return

    print(f"Total de erros de nebulosa: {len(nebulosa_errors_df)}")

    predicted_counts = nebulosa_errors_df["predicted_class"].value_counts()

    print("Nebulosas foram classificadas como:")

    for predicted_class, count in predicted_counts.items():
        print(f"- {predicted_class}: {count}")

    print("Arquivos de nebulosa com erro:")

    for _, row in nebulosa_errors_df.iterrows():
        print(
            f"- {row['filename']} | "
            f"Previsto: {row['predicted_class']} | "
            f"Confiança: {float(row['confidence']) * 100:.2f}%"
        )


def main() -> None:
    """
    Executa a análise dos erros externos.
    """

    evaluation_df = load_external_evaluation()
    probability_columns = get_probability_columns(evaluation_df)

    errors_df = filter_errors(evaluation_df)

    save_errors_report(errors_df)

    print_general_summary(evaluation_df, errors_df)
    print_error_summary_by_class(errors_df)
    print_detailed_errors(errors_df, probability_columns)
    print_nebulosa_focus(errors_df)

    print("-" * 80)
    print("Análise de erros externos finalizada.")


if __name__ == "__main__":
    main()