from collections import Counter

import pandas as pd

from src.config import LABELS_FILE, OUTPUT_REPORTS_DIR


def analyze_labels() -> None:
    """
    Analisa o arquivo labels.csv e mostra a quantidade de imagens por classe.
    """

    if not LABELS_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {LABELS_FILE}")

    labels_df = pd.read_csv(LABELS_FILE)

    if labels_df.empty:
        print("O labels.csv está vazio.")
        return

    if "label" not in labels_df.columns:
        raise ValueError("O labels.csv precisa conter a coluna 'label'.")

    class_counts = Counter(labels_df["label"])

    print("-" * 80)
    print("Resumo do dataset aprovado")
    print(f"Total de imagens aprovadas: {len(labels_df)}")
    print(f"Total de classes: {len(class_counts)}")
    print("-" * 80)

    for label, count in sorted(class_counts.items()):
        print(f"{label}: {count} imagem(ns)")


def analyze_preprocess_report() -> None:
    """
    Analisa o preprocess_report.csv e mostra aprovadas, rejeitadas e motivos.
    """

    report_path = OUTPUT_REPORTS_DIR / "preprocess_report.csv"

    if not report_path.exists():
        print(f"Relatório não encontrado: {report_path}")
        return

    report_df = pd.read_csv(report_path)

    if report_df.empty:
        print("O preprocess_report.csv está vazio.")
        return

    print("-" * 80)
    print("Resumo do pré-processamento")

    status_counts = report_df["status"].value_counts()

    for status, count in status_counts.items():
        print(f"{status}: {count}")

    rejected_df = report_df[report_df["status"] == "rejected"]

    if not rejected_df.empty and "rejection_reason" in rejected_df.columns:
        print("-" * 80)
        print("Motivos de rejeição")

        reason_counts = rejected_df["rejection_reason"].value_counts()

        for reason, count in reason_counts.items():
            print(f"{reason}: {count}")


def main() -> None:
    """
    Executa a análise geral do dataset.
    """

    analyze_labels()
    analyze_preprocess_report()


if __name__ == "__main__":
    main()