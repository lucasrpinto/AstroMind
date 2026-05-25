import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    LABELS_FILE,
    TRAIN_LABELS_FILE,
    VAL_LABELS_FILE,
    TEST_LABELS_FILE,
    RANDOM_SEED,
    ensure_directories,
)


def save_split(df: pd.DataFrame, path) -> None:
    """
    Salva uma divisão do dataset em CSV.
    """

    df.to_csv(path, index=False, encoding="utf-8")
    print(f"Arquivo salvo: {path} | Total: {len(df)}")


def main() -> None:
    """
    Divide o labels.csv em treino, validação e teste.

    Divisão:
    - 70% treino
    - 15% validação
    - 15% teste
    """

    ensure_directories()

    if not LABELS_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {LABELS_FILE}")

    labels_df = pd.read_csv(LABELS_FILE)

    if labels_df.empty:
        raise ValueError("O labels.csv está vazio.")

    if not {"filename", "label"}.issubset(labels_df.columns):
        raise ValueError("O labels.csv precisa conter as colunas filename,label.")

    print("-" * 80)
    print("Total por classe no dataset completo:")
    print(labels_df["label"].value_counts())

    train_df, temp_df = train_test_split(
        labels_df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=labels_df["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df["label"],
    )

    print("-" * 80)
    print("Divisão final:")
    print(f"Treino: {len(train_df)}")
    print(f"Validação: {len(val_df)}")
    print(f"Teste: {len(test_df)}")

    print("-" * 80)
    print("Treino por classe:")
    print(train_df["label"].value_counts())

    print("-" * 80)
    print("Validação por classe:")
    print(val_df["label"].value_counts())

    print("-" * 80)
    print("Teste por classe:")
    print(test_df["label"].value_counts())

    save_split(train_df, TRAIN_LABELS_FILE)
    save_split(val_df, VAL_LABELS_FILE)
    save_split(test_df, TEST_LABELS_FILE)

    print("Split do dataset finalizado.")


if __name__ == "__main__":
    main()