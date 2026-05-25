import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.config import (
    BEST_MODEL_FILE,
    LAST_MODEL_FILE,
    RANDOM_SEED,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEFAULT_IMAGE_SIZE,
    TRAINING_REPORT_FILE,
    EXPERIMENT_SUMMARY_FILE,
    TRAINING_LOSS_PLOT_FILE,
    TRAINING_ACCURACY_PLOT_FILE,
    TRAIN_LABELS_FILE,
    VAL_LABELS_FILE,
    TRAIN_RUNS_LOG_FILE,
    ensure_directories,
)

from src.dataset import (
    AstronomyImageDataset,
    get_eval_transforms,
    get_train_transforms,
)

from src.experiment_logger import (
    append_csv_row,
    compact_dict,
    create_run_id,
    format_float,
    now_iso,
)


class SimpleAstronomyCNN(nn.Module):
    """
    CNN simples para classificar imagens astronômicas.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, image: Tensor) -> Tensor:
        """
        Executa a predição do modelo.
        """

        features = self.features(image)
        output = self.classifier(features)

        return output


def get_device() -> torch.device:
    """
    Define se o treino será feito em GPU ou CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def create_data_loaders() -> tuple[
    AstronomyImageDataset,
    AstronomyImageDataset,
    DataLoader,
    DataLoader,
]:
    """
    Cria datasets e dataloaders usando splits fixos.

    O treino usa train_labels.csv.
    A validação usa val_labels.csv.
    """

    train_dataset = AstronomyImageDataset(
        labels_file=TRAIN_LABELS_FILE,
        transform=get_train_transforms(),
    )

    validation_dataset = AstronomyImageDataset(
        labels_file=VAL_LABELS_FILE,
        transform=get_eval_transforms(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
    )

    return train_dataset, validation_dataset, train_loader, validation_loader


def calculate_accuracy(outputs: Tensor, labels: Tensor) -> float:
    """
    Calcula a acurácia do batch.
    """

    predictions = torch.argmax(outputs, dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)

    return correct / total


def calculate_class_weights(
    dataset: AstronomyImageDataset,
    device: torch.device,
) -> Tensor:
    """
    Calcula pesos por classe para reduzir impacto de desbalanceamento.
    """

    labels = dataset.data["label"].tolist()
    counts = Counter(labels)

    weights = []

    for class_name in dataset.classes:
        count = counts[class_name]
        weight = len(labels) / count
        weights.append(weight)

    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    print(f"Pesos por classe: {dict(zip(dataset.classes, weights))}")

    return weights_tensor


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Treina o modelo por uma época.
    """

    model.train()

    total_loss = 0.0
    total_accuracy = 0.0
    total_batches = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        accuracy = calculate_accuracy(outputs, labels)

        total_loss += float(loss.item())
        total_accuracy += accuracy
        total_batches += 1

    average_loss = total_loss / total_batches
    average_accuracy = total_accuracy / total_batches

    return average_loss, average_accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    validation_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Avalia o modelo usando o conjunto de validação.
    """

    model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    total_batches = 0

    for images, labels in validation_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        accuracy = calculate_accuracy(outputs, labels)

        total_loss += float(loss.item())
        total_accuracy += accuracy
        total_batches += 1

    average_loss = total_loss / total_batches
    average_accuracy = total_accuracy / total_batches

    return average_loss, average_accuracy


def build_checkpoint(
    model: nn.Module,
    dataset: AstronomyImageDataset,
    num_classes: int,
    epoch: int,
    validation_loss: float,
    validation_accuracy: float,
    train_run_id: str,
) -> dict:
    """
    Monta o checkpoint do modelo.
    """

    return {
        "model_state_dict": model.state_dict(),
        "classes": dataset.classes,
        "class_to_idx": dataset.class_to_idx,
        "idx_to_class": dataset.idx_to_class,
        "num_classes": num_classes,
        "image_size": DEFAULT_IMAGE_SIZE,
        "epoch": epoch,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "train_run_id": train_run_id,
    }


def save_model_checkpoint(
    checkpoint: dict,
    model_path: Path,
) -> None:
    """
    Salva um checkpoint do modelo.
    """

    torch.save(checkpoint, model_path)
    print(f"Modelo salvo em: {model_path}")


def save_training_report(rows: list[dict[str, float | int]]) -> None:
    """
    Salva o histórico completo do treino em CSV.
    """

    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
        "is_best_model",
    ]

    with TRAINING_REPORT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Relatório de treino salvo em: {TRAINING_REPORT_FILE}")


def save_training_plots(rows: list[dict[str, float | int]]) -> None:
    """
    Gera gráficos de loss e accuracy.
    """

    epochs = [int(row["epoch"]) for row in rows]

    train_loss = [float(row["train_loss"]) for row in rows]
    validation_loss = [float(row["validation_loss"]) for row in rows]

    train_accuracy = [float(row["train_accuracy"]) for row in rows]
    validation_accuracy = [float(row["validation_accuracy"]) for row in rows]

    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, validation_loss, label="Validation Loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Evolução do Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(TRAINING_LOSS_PLOT_FILE, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accuracy, label="Train Accuracy")
    plt.plot(epochs, validation_accuracy, label="Validation Accuracy")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.title("Evolução da Acurácia")
    plt.legend()
    plt.grid(True)
    plt.savefig(TRAINING_ACCURACY_PLOT_FILE, bbox_inches="tight")
    plt.close()

    print(f"Gráfico de loss salvo em: {TRAINING_LOSS_PLOT_FILE}")
    print(f"Gráfico de acurácia salvo em: {TRAINING_ACCURACY_PLOT_FILE}")


def append_experiment_summary(
    train_dataset: AstronomyImageDataset,
    validation_dataset: AstronomyImageDataset,
    best_epoch: int,
    best_validation_loss: float,
    best_validation_accuracy: float,
    final_train_loss: float,
    final_train_accuracy: float,
    final_validation_loss: float,
    final_validation_accuracy: float,
) -> None:
    """
    Adiciona uma linha no resumo de experimentos.
    """

    file_exists = EXPERIMENT_SUMMARY_FILE.exists()

    fieldnames = [
        "created_at",
        "train_size",
        "validation_size",
        "num_classes",
        "classes",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "best_epoch",
        "best_validation_loss",
        "best_validation_accuracy",
        "final_train_loss",
        "final_train_accuracy",
        "final_validation_loss",
        "final_validation_accuracy",
        "best_model_file",
        "last_model_file",
    ]

    row = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_size": len(train_dataset),
        "validation_size": len(validation_dataset),
        "num_classes": len(train_dataset.classes),
        "classes": "|".join(train_dataset.classes),
        "epochs": TRAIN_EPOCHS,
        "batch_size": TRAIN_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "best_epoch": best_epoch,
        "best_validation_loss": round(best_validation_loss, 6),
        "best_validation_accuracy": round(best_validation_accuracy, 6),
        "final_train_loss": round(final_train_loss, 6),
        "final_train_accuracy": round(final_train_accuracy, 6),
        "final_validation_loss": round(final_validation_loss, 6),
        "final_validation_accuracy": round(final_validation_accuracy, 6),
        "best_model_file": str(BEST_MODEL_FILE),
        "last_model_file": str(LAST_MODEL_FILE),
    }

    with EXPERIMENT_SUMMARY_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print(f"Resumo do experimento salvo em: {EXPERIMENT_SUMMARY_FILE}")


def append_train_run_log(
    train_run_id: str,
    train_dataset: AstronomyImageDataset,
    validation_dataset: AstronomyImageDataset,
    best_epoch: int,
    best_validation_loss: float,
    best_validation_accuracy: float,
    final_train_loss: float,
    final_train_accuracy: float,
    final_validation_loss: float,
    final_validation_accuracy: float,
    class_weights: Tensor,
) -> None:
    """
    Registra uma linha acumulativa com o resultado da rodada de treino.
    """

    fieldnames = [
        "train_run_id",
        "created_at",
        "train_size",
        "validation_size",
        "num_classes",
        "classes",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "best_epoch",
        "best_validation_loss",
        "best_validation_accuracy",
        "final_train_loss",
        "final_train_accuracy",
        "final_validation_loss",
        "final_validation_accuracy",
        "class_weights",
        "best_model_file",
        "last_model_file",
        "training_report_file",
        "training_loss_plot_file",
        "training_accuracy_plot_file",
    ]

    class_weights_dict = {
        class_name: format_float(float(class_weights[index].item()))
        for index, class_name in enumerate(train_dataset.classes)
    }

    row = {
        "train_run_id": train_run_id,
        "created_at": now_iso(),
        "train_size": len(train_dataset),
        "validation_size": len(validation_dataset),
        "num_classes": len(train_dataset.classes),
        "classes": "|".join(train_dataset.classes),
        "epochs": TRAIN_EPOCHS,
        "batch_size": TRAIN_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "best_epoch": best_epoch,
        "best_validation_loss": format_float(best_validation_loss),
        "best_validation_accuracy": format_float(best_validation_accuracy),
        "final_train_loss": format_float(final_train_loss),
        "final_train_accuracy": format_float(final_train_accuracy),
        "final_validation_loss": format_float(final_validation_loss),
        "final_validation_accuracy": format_float(final_validation_accuracy),
        "class_weights": compact_dict(class_weights_dict),
        "best_model_file": str(BEST_MODEL_FILE),
        "last_model_file": str(LAST_MODEL_FILE),
        "training_report_file": str(TRAINING_REPORT_FILE),
        "training_loss_plot_file": str(TRAINING_LOSS_PLOT_FILE),
        "training_accuracy_plot_file": str(TRAINING_ACCURACY_PLOT_FILE),
    }

    append_csv_row(
        file_path=TRAIN_RUNS_LOG_FILE,
        fieldnames=fieldnames,
        row=row,
    )

    print(f"Log acumulativo do treino salvo em: {TRAIN_RUNS_LOG_FILE}")

def main() -> None:
    """
    Executa o treino do modelo.
    """

    ensure_directories()

    train_run_id = create_run_id("train")
    print(f"Train Run ID: {train_run_id}")

    torch.manual_seed(RANDOM_SEED)

    train_dataset, validation_dataset, train_loader, validation_loader = create_data_loaders()

    num_classes = len(train_dataset.classes)

    print(f"Total de imagens de treino: {len(train_dataset)}")
    print(f"Total de imagens de validação: {len(validation_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    device = get_device()

    print(f"Dispositivo usado: {device}")

    model = SimpleAstronomyCNN(num_classes=num_classes)
    model = model.to(device)

    class_weights = calculate_class_weights(
        dataset=train_dataset,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    training_history: list[dict[str, float | int]] = []

    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    best_epoch = 0

    final_train_loss = 0.0
    final_train_accuracy = 0.0
    final_validation_loss = 0.0
    final_validation_accuracy = 0.0

    for epoch in range(TRAIN_EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_loss, validation_accuracy = evaluate(
            model=model,
            validation_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        is_best_model = validation_loss < best_validation_loss

        if is_best_model:
            best_validation_loss = validation_loss
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch + 1

            best_checkpoint = build_checkpoint(
                model=model,
                dataset=train_dataset,
                num_classes=num_classes,
                epoch=best_epoch,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
                train_run_id=train_run_id,
            )

            save_model_checkpoint(
                checkpoint=best_checkpoint,
                model_path=BEST_MODEL_FILE,
            )

        final_train_loss = train_loss
        final_train_accuracy = train_accuracy
        final_validation_loss = validation_loss
        final_validation_accuracy = validation_accuracy

        training_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "is_best_model": int(is_best_model),
            }
        )

        message = (
            f"Época {epoch + 1}/{TRAIN_EPOCHS} | "
            f"Treino Loss: {train_loss:.4f} | "
            f"Treino Acc: {train_accuracy:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy:.4f}"
        )

        if is_best_model:
            message += " | BEST"

        print(message)

    last_checkpoint = build_checkpoint(
        model=model,
        dataset=train_dataset,
        num_classes=num_classes,
        epoch=TRAIN_EPOCHS,
        validation_loss=final_validation_loss,
        validation_accuracy=final_validation_accuracy,
        train_run_id=train_run_id,
    )

    save_model_checkpoint(
        checkpoint=last_checkpoint,
        model_path=LAST_MODEL_FILE,
    )

    save_training_report(training_history)
    save_training_plots(training_history)

    append_experiment_summary(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        best_validation_accuracy=best_validation_accuracy,
        final_train_loss=final_train_loss,
        final_train_accuracy=final_train_accuracy,
        final_validation_loss=final_validation_loss,
        final_validation_accuracy=final_validation_accuracy,
    )

    append_train_run_log(
        train_run_id=train_run_id,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        best_validation_accuracy=best_validation_accuracy,
        final_train_loss=final_train_loss,
        final_train_accuracy=final_train_accuracy,
        final_validation_loss=final_validation_loss,
        final_validation_accuracy=final_validation_accuracy,
        class_weights=class_weights,
    )

    print("-" * 80)
    print("Resumo do treino")
    print(f"Melhor época: {best_epoch}")
    print(f"Melhor Val Loss: {best_validation_loss:.4f}")
    print(f"Melhor Val Acc: {best_validation_accuracy:.4f}")
    print(f"Modelo BEST: {BEST_MODEL_FILE}")
    print(f"Modelo LAST: {LAST_MODEL_FILE}")
    print("Treino finalizado.")


if __name__ == "__main__":
    main()