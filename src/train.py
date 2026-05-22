import csv
from collections import Counter
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset, random_split

from src.config import (
    MODEL_FILE,
    RANDOM_SEED,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEFAULT_IMAGE_SIZE,
    TRAINING_REPORT_FILE,
    ensure_directories,
)
from src.dataset import (
    AstronomyImageDataset,
    get_eval_transforms,
    get_train_transforms,
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
    DataLoader | None,
]:
    """
    Cria datasets e dataloaders para treino e validação.

    O dataset de treino usa augmentation.
    O dataset de validação não usa augmentation.
    """

    train_dataset_full = AstronomyImageDataset(
        transform=get_train_transforms()
    )

    eval_dataset_full = AstronomyImageDataset(
        transform=get_eval_transforms()
    )

    total_size = len(train_dataset_full)

    if total_size == 0:
        raise ValueError("O dataset está vazio.")

    if total_size == 1:
        train_loader = DataLoader(
            train_dataset_full,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
        )

        return train_dataset_full, eval_dataset_full, train_loader, None

    validation_size = max(1, int(total_size * 0.2))
    train_size = total_size - validation_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_subset_raw, validation_subset_raw = random_split(
        range(total_size),
        [train_size, validation_size],
        generator=generator,
    )

    train_indices = list(train_subset_raw)
    validation_indices = list(validation_subset_raw)

    train_subset = Subset(train_dataset_full, train_indices)
    validation_subset = Subset(eval_dataset_full, validation_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_subset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
    )

    return train_dataset_full, eval_dataset_full, train_loader, validation_loader


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

    Mesmo o dataset estando quase balanceado, a classe nebulosa tem menos imagens.
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


def save_training_report(rows: list[dict[str, float | int]]) -> None:
    """
    Salva o histórico do treino em CSV.
    """

    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
    ]

    with TRAINING_REPORT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Relatório de treino salvo em: {TRAINING_REPORT_FILE}")


def save_model(
    model: nn.Module,
    model_path: Path,
    dataset: AstronomyImageDataset,
    num_classes: int,
) -> None:
    """
    Salva o modelo treinado junto com as informações das classes.
    """

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "classes": dataset.classes,
        "class_to_idx": dataset.class_to_idx,
        "idx_to_class": dataset.idx_to_class,
        "num_classes": num_classes,
        "image_size": DEFAULT_IMAGE_SIZE,
    }

    torch.save(checkpoint, model_path)

    print(f"Modelo salvo em: {model_path}")


def main() -> None:
    """
    Executa o treino do modelo.
    """

    ensure_directories()

    torch.manual_seed(RANDOM_SEED)

    train_dataset, eval_dataset, train_loader, validation_loader = create_data_loaders()

    num_classes = len(train_dataset.classes)

    if num_classes <= 1:
        print("Atenção: o dataset possui apenas uma classe.")
        print("Este treino servirá apenas para validar o pipeline inicial.")

    print(f"Total de imagens: {len(train_dataset)}")
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

    for epoch in range(TRAIN_EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_loss = 0.0
        validation_accuracy = 0.0

        message = (
            f"Época {epoch + 1}/{TRAIN_EPOCHS} | "
            f"Treino Loss: {train_loss:.4f} | "
            f"Treino Acc: {train_accuracy:.4f}"
        )

        if validation_loader is not None:
            validation_loss, validation_accuracy = evaluate(
                model=model,
                validation_loader=validation_loader,
                criterion=criterion,
                device=device,
            )

            message += (
                f" | Val Loss: {validation_loss:.4f} | "
                f"Val Acc: {validation_accuracy:.4f}"
            )

        training_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )

        print(message)

    save_training_report(training_history)

    save_model(
        model=model,
        model_path=MODEL_FILE,
        dataset=eval_dataset,
        num_classes=num_classes,
    )

    print("Treino finalizado.")


if __name__ == "__main__":
    main()