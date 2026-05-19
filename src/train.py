from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Subset

from src.config import (
    MODEL_FILE,
    RANDOM_SEED,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    LEARNING_RATE,
    DEFAULT_IMAGE_SIZE,
    ensure_directories,
)
from src.dataset import AstronomyImageDataset


class SimpleAstronomyCNN(nn.Module):
    """
    Primeira CNN simples para classificar imagens astronômicas.

    Esse modelo é pequeno de propósito, apenas para validar o pipeline inicial.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Bloco 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Bloco 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Bloco 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes),
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


def create_data_loaders(
    dataset: AstronomyImageDataset,
) -> tuple[DataLoader, DataLoader | None]:
    """
    Divide o dataset em treino e validação.
    """

    total_size = len(dataset)

    if total_size == 0:
        raise ValueError("O dataset está vazio.")

    if total_size == 1:
        train_loader = DataLoader(
            dataset,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
        )

        return train_loader, None

    validation_size = max(1, int(total_size * 0.2))
    train_size = total_size - validation_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=generator,
    )

    train_subset = train_dataset
    validation_subset = validation_dataset

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

    return train_loader, validation_loader


def calculate_accuracy(outputs: Tensor, labels: Tensor) -> float:
    """
    Calcula a acurácia do batch.
    """

    predictions = torch.argmax(outputs, dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)

    return correct / total


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
    Executa o treino inicial do modelo.
    """

    ensure_directories()

    torch.manual_seed(RANDOM_SEED)

    dataset = AstronomyImageDataset()

    num_classes = len(dataset.classes)

    if num_classes == 0:
        raise ValueError("Nenhuma classe encontrada no labels.csv.")

    if num_classes == 1:
        print("Atenção: o dataset possui apenas uma classe.")
        print("Este treino servirá apenas para validar o pipeline inicial.")

    print(f"Total de imagens: {len(dataset)}")
    print(f"Classes: {dataset.classes}")

    train_loader, validation_loader = create_data_loaders(dataset)

    device = get_device()

    print(f"Dispositivo usado: {device}")

    model = SimpleAstronomyCNN(num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    for epoch in range(TRAIN_EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

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

        print(message)

    save_model(
        model=model,
        model_path=MODEL_FILE,
        dataset=dataset,
        num_classes=num_classes,
    )

    print("Treino finalizado.")


if __name__ == "__main__":
    main()