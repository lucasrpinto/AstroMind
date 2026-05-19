from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import (
    PROCESSED_IMAGES_DIR,
    LABELS_FILE,
    DEFAULT_IMAGE_SIZE,
)


# Tipo esperado para as transformações das imagens
ImageTransform = Callable[[Image.Image], Tensor]


def get_default_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> ImageTransform:
    """
    Define as transformações padrão para o modelo.
    """

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )

    return transform


class AstronomyImageDataset(Dataset[tuple[Tensor, Tensor]]):
    """
    Dataset PyTorch para carregar imagens astronômicas processadas.

    Ele lê o arquivo labels.csv, encontra cada imagem em data/processed/images
    e retorna a imagem transformada junto com seu rótulo numérico.
    """

    def __init__(
        self,
        images_dir: Path = PROCESSED_IMAGES_DIR,
        labels_file: Path = LABELS_FILE,
        transform: ImageTransform | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_file = Path(labels_file)

        # Se nenhuma transformação for enviada, usa a transformação padrão
        self.transform = transform or get_default_transforms()

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Pasta de imagens não encontrada: {self.images_dir}")

        if not self.labels_file.exists():
            raise FileNotFoundError(f"Arquivo de labels não encontrado: {self.labels_file}")

        self.data = pd.read_csv(self.labels_file)

        required_columns = {"filename", "label"}

        if not required_columns.issubset(self.data.columns):
            raise ValueError("O labels.csv precisa conter as colunas: filename,label")

        self.classes = sorted(self.data["label"].unique().tolist())

        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(self.classes)
        }

        self.idx_to_class = {
            index: class_name
            for class_name, index in self.class_to_idx.items()
        }

    def __len__(self) -> int:
        """
        Retorna a quantidade de imagens no dataset.
        """

        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """
        Retorna uma imagem transformada e seu rótulo.
        """

        row = self.data.iloc[index]

        filename = str(row["filename"])
        label_name = str(row["label"])

        image_path = self.images_dir / filename

        if not image_path.exists():
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

        image = Image.open(image_path).convert("RGB")

        image_tensor = self.transform(image)

        label_index = self.class_to_idx[label_name]
        label_tensor = torch.tensor(label_index, dtype=torch.long)

        return image_tensor, label_tensor


def main() -> None:
    """
    Teste simples para verificar se o dataset está carregando corretamente.
    """

    dataset = AstronomyImageDataset()

    print(f"Total de imagens: {len(dataset)}")
    print(f"Classes encontradas: {dataset.classes}")
    print(f"Mapeamento das classes: {dataset.class_to_idx}")

    image, label = dataset[0]

    label_index = int(label.item())

    print(f"Formato da imagem/tensor: {image.shape}")
    print(f"Label numérico: {label_index}")
    print(f"Label original: {dataset.idx_to_class[label_index]}")


if __name__ == "__main__":
    main()