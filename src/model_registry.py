# src/model_registry.py

from torch import Tensor, nn


class AstroMindCNNV1(nn.Module):
    """
    Primeira versão da CNN própria do projeto AstroMind.

    Arquitetura criada do zero, sem uso de modelos pré-treinados.
    Serve como baseline para comparação com versões futuras.
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
        features = self.features(image)
        output = self.classifier(features)

        return output


MODEL_REGISTRY = {
    "AstroMindCNNV1": AstroMindCNNV1,
}


def build_model(
    num_classes: int,
    model_version: str,
) -> nn.Module:
    """
    Cria o modelo com base no nome da versão.

    Exemplo:
    model_version = "AstroMindCNNV1"
    """

    if model_version not in MODEL_REGISTRY:
        available_models = ", ".join(MODEL_REGISTRY.keys())

        raise ValueError(
            f"Modelo não encontrado: {model_version}. "
            f"Modelos disponíveis: {available_models}"
        )

    model_class = MODEL_REGISTRY[model_version]

    return model_class(num_classes=num_classes)


def get_available_models() -> list[str]:
    """
    Retorna as versões de modelos disponíveis.
    """

    return list(MODEL_REGISTRY.keys())