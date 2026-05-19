from pathlib import Path


# Raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent


# Pastas principais
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_IMAGES_DIR = PROCESSED_DATA_DIR / "images"
PROCESSED_ARRAYS_DIR = PROCESSED_DATA_DIR / "arrays"

LABELS_DIR = DATA_DIR / "labels"
LABELS_FILE = LABELS_DIR / "labels.csv"

MODELS_DIR = BASE_DIR / "models"

OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUT_IMAGES_DIR = OUTPUTS_DIR / "images"
OUTPUT_REPORTS_DIR = OUTPUTS_DIR / "reports"


# Configurações iniciais de busca no MAST
DEFAULT_MISSION = "HST"  # HST ou JWST
DEFAULT_TARGET = "M51"  # Galáxia do Redemoinho
DEFAULT_RADIUS = "0.05 deg"

DEFAULT_LIMIT_OBSERVATIONS = 5
DEFAULT_LIMIT_PRODUCTS = 10


# Configurações de pré-processamento
DEFAULT_IMAGE_SIZE = 224

PREPROCESS_PERCENTILE_LOW = 1
PREPROCESS_PERCENTILE_HIGH = 99


# Classes iniciais previstas no projeto
ASTRONOMY_CLASSES = [
    "galaxia",
    "estrela",
    "nebulosa",
    "aglomerado",
    "ruido",
    "objeto_brilhante",
    "anomalia",
]


# Configurações de treino
RANDOM_SEED = 42

TRAIN_BATCH_SIZE = 4
TRAIN_EPOCHS = 5
LEARNING_RATE = 0.001

MODEL_FILE = MODELS_DIR / "astronomy_classifier_v1.pth"


def ensure_directories() -> None:
    """
    Garante que as pastas principais existam.
    """

    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        PROCESSED_IMAGES_DIR,
        PROCESSED_ARRAYS_DIR,
        LABELS_DIR,
        MODELS_DIR,
        OUTPUT_IMAGES_DIR,
        OUTPUT_REPORTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)