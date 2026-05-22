from pathlib import Path


# Raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent


# Pastas principais
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_IMAGES_DIR = PROCESSED_DATA_DIR / "images"
PROCESSED_ARRAYS_DIR = PROCESSED_DATA_DIR / "arrays"
REJECTED_IMAGES_DIR = PROCESSED_DATA_DIR / "rejected"

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

# Configurações de validação automática das imagens
MIN_IMAGE_STD = 8.0

MAX_BLACK_PIXEL_RATIO = 0.95
MAX_WHITE_PIXEL_RATIO = 0.60
MAX_EXTREME_PIXEL_RATIO = 0.85

BLACK_PIXEL_THRESHOLD = 5
WHITE_PIXEL_THRESHOLD = 250

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

# Alvos iniciais para montar um dataset com mais de uma classe
# Cada item representa um objeto astronômico que será buscado no MAST
DATASET_TARGETS = [
    # Galáxias
    {
        "label": "galaxia",
        "target": "M51",
        "mission": "HST",
        "radius": "0.05 deg",
        "limit_observations": 5,
        "limit_products": 8,
    },
    {
        "label": "galaxia",
        "target": "M101",
        "mission": "HST",
        "radius": "0.05 deg",
        "limit_observations": 5,
        "limit_products": 8,
    },
    {
        "label": "galaxia",
        "target": "M104",
        "mission": "HST",
        "radius": "0.05 deg",
        "limit_observations": 5,
        "limit_products": 8,
    },

    # Nebulosas
    {
        "label": "nebulosa",
        "target": "M42",
        "mission": "HST",
        "radius": "0.05 deg",
        "limit_observations": 5,
        "limit_products": 8,
    },
    {
        "label": "nebulosa",
        "target": "M8",
        "mission": "HST",
        "radius": "0.05 deg",
        "limit_observations": 5,
        "limit_products": 8,
    },
    {
        "label": "nebulosa",
        "target": "M16",
        "mission": "HST",
        "radius": "0.05 deg",
        "limit_observations": 5,
        "limit_products": 8,
    },
    {
        "label": "nebulosa",
        "target": "NGC 6302",
        "mission": "HST",
        "radius": "0.05 deg",
        "limit_observations": 5,
        "limit_products": 8,
    },

    # Aglomerados
    {
        "label": "aglomerado",
        "target": "M13",
        "mission": "HST",
        "radius": "0.05 deg",
        "limit_observations": 5,
        "limit_products": 8,
    },
    {
        "label": "aglomerado",
        "target": "M92",
        "mission": "HST",
        "radius": "0.05 deg",
        "limit_observations": 5,
        "limit_products": 8,
    },
    {
        "label": "aglomerado",
        "target": "NGC 104",
        "mission": "HST",
        "radius": "0.05 deg",
        "limit_observations": 5,
        "limit_products": 8,
    },

    # Estrelas
    {
        "label": "estrela",
        "target": "Vega",
        "mission": "HST",
        "radius": "0.05 deg",
        "limit_observations": 5,
        "limit_products": 8,
    },
    {
        "label": "estrela",
        "target": "Sirius",
        "mission": "HST",
        "radius": "0.05 deg",
        "limit_observations": 5,
        "limit_products": 8,
    },
    {
        "label": "estrela",
        "target": "Betelgeuse",
        "mission": "HST",
        "radius": "0.05 deg",
        "limit_observations": 5,
        "limit_products": 8,
    },
    {
        "label": "estrela",
        "target": "Proxima Cen",
        "mission": "HST",
        "radius": "0.05 deg",
        "limit_observations": 5,
        "limit_products": 8,
    },
]

# Configurações de treino
RANDOM_SEED = 42

TRAIN_BATCH_SIZE = 4
TRAIN_EPOCHS = 30
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0001

MODEL_FILE = MODELS_DIR / "astronomy_classifier_v1.pth"
TRAINING_REPORT_FILE = OUTPUT_REPORTS_DIR / "training_report.csv"

EVALUATION_REPORT_FILE = OUTPUT_REPORTS_DIR / "evaluation_report.csv"
CONFUSION_MATRIX_FILE = OUTPUT_REPORTS_DIR / "confusion_matrix.csv"

# Padrões de arquivos FITS que serão rejeitados automaticamente
# Esses arquivos costumam representar produtos menos adequados para treino visual
# src/config.py

REJECT_FILENAME_PATTERNS = [
    "_d0f",
    "_d0m",
    "_c0f",
    "_c1f",
    "_c0m",
    "_c1m",
]


# Filtro para objetos muito brilhantes fora do centro
# Útil principalmente para a classe estrela
BRIGHT_OBJECT_PERCENTILE = 99.5
MAX_BRIGHT_CENTROID_DISTANCE = 0.45

def ensure_directories() -> None:
    """
    Garante que as pastas principais existam.
    """

    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        PROCESSED_IMAGES_DIR,
        PROCESSED_ARRAYS_DIR,
        REJECTED_IMAGES_DIR,
        LABELS_DIR,
        MODELS_DIR,
        OUTPUT_IMAGES_DIR,
        OUTPUT_REPORTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)