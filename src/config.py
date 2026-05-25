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

DOCS_DIR = BASE_DIR / "docs"
EXPERIMENT_HISTORY_MARKDOWN_FILE = DOCS_DIR / "experiment_history.md"

LABELS_DIR = DATA_DIR / "labels"
LABELS_FILE = LABELS_DIR / "labels.csv"
TRAIN_LABELS_FILE = LABELS_DIR / "train_labels.csv"
VAL_LABELS_FILE = LABELS_DIR / "val_labels.csv"
TEST_LABELS_FILE = LABELS_DIR / "test_labels.csv"

MODELS_DIR = BASE_DIR / "models"

# Logs versionáveis dos experimentos
EXPERIMENT_LOGS_DIR = BASE_DIR / "experiment_logs"

TRAIN_RUNS_LOG_FILE = EXPERIMENT_LOGS_DIR / "train_runs.csv"
EVALUATE_RUNS_LOG_FILE = EXPERIMENT_LOGS_DIR / "evaluate_runs.csv"
PREDICT_RUNS_LOG_FILE = EXPERIMENT_LOGS_DIR / "predict_runs.csv"

OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUT_IMAGES_DIR = OUTPUTS_DIR / "images"
OUTPUT_REPORTS_DIR = OUTPUTS_DIR / "reports"


# Configurações iniciais de busca no MAST
DEFAULT_MISSION = "HST"  # HST ou JWST
DEFAULT_TARGET = "M51"  # Galáxia do Redemoinho
DEFAULT_RADIUS = "0.05 deg"

DEFAULT_LIMIT_OBSERVATIONS = 5
DEFAULT_LIMIT_PRODUCTS = 10

# Dados externos para teste manual do modelo
EXTERNAL_DATA_DIR = DATA_DIR / "external"
EXTERNAL_IMAGES_DIR = EXTERNAL_DATA_DIR / "images"
EXTERNAL_LABELS_FILE = EXTERNAL_DATA_DIR / "external_labels.csv"

EXTERNAL_EVALUATION_REPORT_FILE = OUTPUT_REPORTS_DIR / "external_evaluation_report.csv"
EXTERNAL_CONFUSION_MATRIX_FILE = OUTPUT_REPORTS_DIR / "external_confusion_matrix.csv"

EXTERNAL_EVALUATE_RUNS_LOG_FILE = EXPERIMENT_LOGS_DIR / "external_evaluate_runs.csv"

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

# Alvos complementares via SkyView
# A ideia é usar imagens mais padronizadas para equilibrar o dataset,
# principalmente nas classes estrela e galaxia.
SKYVIEW_TARGETS = [
    # Galáxias
    {
        "label": "galaxia",
        "target": "M51",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.18,
        "pixels": 400,
    },
    {
        "label": "galaxia",
        "target": "M101",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.22,
        "pixels": 400,
    },
    {
        "label": "galaxia",
        "target": "M104",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.18,
        "pixels": 400,
    },
    {
        "label": "galaxia",
        "target": "M81",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.25,
        "pixels": 400,
    },
    {
        "label": "galaxia",
        "target": "M82",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.20,
        "pixels": 400,
    },
    {
        "label": "galaxia",
        "target": "NGC 1300",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.18,
        "pixels": 400,
    },

    # Estrelas
    {
        "label": "estrela",
        "target": "Vega",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.08,
        "pixels": 400,
    },
    {
        "label": "estrela",
        "target": "Sirius",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.08,
        "pixels": 400,
    },
    {
        "label": "estrela",
        "target": "Betelgeuse",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.08,
        "pixels": 400,
    },
    {
        "label": "estrela",
        "target": "Rigel",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.08,
        "pixels": 400,
    },
    {
        "label": "estrela",
        "target": "Aldebaran",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.08,
        "pixels": 400,
    },
    {
        "label": "estrela",
        "target": "Polaris",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.08,
        "pixels": 400,
    },
    {
        "label": "estrela",
        "target": "Deneb",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.08,
        "pixels": 400,
    },

    # Nebulosas complementares
    {
        "label": "nebulosa",
        "target": "M42",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.25,
        "pixels": 400,
    },
    {
        "label": "nebulosa",
        "target": "M8",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.25,
        "pixels": 400,
    },
    {
        "label": "nebulosa",
        "target": "M16",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.25,
        "pixels": 400,
    },

    # Aglomerados complementares
    {
        "label": "aglomerado",
        "target": "M13",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.18,
        "pixels": 400,
    },
    {
        "label": "aglomerado",
        "target": "M92",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.18,
        "pixels": 400,
    },
    {
        "label": "aglomerado",
        "target": "NGC 104",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.18,
        "pixels": 400,
    },
]

# Alvos externos para validação fora do dataset principal
# Importante: esses alvos não devem ser os mesmos usados no treino.
EXTERNAL_SKYVIEW_TARGETS = [
    # Galáxias externas
    {
        "label": "galaxia",
        "target": "M33",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.22,
        "pixels": 400,
    },
    {
        "label": "galaxia",
        "target": "NGC 891",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.18,
        "pixels": 400,
    },
    {
        "label": "galaxia",
        "target": "NGC 4565",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.18,
        "pixels": 400,
    },

    # Estrelas externas
    {
        "label": "estrela",
        "target": "Arcturus",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.08,
        "pixels": 400,
    },
    {
        "label": "estrela",
        "target": "Capella",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.08,
        "pixels": 400,
    },
    {
        "label": "estrela",
        "target": "Altair",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.08,
        "pixels": 400,
    },

    # Nebulosas externas
    {
        "label": "nebulosa",
        "target": "M27",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.20,
        "pixels": 400,
    },
    {
        "label": "nebulosa",
        "target": "M57",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.12,
        "pixels": 400,
    },
    {
        "label": "nebulosa",
        "target": "NGC 7000",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.25,
        "pixels": 400,
    },

    # Aglomerados externos
    {
        "label": "aglomerado",
        "target": "M3",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.18,
        "pixels": 400,
    },
    {
        "label": "aglomerado",
        "target": "M5",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.18,
        "pixels": 400,
    },
    {
        "label": "aglomerado",
        "target": "M15",
        "surveys": ["DSS2 Red", "DSS2 Blue"],
        "width_degrees": 0.18,
        "pixels": 400,
    },
]

# Configurações de treino
RANDOM_SEED = 42

TRAIN_BATCH_SIZE = 4
TRAIN_EPOCHS = 30
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0001

BEST_MODEL_FILE = MODELS_DIR / "astronomy_classifier_best.pth"
LAST_MODEL_FILE = MODELS_DIR / "astronomy_classifier_last.pth"

# Mantém compatibilidade com predict.py e evaluate.py
MODEL_FILE = BEST_MODEL_FILE

TRAINING_REPORT_FILE = OUTPUT_REPORTS_DIR / "training_report.csv"
EXPERIMENT_SUMMARY_FILE = OUTPUT_REPORTS_DIR / "experiment_summary.csv"

TRAINING_LOSS_PLOT_FILE = OUTPUT_IMAGES_DIR / "training_loss.png"
TRAINING_ACCURACY_PLOT_FILE = OUTPUT_IMAGES_DIR / "training_accuracy.png"

EVALUATION_REPORT_FILE = OUTPUT_REPORTS_DIR / "evaluation_report.csv"
CONFUSION_MATRIX_FILE = OUTPUT_REPORTS_DIR / "confusion_matrix.csv"
SKYVIEW_DOWNLOAD_REPORT_FILE = OUTPUT_REPORTS_DIR / "skyview_download_report.csv"

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
        EXPERIMENT_LOGS_DIR,
        EXTERNAL_DATA_DIR,
        EXTERNAL_IMAGES_DIR,
        DOCS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)