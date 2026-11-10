from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

PROCESSED_DIR = DATA_DIR / "processed"

RESULTS_DIR = (
    ROOT_DIR
    / "src"
    / "nexusai"
    / "evaluation"
    / "results"
)

APP_NAME = "NexusAI"

APP_DESCRIPTION = (
    "Agentic Retrieval Augmented Generation Platform"
)