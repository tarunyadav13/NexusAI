import json
from pathlib import Path

import mlflow

ROOT_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    ROOT_DIR / "src" / "nexusai" / "evaluation" / "results"
)

# Use SQLite instead of the deprecated filesystem backend.
DATABASE_PATH = ROOT_DIR / "mlflow.db"

TRACKING_URI = f"sqlite:///{DATABASE_PATH.as_posix()}"

mlflow.set_tracking_uri(TRACKING_URI)

mlflow.set_experiment("NexusAI-RAG-Evaluation")


def log_json_results(filename, run_name):

    path = RESULTS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation result not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Support both nested and older flat result formats.
    summary = data.get("summary", data)

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param(
            "model",
            data.get("model", "not_recorded")
        )

        mlflow.log_param(
            "total_queries",
            data.get(
                "total_queries",
                summary.get("total_queries", 0)
            )
        )

        for key, value in summary.items():

            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

        mlflow.log_artifact(str(path))

        print(f"Successfully logged: {run_name}")


def main():

    print("MLflow Tracking URI:", mlflow.get_tracking_uri())

    log_json_results(
        "retrieval_results.json",
        "Retrieval Evaluation"
    )

    log_json_results(
        "answer_results.json",
        "Answer Quality Evaluation"
    )

    log_json_results(
        "benchmark_results.json",
        "Performance Benchmark"
    )

    print("\nAll experiments logged successfully.")


if __name__ == "__main__":
    main()