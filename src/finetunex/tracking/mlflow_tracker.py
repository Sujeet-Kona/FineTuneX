"""Optional MLflow tracking for reproducible FineTuneX training runs.

MLflow is imported lazily so local MPS training and tests do not require it
unless experiment tracking is explicitly enabled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class MLflowTracker:
    """Small lifecycle wrapper around one optional MLflow run."""

    def __init__(self, settings: dict[str, Any], project_root: Path) -> None:
        if not isinstance(settings, dict):
            raise TypeError("experiment.tracking must be a mapping")
        self.settings = settings
        self.project_root = project_root
        self.enabled = settings.get("enabled", False)
        if not isinstance(self.enabled, bool):
            raise ValueError("experiment.tracking.enabled must be a boolean")
        self.mlflow: Any | None = None
        self.started = False

    def start(
        self,
        *,
        run_name: str,
        params: dict[str, Any],
        config_paths: tuple[Path, ...],
    ) -> None:
        if not self.enabled:
            return

        if self.settings.get("backend") != "mlflow":
            raise ValueError("Only the MLflow tracking backend is supported")

        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError(
                "MLflow tracking is enabled but MLflow is not installed. "
                "Install it with: python -m pip install -e '.[tracking]'"
            ) from exc

        tracking_uri = self.settings.get("tracking_uri")
        experiment_name = self.settings.get("experiment_name")
        if not isinstance(tracking_uri, str) or not tracking_uri.strip():
            raise ValueError("experiment.tracking.tracking_uri must be non-empty")
        if not isinstance(experiment_name, str) or not experiment_name.strip():
            raise ValueError("experiment.tracking.experiment_name must be non-empty")

        self.mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        self.started = True

        safe_params = {
            str(key): str(value)[:500]
            for key, value in params.items()
            if value is not None
        }
        if safe_params:
            mlflow.log_params(safe_params)

        if self.settings.get("log_system_metrics", False):
            mlflow.enable_system_metrics_logging()

        if self.settings.get("log_configurations", True):
            for path in config_paths:
                mlflow.log_artifact(str(path), artifact_path="config")

    def log_epoch(self, record: dict[str, Any]) -> None:
        if not self.started or self.mlflow is None:
            return

        metrics = {
            "train_loss": record["train_loss"],
            "validation_loss": record["validation_loss"],
            "train_supervised_tokens": record["train_supervised_tokens"],
            "validation_supervised_tokens": record["validation_supervised_tokens"],
            "train_microbatches": record["train_microbatches"],
            "train_optimizer_steps": record["train_optimizer_steps"],
            "mean_gradient_norm": record["mean_gradient_norm"],
            "validation_microbatches": record["validation_microbatches"],
        }
        self.mlflow.log_metrics(
            {key: float(value) for key, value in metrics.items()},
            step=int(record["epoch"]),
        )

    def finish(self, *, checkpoint_dir: Path | None = None, status: str = "FINISHED") -> None:
        if not self.started or self.mlflow is None:
            return

        try:
            if checkpoint_dir is not None:
                if self.settings.get("log_training_history", True):
                    history = checkpoint_dir / "training_history.json"
                    if history.is_file():
                        self.mlflow.log_artifact(str(history), artifact_path="training")
                if self.settings.get("log_checkpoint_metadata", True):
                    metadata = checkpoint_dir / "checkpoint_metadata.json"
                    if metadata.is_file():
                        self.mlflow.log_artifact(str(metadata), artifact_path="checkpoint")
        finally:
            self.mlflow.end_run(status=status)
            self.started = False
