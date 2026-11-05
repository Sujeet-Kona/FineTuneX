from pathlib import Path
from types import SimpleNamespace

import pytest

from finetunex.tracking.mlflow_tracker import MLflowTracker


def test_disabled_tracker_does_not_import_mlflow(tmp_path):
    tracker = MLflowTracker(
        {
            "tracking": {
                "enabled": False,
                "backend": "mlflow",
                "tracking_uri": "mlruns",
                "experiment_name": "test",
            }
        },
        tmp_path,
    )

    tracker.start(run_name="disabled", params={}, config_paths=())
    tracker.log_epoch({"epoch": 1})
    tracker.finish(checkpoint_dir=tmp_path)

    assert tracker.started is False
    assert tracker.mlflow is None


def test_enabled_tracker_logs_parameters_metrics_and_artifacts(
    monkeypatch,
    tmp_path,
):
    calls = []
    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda value: calls.append(("uri", value)),
        set_experiment=lambda value: calls.append(("experiment", value)),
        start_run=lambda **kwargs: calls.append(("start", kwargs)),
        log_params=lambda value: calls.append(("params", value)),
        log_metrics=lambda value, step: calls.append(("metrics", value, step)),
        log_artifact=lambda value, artifact_path=None: calls.append(
            ("artifact", Path(value).name, artifact_path)
        ),
        end_run=lambda status: calls.append(("end", status)),
    )
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake_mlflow)

    config = tmp_path / "training.yaml"
    config.write_text("training: {}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "training_history.json").write_text("[]\n", encoding="utf-8")
    (checkpoint / "checkpoint_metadata.json").write_text("{}\n", encoding="utf-8")

    tracker = MLflowTracker(
        {
            "tracking": {
                "enabled": True,
                "backend": "mlflow",
                "tracking_uri": "file:./mlruns",
                "experiment_name": "finetunex-test",
            },
            "artifacts": {
                "log_configurations": True,
                "log_training_history": True,
                "log_checkpoint_metadata": True,
            },
        },
        tmp_path,
    )

    tracker.start(
        run_name="test-run",
        params={"learning_rate": 0.0002, "ignored": None},
        config_paths=(config,),
    )
    tracker.log_epoch(
        {
            "epoch": 1,
            "train_loss": 1.2,
            "validation_loss": 1.1,
            "train_supervised_tokens": 120,
            "validation_supervised_tokens": 40,
            "train_microbatches": 3,
            "train_optimizer_steps": 2,
            "mean_gradient_norm": 0.8,
            "validation_microbatches": 1,
        }
    )
    tracker.finish(checkpoint_dir=checkpoint)

    assert ("uri", "file:./mlruns") in calls
    assert ("experiment", "finetunex-test") in calls
    assert ("start", {"run_name": "test-run"}) in calls
    assert ("artifact", "training.yaml", "config") in calls
    assert ("artifact", "training_history.json", "training") in calls
    assert ("artifact", "checkpoint_metadata.json", "checkpoint") in calls
    assert any(call[0] == "metrics" and call[2] == 1 for call in calls)
    assert calls[-1] == ("end", "FINISHED")
    assert tracker.started is False


def test_enabled_tracker_gives_install_hint_when_mlflow_is_missing(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setitem(__import__("sys").modules, "mlflow", None)
    tracker = MLflowTracker(
        {
            "tracking": {
                "enabled": True,
                "backend": "mlflow",
                "tracking_uri": "mlruns",
                "experiment_name": "test",
            }
        },
        tmp_path,
    )

    with pytest.raises(RuntimeError, match=r"\[tracking\]"):
        tracker.start(run_name="missing", params={}, config_paths=())
