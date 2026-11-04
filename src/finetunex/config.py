from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and validate its root structure."""
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    if not config_path.is_file():
        raise ValueError(f"Configuration path is not a file: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping at its root.")

    return config
