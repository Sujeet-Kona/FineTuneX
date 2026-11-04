import pytest

from finetunex.config import load_yaml_config


def test_load_valid_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("training:\n  epochs: 2\n", encoding="utf-8")

    result = load_yaml_config(config_file)

    assert result == {"training": {"epochs": 2}}


def test_missing_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_yaml_config(tmp_path / "missing.yaml")


def test_empty_yaml_raises_error(tmp_path):
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        load_yaml_config(config_file)


def test_yaml_list_at_root_raises_error(tmp_path):
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("- item1\n- item2\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_yaml_config(config_file)
