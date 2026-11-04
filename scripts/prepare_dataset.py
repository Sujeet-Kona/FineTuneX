import argparse
from pathlib import Path

from finetunex.config import load_yaml_config
from finetunex.data.pipeline import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate, format and split an instruction dataset."
    )
    parser.add_argument(
        "--config",
        default="configs/dataset.yaml",
        help="Dataset configuration file.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional input JSONL file override.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)["dataset"]

    input_dir = Path(config["input_path"])
    input_path = Path(args.input) if args.input else input_dir / "payment_support.jsonl"

    split = config["split"]

    manifest = prepare_dataset(
        input_path=input_path,
        output_dir=config["output_path"],
        columns=config["columns"],
        train_ratio=split["train"],
        validation_ratio=split["validation"],
        test_ratio=split["test"],
        seed=config["seed"],
    )

    print("FINETUNEX DATASET PREPARATION")
    print("=" * 40)
    print(f"Loaded:             {manifest['examples_loaded']}")
    print(f"Validated:          {manifest['examples_validated']}")
    print(f"Duplicates removed: {manifest['duplicates_removed']}")
    print(f"Train:              {manifest['split_counts']['train']}")
    print(f"Validation:         {manifest['split_counts']['validation']}")
    print(f"Test:               {manifest['split_counts']['test']}")
    print(f"Output:             {config['output_path']}")


if __name__ == "__main__":
    main()
