import argparse

from finetunex.config import load_yaml_config
from finetunex.data.finqa_pipeline import prepare_finqa_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the official FinQA dataset.")
    parser.add_argument(
        "--config",
        default="configs/dataset.yaml",
        help="FinQA dataset configuration.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)["dataset"]

    manifest = prepare_finqa_dataset(
        repository=config["source"]["repository"],
        revision=config["source"]["revision"],
        output_dir=config["output"]["directory"],
        expected_counts={
            "train": config["expected"]["train_examples"],
            "validation": config["expected"]["validation_examples"],
            "test": config["expected"]["test_examples"],
        },
    )

    print("\nFINETUNEX — FINQA DATASET PREPARATION")
    print("=" * 48)
    print("Repository:", manifest["repository"])
    print("Revision:", manifest["revision"])

    for split, count in manifest["split_counts"].items():
        print(f"{split:12}: {count}")

    print("\nAUDIT")
    for split, details in manifest["audit"]["splits"].items():
        print(
            f"{split:12}: "
            f"blank answers={details['blank_answer_count']}, "
            f"raw final-step mismatches="
            f"{details['final_step_raw_mismatch_count']}"
        )

    print(
        "\nCross-split duplicate questions:",
        manifest["audit"]["cross_split_duplicate_question_count"],
    )
    print("Examples removed:", manifest["examples_removed"])
    print("Output:", config["output"]["directory"])


if __name__ == "__main__":
    main()
