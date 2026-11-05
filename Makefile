.PHONY: install data preprocess audit train train-smoke evaluate-base evaluate-adapter compare export serve docker-build docker-up docker-down test lint format-check clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[ml,api,dev]"

data:
	$(PYTHON) scripts/prepare_dataset.py

preprocess:
	$(PYTHON) scripts/preprocess_finqa.py

audit:
	$(PYTHON) scripts/evaluate_gold_programs.py

train:
	$(PYTHON) scripts/train_finqa.py

train-smoke:
	$(PYTHON) scripts/train_finqa.py --max-train-batches 2 --max-validation-batches 2 --output-dir artifacts/checkpoints/smoke

evaluate-base:
	$(PYTHON) scripts/evaluate_finqa_model.py --base-model --output-dir artifacts/evaluation/v4-base-full

evaluate-adapter:
	@test -n "$(CHECKPOINT)" || (echo "Set CHECKPOINT=artifacts/checkpoints/<checkpoint>" && exit 2)
	$(PYTHON) scripts/evaluate_finqa_model.py --checkpoint "$(CHECKPOINT)" --output-dir artifacts/evaluation/v4-adapter-full

compare:
	$(PYTHON) scripts/compare_finqa_evaluations.py --base-dir artifacts/evaluation/v4-base-full --adapter-dir artifacts/evaluation/v4-adapter-full --output artifacts/evaluation/v4-paired-full-comparison.json

export:
	@test -n "$(CHECKPOINT)" || (echo "Set CHECKPOINT=artifacts/checkpoints/<checkpoint>" && exit 2)
	$(PYTHON) scripts/export_finqa_adapter.py --checkpoint "$(CHECKPOINT)"

serve:
	@test -n "$(CHECKPOINT)" || (echo "Set CHECKPOINT=artifacts/checkpoints/<checkpoint>" && exit 2)
	FINETUNEX_CHECKPOINT="$(CHECKPOINT)" uvicorn finetunex.api.app:app --host 127.0.0.1 --port 8000

docker-build:
	docker compose build

docker-up:
	docker compose up --build

docker-down:
	docker compose down

test:
	pytest -q

lint:
	ruff check .

format-check:
	ruff format --check .

clean:
	rm -rf .pytest_cache .ruff_cache
