"""Evaluate FinQA gold programs against their execution answers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from finetunex.evaluation.finqa_executor import (
    FinQAExecutionError,
    execute_program,
    parse_number,
)
from finetunex.evaluation.finqa_parser import parse_program
from finetunex.evaluation.finqa_reference import reference_execute_program

DATA_DIR = Path("data/processed/finqa")
SPLITS = ("train", "validation")
ERROR_EXAMPLES = 20

