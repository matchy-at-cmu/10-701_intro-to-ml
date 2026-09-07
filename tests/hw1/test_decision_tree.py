from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HOMEWORK_ROOT = REPOSITORY_ROOT / "hw1"
EXPECTED_OUTPUT_DIRECTORY = Path(__file__).with_name("expected")
EXPECTED_OUTPUTS = sorted(EXPECTED_OUTPUT_DIRECTORY.glob("*.txt"))
CASE_NAME_PATTERN = re.compile(
    r"(?P<dataset>[a-z0-9_]+)_max-depth(?P<max_depth>\d+)_"
    r"(?P<mode>train|prune)\.txt"
)
OUTPUT_FILE_NAMES = {
    "train": "trained_tree.txt",
    "prune": "pruned_tree.txt",
}


def find_decision_tree_script() -> Path:
    for script_path in (
        HOMEWORK_ROOT / "code" / "decision_tree.py",
        HOMEWORK_ROOT / "decision_tree.py",
    ):
        if script_path.is_file():
            return script_path

    raise FileNotFoundError("could not find the HW1 decision_tree.py script")


@pytest.mark.parametrize(
    "expected_output_path",
    EXPECTED_OUTPUTS,
    ids=[path.stem for path in EXPECTED_OUTPUTS],
)
def test_decision_tree_cli_matches_expected_output(
    expected_output_path: Path,
    tmp_path: Path,
) -> None:
    case_name = CASE_NAME_PATTERN.fullmatch(expected_output_path.name)
    assert case_name is not None, (
        f"invalid decision-tree fixture name: {expected_output_path.name}"
    )

    dataset = case_name["dataset"]
    max_depth = case_name["max_depth"]
    mode = case_name["mode"]
    result = subprocess.run(
        [
            sys.executable,
            str(find_decision_tree_script()),
            str(HOMEWORK_ROOT / "data" / f"{dataset}_train.tsv"),
            str(HOMEWORK_ROOT / "data" / f"{dataset}_val.tsv"),
            max_depth,
            mode,
        ],
        cwd=tmp_path,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode()

    expected_output = expected_output_path.read_bytes()
    assert expected_output, f"expected output is empty: {expected_output_path.name}"

    actual_outputs = {"stdout": result.stdout}
    generated_output_path = tmp_path / OUTPUT_FILE_NAMES[mode]
    if generated_output_path.exists():
        actual_outputs[generated_output_path.name] = generated_output_path.read_bytes()

    assert expected_output in actual_outputs.values(), actual_outputs
