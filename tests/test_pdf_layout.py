import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_pdf_layout import LayoutMismatchError, discover_homework_templates

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HOMEWORK_TEMPLATES = discover_homework_templates(REPOSITORY_ROOT)


def test_homework_templates_are_discovered_by_directory_name(
    tmp_path: Path,
) -> None:
    for homework_name in ("hw1", "hw2"):
        homework_directory = tmp_path / homework_name
        latex_directory = homework_directory / "latex"
        latex_directory.mkdir(parents=True)
        (latex_directory / "main.tex").touch()
        (homework_directory / f"{homework_name}.pdf").touch()

    templates = discover_homework_templates(tmp_path)

    assert templates == [
        (tmp_path / "hw1" / "latex" / "main.tex", tmp_path / "hw1" / "hw1.pdf"),
        (tmp_path / "hw2" / "latex" / "main.tex", tmp_path / "hw2" / "hw2.pdf"),
    ]


def test_incomplete_homework_template_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "hw2").mkdir()

    with pytest.raises(LayoutMismatchError, match="hw2"):
        discover_homework_templates(tmp_path)


def run_layout_check(
    source: Path,
    reference: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "check_pdf_layout.py"),
            "--source",
            str(source),
            "--reference",
            str(reference),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )


@pytest.mark.parametrize(
    ("source", "reference"),
    HOMEWORK_TEMPLATES,
    ids=[source.parents[1].name for source, _ in HOMEWORK_TEMPLATES],
)
def test_empty_homework_visually_matches_course_handout(
    source: Path,
    reference: Path,
) -> None:
    result = run_layout_check(source, reference)

    assert result.returncode == 0, result.stdout + result.stderr


def test_visual_change_is_rejected(tmp_path: Path) -> None:
    homework_copy = tmp_path / "hw1" / "latex"
    shutil.copytree(
        REPOSITORY_ROOT / "hw1" / "latex",
        homework_copy,
        ignore=shutil.ignore_patterns(
            "*.aux",
            "*.fdb_latexmk",
            "*.fls",
            "*.log",
            "*.out",
            "*.pdf",
            "*.synctex.gz",
        ),
    )
    shutil.copytree(REPOSITORY_ROOT / "latex-commons", tmp_path / "latex-commons")
    source = homework_copy / "main.tex"
    original = source.read_text(encoding="utf-8")
    source.write_text(
        original.replace(
            "\\begin{document}",
            "\\begin{document}\n\\color{red}",
            1,
        ),
        encoding="utf-8",
    )

    result = run_layout_check(source, REPOSITORY_ROOT / "hw1" / "hw1.pdf")

    assert result.returncode == 1
    assert "visual output changed" in result.stderr


def test_global_t1_encoding_is_rejected(tmp_path: Path) -> None:
    homework_copy = tmp_path / "hw1" / "latex"
    shutil.copytree(
        REPOSITORY_ROOT / "hw1" / "latex",
        homework_copy,
        ignore=shutil.ignore_patterns(
            "*.aux",
            "*.fdb_latexmk",
            "*.fls",
            "*.log",
            "*.out",
            "*.pdf",
            "*.synctex.gz",
        ),
    )
    shutil.copytree(REPOSITORY_ROOT / "latex-commons", tmp_path / "latex-commons")
    source = homework_copy / "main.tex"
    original = source.read_text(encoding="utf-8")
    source.write_text(
        original.replace(
            "\\documentclass{article}",
            "\\documentclass{article}\n\\usepackage[T1]{fontenc}",
            1,
        ),
        encoding="utf-8",
    )

    result = run_layout_check(source, REPOSITORY_ROOT / "hw1" / "hw1.pdf")

    assert result.returncode == 1
    assert "visual output changed" in result.stderr
