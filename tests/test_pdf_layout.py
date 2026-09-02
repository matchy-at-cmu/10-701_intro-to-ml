import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def run_layout_check(source: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "check_pdf_layout.py"),
            "--source",
            str(source),
            "--reference",
            str(REPOSITORY_ROOT / "hw1" / "hw1.pdf"),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )


def test_empty_hw1_visually_matches_course_handout() -> None:
    source = REPOSITORY_ROOT / "hw1" / "hw1_latex" / "main.tex"
    result = run_layout_check(source)

    assert result.returncode == 0, result.stdout + result.stderr


def test_visual_change_is_rejected(tmp_path: Path) -> None:
    homework_copy = tmp_path / "hw1" / "hw1_latex"
    shutil.copytree(
        REPOSITORY_ROOT / "hw1" / "hw1_latex",
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
    shutil.copytree(REPOSITORY_ROOT / "latex", tmp_path / "latex")
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

    result = run_layout_check(source)

    assert result.returncode == 1
    assert "visual output changed" in result.stderr


def test_global_t1_encoding_is_rejected(tmp_path: Path) -> None:
    homework_copy = tmp_path / "hw1" / "hw1_latex"
    shutil.copytree(
        REPOSITORY_ROOT / "hw1" / "hw1_latex",
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
    shutil.copytree(REPOSITORY_ROOT / "latex", tmp_path / "latex")
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

    result = run_layout_check(source)

    assert result.returncode == 1
    assert "visual output changed" in result.stderr
