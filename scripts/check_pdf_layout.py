import argparse
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import pymupdf

DEFAULT_TOLERANCE_POINTS = 0.5
PDF_DATE_PATTERN = re.compile(
    r"^D:(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})"
    r"(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})Z$"
)

Box = Tuple[float, float, float, float]
Word = Tuple[str, Box]


class LayoutMismatchError(Exception):
    pass


def reference_epoch(reference: pymupdf.Document) -> int:
    creation_date = reference.metadata.get("creationDate", "")
    match = PDF_DATE_PATTERN.fullmatch(creation_date)
    if match is None:
        raise LayoutMismatchError(
            "Reference PDF needs a UTC creation date for a reproducible build"
        )

    timestamp = datetime(
        *(
            int(match.group(name))
            for name in (
                "year",
                "month",
                "day",
                "hour",
                "minute",
                "second",
            )
        ),
        tzinfo=UTC,
    )
    return int(timestamp.timestamp())


def compile_pdf(source: Path, output_directory: Path, epoch: int) -> Path:
    environment = os.environ.copy()
    environment.update(
        {
            "FORCE_SOURCE_DATE": "1",
            "SOURCE_DATE_EPOCH": str(epoch),
            "TZ": "UTC",
        }
    )
    command = [
        "latexmk",
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-outdir={output_directory}",
        source.name,
    ]
    result = subprocess.run(
        command,
        cwd=source.parent,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        raise LayoutMismatchError(
            f"pdfLaTeX compilation failed:\n{result.stdout}\n{result.stderr}"
        )

    output = output_directory / f"{source.stem}.pdf"
    if not output.is_file():
        raise LayoutMismatchError(f"latexmk did not create {output}")
    return output


def words(page: pymupdf.Page) -> List[Word]:
    extracted = page.get_text("words", sort=True)
    return [(word[4], (word[0], word[1], word[2], word[3])) for word in extracted]


def drawing_boxes(page: pymupdf.Page) -> List[Box]:
    return [tuple(drawing["rect"]) for drawing in page.get_drawings()]


def maximum_coordinate_delta(expected: Box, actual: Box) -> float:
    return max(
        abs(expected_value - actual_value)
        for expected_value, actual_value in zip(expected, actual, strict=True)
    )


def compare_required_boxes(
    label: str,
    expected: Sequence[Box],
    actual: Sequence[Box],
    tolerance: float,
) -> None:
    unused_actual = set(range(len(actual)))
    for index, expected_box in enumerate(expected):
        candidates = [
            (maximum_coordinate_delta(expected_box, actual[actual_index]), actual_index)
            for actual_index in unused_actual
        ]
        if not candidates:
            raise LayoutMismatchError(f"{label} {index} is missing")

        delta, actual_index = min(candidates)
        if delta > tolerance:
            raise LayoutMismatchError(
                f"{label} {index} moved by {delta:.3f} pt (allowed {tolerance:.3f} pt)"
            )
        unused_actual.remove(actual_index)


def compare_required_words(
    label: str,
    expected: Sequence[Word],
    actual: Sequence[Word],
    tolerance: float,
) -> None:
    actual_by_text = {}
    for actual_index, (actual_text, _) in enumerate(actual):
        actual_by_text.setdefault(actual_text, set()).add(actual_index)

    for index, (expected_text, expected_box) in enumerate(expected):
        available = actual_by_text.get(expected_text, set())
        candidates = [
            (
                maximum_coordinate_delta(expected_box, actual[actual_index][1]),
                actual_index,
            )
            for actual_index in available
        ]
        if not candidates:
            raise LayoutMismatchError(f"{label} {index} ({expected_text!r}) is missing")

        delta, actual_index = min(candidates)
        if delta > tolerance:
            raise LayoutMismatchError(
                f"{label} {index} ({expected_text!r}) moved by {delta:.3f} pt "
                f"(allowed {tolerance:.3f} pt)"
            )
        available.remove(actual_index)


def compare_documents(
    reference: pymupdf.Document,
    generated: pymupdf.Document,
    tolerance: float,
) -> None:
    if not generated.metadata.get("producer", "").startswith("pdfTeX-"):
        raise LayoutMismatchError("Generated PDF was not produced by pdfLaTeX")

    if generated.page_count < reference.page_count:
        raise LayoutMismatchError(
            f"Pages are missing: expected at least {reference.page_count}, "
            f"got {generated.page_count}"
        )

    for page_number in range(reference.page_count):
        expected_page = reference[page_number]
        actual_page = generated[page_number]
        expected_page_box = tuple(expected_page.rect)
        actual_page_box = tuple(actual_page.rect)
        compare_required_boxes(
            f"Page {page_number + 1} boundary",
            [expected_page_box],
            [actual_page_box],
            tolerance,
        )

        expected_words = words(expected_page)
        actual_words = words(actual_page)
        compare_required_words(
            f"Page {page_number + 1} text",
            expected_words,
            actual_words,
            tolerance,
        )
        compare_required_boxes(
            f"Page {page_number + 1} drawing",
            drawing_boxes(expected_page),
            drawing_boxes(actual_page),
            tolerance,
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile a homework with pdfLaTeX and check its template layout"
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument(
        "--tolerance-points",
        default=DEFAULT_TOLERANCE_POINTS,
        type=float,
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    source = arguments.source.resolve()
    reference_path = arguments.reference.resolve()

    try:
        with pymupdf.open(reference_path) as reference:
            epoch = reference_epoch(reference)
            with tempfile.TemporaryDirectory(prefix="homework-layout-") as temporary:
                generated_path = compile_pdf(source, Path(temporary), epoch)
                with pymupdf.open(generated_path) as generated:
                    compare_documents(
                        reference,
                        generated,
                        arguments.tolerance_points,
                    )
    except (LayoutMismatchError, OSError, pymupdf.FileDataError) as error:
        print(f"layout check failed: {error}", file=sys.stderr)
        return 1

    print(f"layout check passed: {source} matches {reference_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
