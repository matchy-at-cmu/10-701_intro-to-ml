import argparse
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pymupdf
from numpy.typing import NDArray

RENDER_DPI = 144
# Independently generated PDFs can rasterize the same vector edge a few shades
# apart. Larger per-channel differences still expose visible content changes.
MAX_CHANNEL_DELTA = 20
MAX_CHANGED_PIXELS_PER_PAGE = 8
PDF_DATE_PATTERN = re.compile(
    r"^D:(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})"
    r"(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})Z$"
)

Pixels = NDArray[np.uint8]


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


def render_page(page: pymupdf.Page) -> Pixels:
    pixmap = page.get_pixmap(dpi=RENDER_DPI, colorspace=pymupdf.csRGB, alpha=False)
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height,
        pixmap.width,
        pixmap.n,
    )


def compare_documents(
    reference: pymupdf.Document,
    generated: pymupdf.Document,
) -> None:
    if not generated.metadata.get("producer", "").startswith("pdfTeX-"):
        raise LayoutMismatchError("Generated PDF was not produced by pdfLaTeX")

    if generated.page_count != reference.page_count:
        raise LayoutMismatchError(
            f"Page count changed: expected {reference.page_count}, "
            f"got {generated.page_count}"
        )

    for page_number in range(reference.page_count):
        expected = render_page(reference[page_number])
        actual = render_page(generated[page_number])
        if expected.shape != actual.shape:
            raise LayoutMismatchError(
                f"Page {page_number + 1} dimensions changed: "
                f"expected {expected.shape}, got {actual.shape}"
            )

        channel_delta = np.abs(expected.astype(np.int16) - actual.astype(np.int16))
        changed_pixels = np.any(channel_delta > MAX_CHANNEL_DELTA, axis=2)
        changed_count = int(np.count_nonzero(changed_pixels))
        if changed_count > MAX_CHANGED_PIXELS_PER_PAGE:
            raise LayoutMismatchError(
                f"visual output changed on page {page_number + 1}: "
                f"{changed_count} pixels differ at {RENDER_DPI} DPI"
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile a homework with pdfLaTeX and check its template layout"
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
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
                    compare_documents(reference, generated)
    except (LayoutMismatchError, OSError, pymupdf.FileDataError) as error:
        print(f"layout check failed: {error}", file=sys.stderr)
        return 1

    print(f"layout check passed: {source} matches {reference_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
