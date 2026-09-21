"""Plot metrics written by the training scripts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRIC_NAMES = (
    "episode_time",
    "total_influence",
    "discovery_band",
    "spread_band",
    "minimum_discovery",
    "minimum_spread",
)


def _number(value: str) -> float:
    return float(value.strip().strip("[]"))


def load_metrics(path: Path, section: int = 0) -> tuple[list[str], np.ndarray]:
    """Read one experiment section from a training output file."""
    rows: list[list[float]] = []
    current_section = 0

    for raw_line in path.read_text().splitlines()[1:]:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("!"):
            current_section += 1
            continue
        if current_section != section:
            continue

        parts = [part.strip() for part in next(csv.reader([line]))]
        if len(parts) == 1:
            rows.append([_number(parts[0])])
            continue

        values = [_number(part) for part in parts[:4]]
        if len(parts) >= 6 and not parts[4].lstrip().startswith("["):
            values.extend((_number(parts[4]), _number(parts[5])))
        elif len(parts) >= 10:
            values.extend(
                (
                    min(_number(part) for part in parts[4:7]),
                    min(_number(part) for part in parts[7:10]),
                )
            )
        rows.append(values)

    if not rows:
        raise ValueError(f"No numeric rows found in section {section} of {path}")

    width = min(len(row) for row in rows)
    data = np.asarray([row[:width] for row in rows], dtype=float)
    names = ["total_influence"] if width == 1 else list(METRIC_NAMES[:width])
    return names, data


def plot_metrics(path: Path, output_dir: Path, section: int = 0) -> list[Path]:
    names, data = load_metrics(path, section=section)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    start_column = 0 if len(names) == 1 else 1
    for column in range(start_column, len(names)):
        metric = names[column]
        values = data[:, column]
        figure, axis = plt.subplots()
        axis.plot(values)
        axis.axhline(values.mean(), linestyle=":", label=f"mean = {values.mean():.5f}")
        axis.set_xlabel("Episode")
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.legend()
        figure.tight_layout()

        destination = output_dir / f"{path.stem}_{metric}.png"
        figure.savefig(destination, dpi=150)
        plt.close(figure)
        written.append(destination)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Training output text file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/generated"),
        help="Directory for generated figures",
    )
    parser.add_argument("--section", type=int, default=0, help="Zero-based experiment section")
    args = parser.parse_args()

    for output in plot_metrics(args.input, args.output_dir, section=args.section):
        print(output)


if __name__ == "__main__":
    main()
