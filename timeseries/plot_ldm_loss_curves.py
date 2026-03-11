#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


LOSS_PATTERN = re.compile(r"Loss\s*:\s*([0-9]*\.?[0-9]+)")


def parse_losses(log_path: Path) -> list[float]:
    losses: list[float] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            match = LOSS_PATTERN.search(line)
            if match:
                losses.append(float(match.group(1)))
    if not losses:
        raise ValueError(f"No losses found in {log_path}")
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LDM loss curves from four logs.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window in addition to saving the PNG.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    logs = [
        ("baseline", root / "ldm_training_log.txt", "tab:blue"),
        ("timeseries", root / "ldm_training_log_ts.txt", "tab:orange"),
        ("conv1d", root / "ldm_training_log_ts_conv1d.txt", "tab:green"),
        ("gru", root / "ldm_training_log_ts_gru.txt", "tab:red"),
    ]

    plt.figure(figsize=(8, 5))
    for label, path, color in logs:
        losses = parse_losses(path)
        epochs = range(1, len(losses) + 1)
        plt.plot(epochs, losses, label=label, color=color, linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LDM Training Loss Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = root / "ldm_loss_curves.png"
    plt.savefig(output_path, dpi=200)
    if args.show:
        plt.show()
    else:
        plt.close()
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
