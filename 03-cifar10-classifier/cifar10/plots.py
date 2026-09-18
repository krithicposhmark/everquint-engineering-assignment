from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cifar10.data import CIFAR10_MEAN, CIFAR10_STD


def plot_history(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(
        epochs, [row["validation_loss"] for row in history], label="validation"
    )
    axes[0].set(title="Loss", xlabel="Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="train")
    axes[1].plot(
        epochs, [row["validation_accuracy"] for row in history], label="validation"
    )
    axes[1].set(title="Accuracy", xlabel="Epoch", ylim=(0, 1))
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_misclassified(
    examples: list[tuple[torch.Tensor, int, int]],
    class_names: list[str],
    output_path: Path,
) -> None:
    if not examples:
        return
    columns = 4
    rows = int(np.ceil(len(examples) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(10, rows * 2.5), squeeze=False)
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    standard_deviation = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    for axis, (image, target, prediction) in zip(axes.flat, examples):
        image = (image * standard_deviation + mean).clamp(0, 1)
        axis.imshow(image.permute(1, 2, 0).numpy())
        axis.set_title(
            f"actual: {class_names[target]}\npredicted: {class_names[prediction]}"
        )
        axis.axis("off")
    for axis in axes.flat[len(examples) :]:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
