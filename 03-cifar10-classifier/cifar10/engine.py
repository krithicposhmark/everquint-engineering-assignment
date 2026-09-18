from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn


def _run_epoch(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in batches:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("cannot run an epoch with no examples")
    return total_loss / total_examples, total_correct / total_examples


def train_one_epoch(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    return _run_epoch(model, batches, criterion, device, optimizer)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    return _run_epoch(model, batches, criterion, device)


def fit(
    model: nn.Module,
    train_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    validation_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    checkpoint_path: Path,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    best_validation_accuracy = -1.0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, criterion, device
        )
        history.append(
            {
                "epoch": float(epoch),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.3f} "
            f"val_loss={validation_loss:.4f} val_acc={validation_accuracy:.3f}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_accuracy": validation_accuracy,
                },
                checkpoint_path,
            )
        scheduler.step()

    return history


@torch.inference_mode()
def collect_misclassified(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    limit: int = 16,
) -> list[tuple[torch.Tensor, int, int]]:
    model.eval()
    examples: list[tuple[torch.Tensor, int, int]] = []
    for inputs, targets in batches:
        predictions = model(inputs.to(device)).argmax(dim=1).cpu()
        for image, target, prediction in zip(inputs, targets, predictions):
            if target.item() != prediction.item():
                examples.append((image.cpu(), target.item(), prediction.item()))
                if len(examples) == limit:
                    return examples
    return examples
