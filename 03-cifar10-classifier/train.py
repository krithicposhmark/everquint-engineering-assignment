from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from cifar10.data import create_loaders
from cifar10.engine import collect_misclassified, evaluate, fit
from cifar10.models import build_model
from cifar10.plots import plot_history, plot_misclassified
from cifar10.utils import select_device, set_seed
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 classifiers.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("baseline", "residual"),
        default=("baseline", "residual"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")

    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}")
    results: dict[str, object] = {
        "configuration": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "seed": args.seed,
            "device": str(device),
            "train_limit": args.train_limit,
            "test_limit": args.test_limit,
        },
        "models": {},
    }

    train_loader, validation_loader, test_loader = create_loaders(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        args.seed,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
    )
    class_names = list(test_loader.dataset.dataset.classes)

    for model_name in args.models:
        print(f"\nmodel={model_name}")
        set_seed(args.seed)
        model = build_model(model_name, dropout=args.dropout).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        checkpoint_path = args.output_dir / f"{model_name}.pt"
        history = fit(
            model,
            train_loader,
            validation_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            args.epochs,
            checkpoint_path,
        )

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        examples = collect_misclassified(model, test_loader, device)
        plot_history(history, args.output_dir / f"{model_name}-training.png")
        plot_misclassified(
            examples, class_names, args.output_dir / f"{model_name}-misclassified.png"
        )
        results["models"][model_name] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "best_epoch": checkpoint["epoch"],
            "best_validation_accuracy": checkpoint["validation_accuracy"],
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "history": history,
        }
        print(f"test_loss={test_loss:.4f} test_acc={test_accuracy:.3f}")

    models = results["models"]
    if "baseline" in models and "residual" in models:
        results["test_accuracy_delta"] = (
            models["residual"]["test_accuracy"] - models["baseline"]["test_accuracy"]
        )
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)


if __name__ == "__main__":
    main()
