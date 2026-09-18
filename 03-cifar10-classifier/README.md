# CIFAR-10 classifier

Two small models trained from scratch: a baseline CNN and the same-sized alternative built around residual blocks. Both use random crops, horizontal flips, channel normalization, AdamW and cosine learning-rate decay.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py --epochs 20
```

`train.py` downloads CIFAR-10, keeps a deterministic 10% validation split and writes checkpoints, metrics, learning curves and misclassified examples to `results/`. Use `--device auto` for CUDA, MPS or CPU selection. `--train-limit` and `--test-limit` are available for smoke runs.

## Layout

- `cifar10/data.py`: transforms and data loaders
- `cifar10/models.py`: baseline and residual networks
- `cifar10/engine.py`: training, checkpointing and evaluation
- `cifar10/plots.py`: curves and error samples
- `train.py`: experiment entry point
- `tests/`: model and training-loop checks

The residual model is the improvement experiment. The same split, seed, optimizer and training budget are used for both models so the comparison is repeatable.

## Recorded comparison

The checked-in run used seed 42, 10 epochs, 10,000 training-source images (9,000 train / 1,000 validation) and 2,000 test images on Apple MPS. It is a bounded comparison, not a full-dataset benchmark.

| Model | Parameters | Best validation | Test accuracy |
| --- | ---: | ---: | ---: |
| Baseline CNN | 1,148,874 | 62.30% | 59.85% |
| Residual CNN | 1,227,594 | 64.20% | 61.00% |

The residual block improved test accuracy by 1.15 percentage points under the same budget. Exact configuration, per-epoch metrics and the raw delta are in `results/metrics.json`.
