import unittest

import torch
from cifar10.engine import evaluate, train_one_epoch
from cifar10.models import build_model
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class EngineTest(unittest.TestCase):
    def test_train_and_evaluate_one_batch(self) -> None:
        torch.manual_seed(7)
        model = build_model("baseline", dropout=0.0)
        loader = DataLoader(
            TensorDataset(torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 3])),
            batch_size=4,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        train_loss, train_accuracy = train_one_epoch(
            model, loader, criterion, optimizer, torch.device("cpu")
        )
        test_loss, test_accuracy = evaluate(
            model, loader, criterion, torch.device("cpu")
        )

        self.assertGreater(train_loss, 0)
        self.assertGreater(test_loss, 0)
        self.assertGreaterEqual(train_accuracy, 0)
        self.assertLessEqual(test_accuracy, 1)


if __name__ == "__main__":
    unittest.main()
