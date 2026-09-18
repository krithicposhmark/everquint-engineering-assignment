import unittest

import torch
from cifar10.models import ResidualBlock, build_model


class ModelTest(unittest.TestCase):
    def test_classifiers_return_ten_logits(self) -> None:
        inputs = torch.randn(2, 3, 32, 32)
        for name in ("baseline", "residual"):
            with self.subTest(name=name):
                self.assertEqual(build_model(name)(inputs).shape, (2, 10))

    def test_residual_block_projects_skip_connection(self) -> None:
        block = ResidualBlock(32, 64, stride=2)
        self.assertEqual(block(torch.randn(2, 32, 16, 16)).shape, (2, 64, 8, 8))

    def test_unknown_model_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_model("wide")


if __name__ == "__main__":
    unittest.main()
