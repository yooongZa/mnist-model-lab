from collections import OrderedDict
import io
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from mnist_lab.inference import load_model_bytes
from mnist_lab.sequential import build_sequential


def save_weights(weights):
    stream = io.BytesIO()
    torch.save(weights, stream)
    return stream.getvalue()


class SequentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_custom_cnn_and_batchnorm_match_original(self):
        original = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8),
                                 nn.ReLU(), nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
                                 nn.Linear(32, 10)).eval()
        source = """import torch
from torch import nn
model = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8),
    nn.ReLU(), nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
    nn.Linear(8 * 2 * 2, 10))"""
        loaded = load_model_bytes(save_weights(original.state_dict()), "new-cnn.pth", source)
        for batch in (1, 8):
            data = torch.randn(batch, 1, 28, 28)
            torch.testing.assert_close(original(data), loaded.model(data))

    def test_flat_mlp_and_wrapper_prefix(self):
        original = nn.Sequential(nn.Linear(784, 24), nn.Tanh(), nn.Linear(24, 10)).eval()
        source = "model = nn.Sequential(nn.Linear(28*28, 24), nn.Tanh(), nn.Linear(24, 10))"
        prefixed = {"module.network." + key: value for key, value in original.state_dict().items()}
        loaded = load_model_bytes(save_weights({"model_state_dict": prefixed}), "mlp.pth", source, "flat")
        data = torch.randn(2, 1, 28, 28)
        torch.testing.assert_close(original(data.flatten(1)), loaded.model(data))

    def test_named_nested_blocks(self):
        original = nn.Sequential(OrderedDict([
            ("features", nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU())),
            ("classifier", nn.Sequential(nn.Flatten(), nn.Linear(2 * 26 * 26, 10))),
        ])).eval()
        source = """from collections import OrderedDict
features = nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU())
classifier = nn.Sequential(nn.Flatten(), nn.Linear(2*26*26, 10))
model = nn.Sequential(OrderedDict([('features', features), ('classifier', classifier)]))
"""
        loaded = load_model_bytes(save_weights({"state_dict": original.state_dict()}), "named.pth", source)
        data = torch.randn(1, 1, 28, 28)
        torch.testing.assert_close(original(data), loaded.model(data))

    def test_structure_and_input_errors(self):
        weights = save_weights(nn.Sequential(nn.Flatten(), nn.Linear(784, 10)).state_dict())
        with self.assertRaisesRegex(ValueError, "가중치 이름·크기"):
            load_model_bytes(weights, "wrong.pth", "nn.Sequential(nn.Flatten(), nn.Linear(784, 9))")
        mlp = save_weights(nn.Sequential(nn.Linear(784, 10)).state_dict())
        with self.assertRaisesRegex(ValueError, "입력 형태"):
            load_model_bytes(mlp, "flat.pth", "nn.Sequential(nn.Linear(784, 10))")
        output9 = save_weights(nn.Sequential(nn.Flatten(), nn.Linear(784, 9)).state_dict())
        with self.assertRaisesRegex(ValueError, "출력"):
            load_model_bytes(output9, "nine.pth", "nn.Sequential(nn.Flatten(), nn.Linear(784, 9))")

    def test_definitions_cannot_execute_python(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "executed"
            sources = [
                f"open({str(target)!r}, 'w')",
                f"__import__('pathlib').Path({str(target)!r}).touch()",
                "import os\nmodel = nn.Sequential(nn.Flatten())",
                "model = nn.Sequential(*[nn.ReLU() for _ in range(3)])",
                "model = nn.Sequential(nn.Linear(10, 10)).eval()",
                "model = nn.Sequential(unknown())",
                "model = nn.Sequential(",
                "",
            ]
            for source in sources:
                with self.subTest(source=source), self.assertRaises(ValueError):
                    build_sequential(source)
            self.assertFalse(target.exists())

    def test_model_identity_and_registered_structure_remain_independent(self):
        original = nn.Sequential(nn.Flatten(), nn.Linear(784, 16), nn.ReLU(), nn.Linear(16, 10)).eval()
        weights = save_weights(original.state_dict())
        source = "nn.Sequential(nn.Flatten(), nn.Linear(784,16), nn.ReLU(), nn.Linear(16,10))"
        first = load_model_bytes(weights, "one.pth", source)
        repeated = load_model_bytes(weights, "two.pth", source)
        changed = load_model_bytes(weights, "one.pth", source.replace("ReLU", "Tanh"))
        self.assertEqual(first.identifier, repeated.identifier)
        self.assertNotEqual(first.identifier, changed.identifier)
        self.assertIsInstance(first.model.model[2], nn.ReLU)
        self.assertIsInstance(changed.model.model[2], nn.Tanh)
        data = torch.randn(1, 1, 28, 28)
        torch.testing.assert_close(original(data), first.model(data))

    def test_torchscript_ignores_sequential_editor(self):
        root = Path(__file__).resolve().parents[1]
        data = (root / "models/00_D46_CNN_torchscript.pt").read_bytes()
        first = load_model_bytes(data, "original.pt")
        second = load_model_bytes(data, "original.pt", "unfinished code", "flat")
        self.assertEqual(first.identifier, second.identifier)


if __name__ == "__main__":
    unittest.main()
