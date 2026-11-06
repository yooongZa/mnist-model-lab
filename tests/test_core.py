import base64
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw
import torch

from mnist_lab.models import EXPERIMENTS, MNISTClassifier
from mnist_lab.inference import load_model_bytes, prepare_drawing, image_to_tensor, predict
from mnist_lab.training import cpu_snapshot, EarlyStopping, export_model

ROOT = Path(__file__).resolve().parents[1]


def drawing_url(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_d46_original_formats_match(self):
        raw = load_model_bytes((ROOT / "models/00_D46_CNN_state_dict.pth").read_bytes(), "original.pth")
        script = load_model_bytes((ROOT / "models/00_D46_CNN_torchscript.pt").read_bytes(), "original.pt")
        for digit in range(10):
            image = Image.open(ROOT / f"samples/{digit}.png")
            a, pa = predict(raw, image)
            b, pb = predict(script, image)
            self.assertEqual(a, b)
            np.testing.assert_allclose(pa, pb, rtol=1e-5, atol=1e-6)

    def test_export_all_architectures_and_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            for spec in EXPERIMENTS:
                original = MNISTClassifier(spec.architecture).eval()
                metadata = {**spec.to_dict(), "preprocess": None}
                directory = export_model(original, metadata, temporary)
                state = load_model_bytes(next(directory.glob("*.pth")).read_bytes(), "model.pth")
                script = load_model_bytes(next(directory.glob("*.pt")).read_bytes(), "model.pt")
                self.assertEqual(state.metadata["architecture"], spec.architecture)
                self.assertEqual(script.metadata["name"], spec.name)
                image = Image.open(ROOT / "samples/7.png")
                _, a = predict(state, image)
                _, b = predict(script, image)
                np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
                with self.assertRaises(FileExistsError):
                    export_model(original, metadata, temporary)

    def test_snapshot_includes_independent_batchnorm_buffers(self):
        model = MNISTClassifier("deep_bn")
        snapshot = cpu_snapshot(model)
        original = {key: value.clone() for key, value in snapshot.items()}
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1)
            model(torch.ones(2, 1, 28, 28))
        for key in original:
            torch.testing.assert_close(snapshot[key], original[key])

    def test_early_stopping_resets_after_improvement(self):
        stopper = EarlyStopping(patience=3)
        self.assertEqual([stopper.update(value) for value in [.9, .9, .91, .91, .91, .91]],
                         [False, False, False, False, False, True])

    def test_invalid_files_are_rejected(self):
        for data in (b"", b"not a model"):
            with self.assertRaises(Exception):
                load_model_bytes(data, "bad.pth")
        stream = io.BytesIO()
        torch.save({"weight": torch.zeros(2, 3)}, stream)
        with self.assertRaisesRegex(ValueError, "구조"):
            load_model_bytes(stream.getvalue(), "other.pth")

    def test_canvas_empty_and_centered_tensor(self):
        empty = Image.new("L", (280, 280), 0)
        self.assertIsNone(prepare_drawing(drawing_url(empty)))
        self.assertIsNone(prepare_drawing(None))
        drawing = empty.copy()
        ImageDraw.Draw(drawing).line([(220, 25), (240, 60), (210, 150)], fill=255, width=18)
        normalized = prepare_drawing(drawing_url(drawing))
        self.assertEqual(normalized.size, (28, 28))
        pixels = np.asarray(normalized)
        yy, xx = np.indices(pixels.shape)
        self.assertLess(abs(float((xx * pixels).sum() / pixels.sum()) - 13.5), .6)
        self.assertLess(abs(float((yy * pixels).sum() / pixels.sum()) - 13.5), .6)
        tensor = image_to_tensor(normalized)
        self.assertEqual(tuple(tensor.shape), (1, 1, 28, 28))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.isfinite(tensor).all())


if __name__ == "__main__":
    unittest.main()
