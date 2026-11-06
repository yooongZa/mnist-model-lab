from pathlib import Path
import io
import unittest
from unittest.mock import patch
import torch
from torch import nn
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class AppTests(unittest.TestCase):
    def test_custom_structure_upload_and_failure_preserves_models(self):
        original = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
        upload = io.BytesIO()
        torch.save(original.state_dict(), upload)
        upload.name = "custom_mlp.pth"
        with patch("streamlit.file_uploader", return_value=[upload]) as uploader:
            app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
            self.assertEqual(uploader.call_args.args[0], "모델 파일")
            self.assertFalse(any(item.label == "Colab 학습 노트북" for item in app.expander))
            app.checkbox(key="custom_structure").check().run()
            source = "model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))"
            app.text_area(key="sequential_source").set_value(source).run()
            next(button for button in app.button if button.label == "선택한 파일 추가").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.session_state["models"]), 3)
            custom_id = list(app.session_state["models"])[-1]
            app.radio[0].set_value("MNIST 예제").run()
            next(box for box in app.selectbox if box.label == "사용할 모델").select(custom_id).run()
            self.assertEqual(len(app.error), 0)
            app.text_area(key="sequential_source").set_value(source.replace("784, 10", "784, 9")).run()
            next(button for button in app.button if button.label == "선택한 파일 추가").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.session_state["models"]), 3)
            self.assertTrue(any("가중치 이름·크기" in error.value for error in app.error))
            self.assertEqual(app.session_state["models"][custom_id].metadata["sequential_code"], source)

    def test_choose_sample_switch_model_and_compare(self):
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.session_state["models"]), 2)
        app.radio[0].set_value("MNIST 예제").run()
        self.assertEqual(len(app.exception), 0)
        sample = next(box for box in app.selectbox if box.label == "예제 숫자")
        sample.select(ROOT / "samples/7.png").run()
        self.assertTrue(any('class="prediction">7' in value.value for value in app.markdown))
        chooser = next(box for box in app.selectbox if box.label == "사용할 모델")
        identifiers = list(app.session_state["models"])
        chooser.select(identifiers[1]).run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any('class="prediction">7' in value.value for value in app.markdown))
        next(button for button in app.button if button.label == "등록된 모델 모두 비교").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.dataframe[0].value), 2)


if __name__ == "__main__":
    unittest.main()
