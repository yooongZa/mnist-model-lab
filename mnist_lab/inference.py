"""업로드 모델 복원과 손글씨 전처리. 외부 서버/API 호출 없음."""
import base64
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from mnist_lab.models import MNISTClassifier, PREPROCESS, infer_architecture
from mnist_lab.sequential import restore_sequential


@dataclass
class LoadedModel:
    identifier: str
    name: str
    kind: str
    model: object
    metadata: dict


def load_model_bytes(data: bytes, filename: str, sequential_code=None, input_layout="image") -> LoadedModel:
    if not data or len(data) > 25 * 1024 * 1024:
        raise ValueError("모델 파일은 0보다 크고 25MB 이하여야 합니다.")
    stream = io.BytesIO(data)
    is_script = False
    if zipfile.is_zipfile(stream):
        with zipfile.ZipFile(stream) as archive:
            is_script = any("/code/" in name for name in archive.namelist())
    stream.seek(0)
    metadata = {}
    identifier = hashlib.sha256(data)
    if is_script:
        extra = {"metadata.json": ""}
        model = torch.jit.load(stream, map_location="cpu", _extra_files=extra)
        if extra["metadata.json"]:
            metadata = json.loads(extra["metadata.json"])
        kind = "TorchScript"
    else:
        saved = torch.load(stream, map_location="cpu", weights_only=True)
        if isinstance(saved, dict) and "state_dict" in saved:
            weights, metadata = saved["state_dict"], saved.get("metadata", {})
        elif isinstance(saved, dict) and "model_state_dict" in saved:
            weights, metadata = saved["model_state_dict"], saved.get("metadata", {})
        else:
            weights = saved
        if not isinstance(metadata, dict):
            raise ValueError("모델 메타데이터 형식이 올바르지 않습니다.")
        if sequential_code is not None:
            model = restore_sequential(sequential_code, weights, input_layout)
            metadata = {**metadata, "architecture": "직접 입력한 Sequential",
                        "sequential_code": sequential_code, "input_layout": input_layout}
            identity = json.dumps([sequential_code, input_layout], ensure_ascii=False)
            identifier.update(b"\0sequential\0" + identity.encode())
        else:
            try:
                architecture = infer_architecture(weights)
            except ValueError as error:
                raise ValueError("이 state_dict의 구조를 찾을 수 없습니다. 'Sequential 코드 직접 입력'에서 학습에 사용한 구조를 입력해 주세요.") from error
            model = MNISTClassifier(architecture)
            model.load_state_dict(weights, strict=True)
            metadata = {**metadata, "architecture": architecture}
        kind = "state_dict"
    if not isinstance(metadata, dict):
        raise ValueError("모델 메타데이터 형식이 올바르지 않습니다.")
    preprocessing = metadata.get("preprocess", PREPROCESS)
    if preprocessing != PREPROCESS:
        raise ValueError("이 모델의 입력 전처리가 MNIST Lab과 다릅니다.")
    model.eval()
    with torch.inference_mode():
        for batch_size in (1, 2):
            output = model(torch.zeros(batch_size, 1, 28, 28))
            if not isinstance(output, torch.Tensor) or output.shape != (batch_size, 10):
                raise ValueError("모델 입력은 [B,1,28,28], 출력은 숫자별 점수 [B,10]이어야 합니다.")
            if not torch.isfinite(output).all():
                raise ValueError("모델이 유효하지 않은 값을 출력했습니다.")
    name = str(metadata.get("name", filename))[:120]
    return LoadedModel(identifier.hexdigest(), name, kind, model, metadata)


def prepare_drawing(data_url):
    if not data_url:
        return None
    if len(data_url) > 4_000_000 or not data_url.startswith("data:image/png;base64,"):
        raise ValueError("손글씨 이미지 형식이 올바르지 않습니다.")
    data = base64.b64decode(data_url.split(",", 1)[1], validate=True)
    with Image.open(io.BytesIO(data)) as original:
        if original.width * original.height > 1_000_000:
            raise ValueError("입력 이미지가 너무 큽니다.")
        image = original.convert("L")
    bounds = image.point(lambda value: 255 if value > 16 else 0).getbbox()
    if bounds is None:
        return None
    digit = image.crop(bounds)
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)
    image = Image.new("L", (28, 28), 0)
    image.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))
    pixels = np.asarray(image, dtype=np.float32)
    total = pixels.sum()
    if total > 0:
        yy, xx = np.indices(pixels.shape)
        shift_x = int(round(13.5 - float((pixels * xx).sum() / total)))
        shift_y = int(round(13.5 - float((pixels * yy).sum() / total)))
        centered = Image.new("L", (28, 28), 0)
        centered.paste(image, (shift_x, shift_y))
        image = centered
    return image


def image_to_tensor(image):
    if image.size != (28, 28):
        raise ValueError("전처리된 입력은 28×28이어야 합니다.")
    pixels = np.array(image.convert("L"), dtype=np.float32) / 255.0
    return ((torch.from_numpy(pixels) - PREPROCESS["mean"]) / PREPROCESS["std"]).unsqueeze(0).unsqueeze(0)


def predict(loaded, image):
    with torch.inference_mode():
        logits = loaded.model(image_to_tensor(image))
        if logits.shape != (1, 10) or not torch.isfinite(logits).all():
            raise ValueError("유효한 숫자별 점수를 얻지 못했습니다.")
        probabilities = logits.softmax(dim=1)[0].cpu().numpy()
    return int(probabilities.argmax()), probabilities
