"""Sequential 모델 정의를 제한된 구문으로 읽는다. Python 코드를 실행하지 않는다."""
import ast
from collections import OrderedDict
import operator

import torch
from torch import nn


LAYER_NAMES = (
    "Sequential", "Conv2d", "Linear", "ReLU", "LeakyReLU", "PReLU", "GELU",
    "ELU", "SELU", "SiLU", "Tanh", "Sigmoid", "LogSoftmax", "Identity", "Flatten", "Unflatten",
    "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d", "AdaptiveMaxPool2d",
    "BatchNorm1d", "BatchNorm2d", "LayerNorm", "GroupNorm", "InstanceNorm2d",
    "Dropout", "Dropout1d", "Dropout2d", "ZeroPad2d", "ReflectionPad2d",
)
LAYERS = {name: getattr(nn, name) for name in LAYER_NAMES}
EXAMPLE_CODE = """model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(16 * 14 * 14, 10),
)"""


class _Reader:
    def __init__(self):
        self.values = {}

    def value(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool, type(None))):
            return node.value
        if isinstance(node, ast.Name) and node.id in self.values:
            return self.values[node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            values = [self.value(item) for item in node.elts]
            return tuple(values) if isinstance(node, ast.Tuple) else values
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self.value(node.operand)
            if type(value) in (int, float):
                return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            operations = {ast.Add: operator.add, ast.Sub: operator.sub,
                          ast.Mult: operator.mul, ast.Div: operator.truediv,
                          ast.FloorDiv: operator.floordiv}
            left, right = self.value(node.left), self.value(node.right)
            if type(left) in (int, float) and type(right) in (int, float) and type(node.op) in operations:
                value = operations[type(node.op)](left, right)
                if abs(value) <= 10_000_000:
                    return value
        if isinstance(node, ast.Call):
            path = self.call_name(node.func)
            if path == "OrderedDict":
                constructor = OrderedDict
            elif path.startswith("nn.") or path.startswith("torch.nn."):
                name = path.removeprefix("torch.").removeprefix("nn.")
                if name not in LAYERS:
                    raise ValueError(f"지원하는 층 목록에서 {name}을 확인해 주세요.")
                constructor = LAYERS[name]
            else:
                raise ValueError("nn.Sequential과 지원하는 nn 층으로 구조를 입력해 주세요.")
            if any(item.arg is None or item.arg in {"device", "dtype"} for item in node.keywords):
                raise ValueError("층의 인수에는 숫자·문자·튜플을 사용해 주세요. device와 dtype은 자동 설정됩니다.")
            args = [self.value(item) for item in node.args]
            kwargs = {item.arg: self.value(item.value) for item in node.keywords}
            return constructor(*args, **kwargs)
        raise ValueError(f"{getattr(node, 'lineno', '?')}행: 층 정의, 변수 대입, 숫자·튜플 인수를 사용해 주세요.")

    @staticmethod
    def call_name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return _Reader.call_name(node.value) + "." + node.attr
        return ""

    @staticmethod
    def allowed_import(node):
        if isinstance(node, ast.Import):
            return all((item.name, item.asname) in {
                ("torch", None), ("torch.nn", "nn")
            } for item in node.names)
        if isinstance(node, ast.ImportFrom):
            expected = {"torch": "nn", "collections": "OrderedDict"}
            return node.level == 0 and node.module in expected and all(
                item.name == expected[node.module] and item.asname in (None, item.name)
                for item in node.names
            )
        return False


def build_sequential(source):
    """구조를 meta 장치에 만들고, 반환 후 실제 가중치를 연결한다."""
    if not source.strip():
        raise ValueError("학습에 사용한 nn.Sequential 코드를 입력해 주세요.")
    if len(source) > 16_000:
        raise ValueError("모델 구조 코드는 16,000자 이내로 입력해 주세요.")
    try:
        tree = ast.parse(source)
        if sum(1 for _ in ast.walk(tree)) > 2000:
            raise ValueError("층 정의가 너무 깁니다.")
        reader = _Reader()
        last = None
        with torch.device("meta"):
            for statement in tree.body:
                if _Reader.allowed_import(statement):
                    continue
                if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                    name = statement.targets[0].id
                    if name in {"nn", "torch", "OrderedDict"} or name.startswith("_"):
                        raise ValueError("모델 변수 이름은 model 또는 net처럼 지정해 주세요.")
                    last = reader.value(statement.value)
                    reader.values[name] = last
                elif isinstance(statement, ast.Expr):
                    last = reader.value(statement.value)
                else:
                    raise ValueError(f"{statement.lineno}행: nn.Sequential 정의와 변수 대입을 입력해 주세요.")
        model = reader.values.get("model", last)
        if not isinstance(model, nn.Sequential):
            raise ValueError("최종 모델을 model = nn.Sequential(...) 형태로 입력해 주세요.")
        if len(list(model.modules())) > 100 or sum(p.numel() for p in model.parameters()) > 10_000_000:
            raise ValueError("모델은 100개 이하의 층과 1천만 개 이하의 파라미터를 사용해 주세요.")
        return model
    except SyntaxError as error:
        raise ValueError(f"{error.lineno}행 문법을 확인해 주세요: {error.msg}") from error
    except (TypeError, RuntimeError, ZeroDivisionError) as error:
        raise ValueError(f"층의 인수를 확인해 주세요: {error}") from error


def matching_weights(model, weights):
    """전체 key/shape가 일치할 때만 공통 wrapper 접두어를 제거한다."""
    if not isinstance(weights, dict) or not weights or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in weights.items()
    ):
        raise ValueError("파일에 올바른 state_dict 가중치가 없습니다.")
    expected = model.state_dict()
    prefixes = {""}
    first_key = next(iter(weights))
    for position, character in enumerate(first_key):
        if character == ".":
            prefix = first_key[:position + 1]
            if all(key.startswith(prefix) for key in weights):
                prefixes.add(prefix)
    for prefix in sorted(prefixes, key=len):
        candidate = {key[len(prefix):]: value for key, value in weights.items()}
        if set(candidate) == set(expected) and all(
            candidate[key].shape == expected[key].shape for key in expected
        ):
            return candidate
    raise ValueError("입력한 구조와 가중치 이름·크기가 다릅니다. 층 순서, 채널 수, Linear 크기, 층 이름을 학습 코드와 맞춰 주세요.")


class SequentialInput(nn.Module):
    def __init__(self, model, input_layout):
        super().__init__()
        self.model = model
        self.input_layout = input_layout

    def forward(self, images):
        if self.input_layout == "flat":
            images = images.flatten(1)
        return self.model(images)


def restore_sequential(source, weights, input_layout="image"):
    if input_layout not in {"image", "flat"}:
        raise ValueError("입력 형태를 확인해 주세요.")
    model = build_sequential(source)
    matched = matching_weights(model, weights)
    wrapped = SequentialInput(model, input_layout).eval()
    try:
        with torch.inference_mode():
            for batch in (1, 2):
                output = wrapped(torch.zeros(batch, 1, 28, 28, device="meta"))
                if not isinstance(output, torch.Tensor) or output.shape != (batch, 10):
                    raise ValueError("출력은 숫자 0~9의 점수 [B,10]이어야 합니다. 입력 형태와 마지막 층을 확인해 주세요.")
    except RuntimeError as error:
        raise ValueError(f"입력 형태와 층 사이 크기를 확인해 주세요: {error}") from error
    model.load_state_dict(matched, strict=True, assign=True)
    return wrapped
