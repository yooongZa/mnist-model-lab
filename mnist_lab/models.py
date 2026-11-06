"""D46 CNN과 합성곱/BatchNorm 개선 모델. 출력은 logits [B, 10]."""
from dataclasses import dataclass, asdict
import torch
from torch import nn

PREPROCESS = {"size": 28, "mean": 0.1307, "std": 0.3081,
              "foreground": "white", "output": "logits"}


class MNISTClassifier(nn.Module):
    def __init__(self, architecture: str = "baseline"):
        super().__init__()
        if architecture not in {"baseline", "deep", "deep_bn"}:
            raise ValueError(f"지원하지 않는 구조: {architecture}")
        layers = []
        channels = [(1, 32)]
        if architecture != "baseline":
            channels.append((32, 32))
        channels.append((32, 64))
        if architecture != "baseline":
            channels.append((64, 64))
        for index, (incoming, outgoing) in enumerate(channels):
            layers.append(nn.Conv2d(incoming, outgoing, kernel_size=3, padding=1))
            if architecture == "deep_bn":
                layers.append(nn.BatchNorm2d(outgoing))
            layers.append(nn.ReLU())
            if architecture == "baseline" or index in (1, 3):
                layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@dataclass(frozen=True)
class Experiment:
    id: str
    name: str
    architecture: str
    epochs: int
    scheduler: bool
    early_stopping: bool
    learning_rate: float = 0.001
    patience: int = 3
    min_delta: float = 0.0001

    def to_dict(self):
        return asdict(self)


EXPERIMENTS = (
    Experiment("01_baseline", "01 · 수업 CNN", "baseline", 3, False, False),
    Experiment("02_tuned", "02 · 학습 개선", "baseline", 10, True, False),
    Experiment("03_deep", "03 · 합성곱 추가", "deep", 10, True, True),
    Experiment("04_batchnorm", "04 · BatchNorm 추가", "deep_bn", 10, True, True),
)


def infer_architecture(state_dict):
    """지원하는 세 구조를 실제 key·shape로 확인한다. 원본 D46도 포함."""
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("state_dict에 가중치가 없습니다.")
    for architecture in ("baseline", "deep", "deep_bn"):
        reference = MNISTClassifier(architecture).state_dict()
        if set(reference) == set(state_dict) and all(
            isinstance(state_dict[key], torch.Tensor)
            and state_dict[key].shape == reference[key].shape for key in reference
        ):
            return architecture
    raise ValueError("이 state_dict의 구조를 찾을 수 없습니다. 이 프로젝트의 모델을 사용하거나 TorchScript로 내보내 주세요.")
