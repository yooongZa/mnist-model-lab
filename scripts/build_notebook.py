"""공유 Python 소스를 내장한, 별도 프로젝트 파일이 필요 없는 Colab 노트북."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text, tag=None):
    cell = nbf.v4.new_code_cell(text.strip())
    if tag:
        cell.metadata["tags"] = [tag]
    cells.append(cell)


md("""
# MNIST 정확도 개선: 네 모델 학습과 저장

수업 CNN에서 학습 방법과 합성곱 구성을 바꾸고, 손글씨 숫자를 얼마나 잘 맞히는지 비교합니다.
학습한 모델은 Google Drive에 **state_dict와 TorchScript 두 형식**으로 저장합니다.
Streamlit의 **모델 추가**에서 저장한 `.pth` 또는 `.pt` 파일을 선택하면 됩니다.

| 실험 | 모델 구조 | 최대 Epoch(학습 반복) | 학습률 조절 | Early stopping(조기 종료) |
|---|---|---:|---|---|
| 01 수업 CNN | 합성곱 2층 + MLP | 3 | 고정 0.001 | 없음 |
| 02 학습 개선 | 01과 같은 구조 | 10 | 검증 손실 정체 시 감소 | 없음 |
| 03 합성곱 추가 | 풀링 전 합성곱을 하나씩 추가: 총 4층 | 10 | 동일 | 검증 정확도 3회 연속 미개선 |
| 04 BatchNorm 추가 | 03의 각 Conv 뒤에 BatchNorm2d | 10 | 동일 | 동일 |

공통: Adam, 배치 64, seed 42, 같은 학습 55,000장·검증 5,000장·테스트 10,000장.
검증 정확도가 가장 높았던 가중치를 저장하며, 정확도가 같으면 검증 손실이 작은 시점을 고릅니다.
테스트 결과는 각 실험의 학습과 선택이 끝난 뒤 측정합니다.

**실행 준비:** Colab의 `런타임 → 런타임 유형 변경 → GPU`를 선택한 뒤 위에서부터 실행하세요.
첫 실행에는 MNIST 다운로드와 Google Drive 연결 승인이 있습니다. API 키는 사용하지 않습니다.
이 노트북에는 아직 전체 학습 결과가 저장되어 있지 않습니다. 아래 셀을 실행하면 결과가 만들어집니다.
""")

md("""
## 1. 실행 설정

`SMOKE_TEST=False`는 전체 실험입니다. `True`는 작은 데이터로 저장 흐름을 확인하는 모드입니다.
`MOUNT_DRIVE=True`이면 Google Drive의 `MyDrive/MNIST_Model_Lab` 아래에 결과가 쌓입니다.
같은 노트북을 다시 실행해도 이전 결과는 새 폴더로 보존됩니다.
""")
code('''
MOUNT_DRIVE = True
SMOKE_TEST = False
DOWNLOAD_DATA = True
SEED = 42
BATCH_SIZE = 64
DATA_ROOT = "/content/mnist_data"
OUTPUT_ROOT = "/content/drive/MyDrive/MNIST_Model_Lab"
''', "configuration")

md("## 2. 실행 환경과 Google Drive")
code('''
import importlib.util
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

if MOUNT_DRIVE:
    if importlib.util.find_spec("google.colab") is None:
        raise RuntimeError("Colab에서 실행하거나 MOUNT_DRIVE=False와 로컬 OUTPUT_ROOT를 설정하세요.")
    from google.colab import drive
    drive.mount("/content/drive")

import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if SMOKE_TEST:
    device = torch.device("cpu")
    torch.set_num_threads(2)
run_name = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
if SMOKE_TEST:
    run_name = "smoke-" + run_name
RUN_DIR = Path(OUTPUT_ROOT) / run_name
RUN_DIR.mkdir(parents=True, exist_ok=False)
print("PyTorch:", torch.__version__, "| torchvision:", torchvision.__version__)
print("장치:", device, "| 소규모 확인:", SMOKE_TEST)
print("결과 폴더:", RUN_DIR)
''')

md("""
## 3. 네 가지 실험의 모델 구조

`features`는 이미지의 특징을 뽑고, `classifier`는 그 특징으로 숫자별 점수 10개를 만듭니다.
③과 ④도 풀링을 두 번 사용하므로 특징의 최종 크기는 `64×7×7`입니다.
출력에는 Softmax를 붙이지 않고 `CrossEntropyLoss`에 점수를 바로 전달합니다.

④의 BatchNorm은 `Conv → BatchNorm → ReLU` 순서입니다.
아래 셀에는 웹과 동일한 모델 정의가 들어 있습니다.
""")
code((ROOT / "mnist_lab/models.py").read_text(), "shared-models")
code('''
display(pd.DataFrame([experiment.to_dict() for experiment in EXPERIMENTS])[
    ["name", "architecture", "epochs", "scheduler", "early_stopping"]
])
for architecture in ("baseline", "deep", "deep_bn"):
    print("\\n", architecture)
    print(MNISTClassifier(architecture))
''')

md("""
## 4. MNIST 데이터와 고정 분할

전처리: `ToTensor()`로 0~1 값의 float32 텐서를 만들고, 수업과 같은 평균 `0.1307`·표준편차 `0.3081`로 정규화합니다.
학습용 데이터에서 숫자마다 500장씩 뽑아 검증용 5,000장을 만듭니다. 네 실험이 같은 분할을 공유합니다.
정답은 `int64`, 이미지 한 묶음은 `[B, 1, 28, 28]`입니다.
""")
code('''
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_all = datasets.MNIST(DATA_ROOT, train=True, download=DOWNLOAD_DATA, transform=transform)
test_all = datasets.MNIST(DATA_ROOT, train=False, download=DOWNLOAD_DATA, transform=transform)
generator = torch.Generator().manual_seed(SEED)
train_indices, val_indices = [], []
for digit in range(10):
    indices = torch.where(train_all.targets == digit)[0]
    indices = indices[torch.randperm(len(indices), generator=generator)]
    n_val = 2 if SMOKE_TEST else 500
    val_indices.extend(indices[:n_val].tolist())
    remaining = indices[n_val:]
    train_indices.extend((remaining[:16] if SMOKE_TEST else remaining).tolist())
train_dataset = Subset(train_all, train_indices)
val_dataset = Subset(train_all, val_indices)
test_dataset = Subset(test_all, range(80)) if SMOKE_TEST else test_all
assert not set(train_indices) & set(val_indices)
np.savez(RUN_DIR / "split_indices.npz", train=train_indices, validation=val_indices)
print("학습:", len(train_dataset), "| 검증:", len(val_dataset), "| 테스트:", len(test_dataset))

def make_loaders():
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED), num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader

images, labels = next(iter(make_loaders()[0]))
print("이미지:", images.shape, images.dtype, "| 정답:", labels.shape, labels.dtype)
''')

md("""
## 5. 학습·검증·최고 가중치 저장 함수

- 학습: 예측 → 손실 → 기울기 계산 → Adam으로 가중치 갱신.
- 검증·테스트: `model.eval()`과 기울기 계산 중단을 적용합니다.
- 학습률: ②~④에서 검증 손실이 정체하면 0.3배로 줄입니다. 최소 0.00001입니다.
- 조기 종료: ③~④에서 검증 정확도가 0.0001(0.01%p)을 넘게 개선되지 않은 회차가 3회 쌓이면 멈춥니다.
- 최고 가중치: CPU의 독립 복제본으로 보관합니다. 정확도 동률에서는 손실로 선택합니다.
- 저장: 두 형식을 다시 불러와 이미지 1장·8장의 출력이 원 모델과 일치하는지 확인합니다.
""")
training_source = (ROOT / "mnist_lab/training.py").read_text()
training_source = training_source.replace("from mnist_lab.models import MNISTClassifier, PREPROCESS\n", "")
code(training_source, "shared-training")

md("""
## 6. 네 모델 학습과 Drive 저장

실험 하나가 끝날 때마다 해당 모델을 저장합니다. 중간에 런타임을 종료해도 이미 저장된 실험은 남습니다.
소규모 확인 모드에서는 각 실험을 2회까지만 실행합니다.
""")
code('''
from dataclasses import replace

results = []
for original_spec in EXPERIMENTS:
    spec = replace(original_spec, epochs=2) if SMOKE_TEST else original_spec
    print("\\n" + "=" * 55 + "\\n" + spec.name)
    train_loader, val_loader, test_loader = make_loaders()
    model, result = fit_experiment(spec, train_loader, val_loader, test_loader, device, seed=SEED)
    result.update({"smoke_test": SMOKE_TEST, "dataset": "MNIST",
                   "train_size": len(train_dataset), "validation_size": len(val_dataset),
                   "test_size": len(test_dataset), "batch_size": BATCH_SIZE,
                   "run_name": run_name})
    saved_dir = export_model(model, result, RUN_DIR)
    pd.DataFrame(result["history"]).to_csv(saved_dir / "history.csv", index=False)
    results.append(result)
    (RUN_DIR / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장:", saved_dir)
    print("테스트 정확도:", f"{result['test']['accuracy']:.2%}", "| 저장 회차:", result["best_epoch"])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
''')

md("""
## 7. 정확도와 학습 과정 비교

표에는 **이번에 실제 실행한 결과**가 표시됩니다.
`test_accuracy`는 테스트 정답 수 ÷ 테스트 이미지 수이며, `errors`는 틀린 이미지 수입니다.
③·④의 실제 학습 횟수와 최고 성능 회차도 함께 확인하세요.
""")
code('''
summary = pd.DataFrame([{
    "model": r["name"], "epochs_ran": r["epochs_ran"], "best_epoch": r["best_epoch"],
    "val_accuracy_pct": r["best_val_accuracy"] * 100,
    "test_accuracy_pct": r["test"]["accuracy"] * 100,
    "errors": r["test"]["count"] - r["test"]["correct"],
    "training_seconds": r["training_seconds"], "stop_reason": r["stop_reason"],
} for r in results])
display(summary.round(3))
summary.to_csv(RUN_DIR / "comparison.csv", index=False)

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
colors = ["#3568A0", "#B58B2A", "#C46D35", "#778C45"]
markers = ["o", "s", "^", "D"]
for r, color, marker in zip(results, colors, markers):
    frame = pd.DataFrame(r["history"])
    axes[0].plot(frame["epoch"], frame["val_accuracy"] * 100, marker=marker, color=color, label=r["id"])
    axes[1].plot(frame["epoch"], frame["val_loss"], marker=marker, color=color, label=r["id"])
axes[0].set(xlabel="Epoch", ylabel="Validation accuracy (%)", title="Validation accuracy")
axes[1].set(xlabel="Epoch", ylabel="Validation loss", title="Validation loss")
for ax in axes:
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(RUN_DIR / "learning_curves.png", bbox_inches="tight")
plt.show()
if SMOKE_TEST:
    print("소규모 실행 확인 결과입니다. 전체 성능 실험은 SMOKE_TEST=False로 실행하세요.")
''')

md("""
## 8. 결과 묶음과 Streamlit 연결

각 모델 폴더에는 `_state_dict.pth`, `_torchscript.pt`, `metadata.json`, `history.csv`가 있습니다.
`.pth`에는 가중치와 모델 구조 정보가, `.pt`에는 실행 구조·가중치와 메타데이터가 들어 있습니다.
두 파일 중 하나씩 골라 웹에 추가하거나, 두 형식을 함께 추가해 동일한 예측을 확인할 수 있습니다.

Google Drive에서 원하는 파일을 내려받고 Streamlit의 **모델 추가 → 파일 선택 → 선택한 파일 추가**를 누르세요.
**사용할 모델** 메뉴에서 네 모델을 번갈아 선택하면 같은 그림으로 다시 예측합니다.
""")
code('''
import shutil
archive = shutil.make_archive(str(RUN_DIR), "zip", root_dir=RUN_DIR)
print("모든 결과 ZIP:", archive)
print("모델 파일:")
for path in sorted(RUN_DIR.glob("*/*")):
    if path.suffix in {".pth", ".pt"}:
        print(" -", path.relative_to(RUN_DIR))
''')

md("""
## 참고와 변경 범위

- 직접 기준: D46 `SimpleClassifier`, Adam 0.001, 배치 64, 정규화 0.1307/0.3081.
- ③ 합성곱 추가, ④ BatchNorm2d, 검증 분할과 조기 종료·학습률 조절은 이번 실험의 확장입니다.
- TorchScript 저장은 요청한 수업 형식을 따릅니다.
- [PyTorch MNIST 예제](https://github.com/pytorch/examples/blob/main/mnist/main.py)
- [Streamlit](https://docs.streamlit.io/)

Based-On: `MLDL-03-02-C041@eb0de07d0dfd`, `MLDL-03-02-C046@6ec4349227cb`,
`MLDL-03-02-C048@27b3217e12e2`, `MLDL-03-02-C057@46154b169e1c`,
`MLDL-03-02-C058@63f3534b8362`, `MLDL-03-03-C026@bc8cb6348092`.
산출물 분류: project_extension. LMS 보관본은 포함하지 않습니다.
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.update({
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
    "colab": {"name": "MNIST_4모델_Colab.ipynb", "provenance": []},
    "accelerator": "GPU",
})
nbf.validate(notebook)
destination = ROOT / "MNIST_4모델_Colab.ipynb"
nbf.write(notebook, destination)
print(destination)
