# MNIST Model Lab

Colab에서 네 가지 CNN을 학습하고, Streamlit에서 직접 그린 숫자로 모델을 바꿔 가며 비교합니다. API 키 없이 Streamlit 서버에서 CPU 추론합니다.

**웹 앱:** [MNIST Model Lab](https://mnist-model-lab-yongza.streamlit.app/) · [GitHub](https://github.com/yooongZa/mnist-model-lab)

## Colab 학습

1. `MNIST_4모델_Colab.ipynb`를 Colab의 **파일 → 노트북 업로드**에서 엽니다.
2. **런타임 → 런타임 유형 변경 → GPU**를 선택합니다.
3. 위에서부터 실행하고 Google Drive 연결을 승인합니다.
4. `MyDrive/MNIST_Model_Lab/<실행 시각>/`에서 모델 파일을 내려받습니다.

노트북은 모델 정의와 학습 함수를 모두 포함합니다. 프로젝트 ZIP을 Colab에 올릴 필요는 없습니다. Colab 기본 PyTorch·torchvision·NumPy·pandas·Matplotlib 환경을 사용합니다.

| 모델 | 구조 | 학습 | 학습률 조절 | 조기 종료 |
|---|---|---:|---|---|
| 01 | 수업 CNN | 3회 | 없음 | 없음 |
| 02 | 수업 CNN | 10회 | ReduceLROnPlateau | 없음 |
| 03 | 합성곱 4층 | 최대 10회 | 동일 | patience=3 |
| 04 | 03 + BatchNorm2d | 최대 10회 | 동일 | patience=3 |

공통 Adam 0.001, 배치 64, seed 42. 학습용 6만 장을 55,000/5,000으로 고정 분할하고 별도 테스트 10,000장으로 최종 평가합니다. 최고 검증 정확도 시점의 가중치를 저장합니다. 정확도 동률은 손실로 결정합니다. 기존 수업의 98.81%는 당시 6만 장 전체 학습 기록이며, 이번 비교표는 새 분할에서 실행한 결과를 표시합니다.

실험당 저장 파일:

- `*_state_dict.pth`: state_dict + 구조·학습 결과·전처리 메타데이터.
- `*_torchscript.pt`: 실행 구조와 가중치, 같은 메타데이터.
- `metadata.json`, `history.csv`: 학습 설정·평가 결과·회차별 기록.
- 실행 폴더: `comparison.csv`, `learning_curves.png`, `split_indices.npz`, 전체 ZIP.

TorchScript는 요청한 저장 형식을 사용합니다. 내보낸 두 파일을 다시 불러와 이미지 1장·8장의 출력이 원 모델과 일치하는지 자동 검사합니다.

## 웹 실행

Python 3.12 환경에서:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**모델 추가 → 파일 선택 → 선택한 파일 추가**에서 `.pth`/`.pt`를 여러 개 등록합니다. **사용할 모델**에서 선택을 바꾸면 같은 손글씨로 예측합니다. 추가한 모델은 접속 세션 동안 유지됩니다. 재시작 후에도 기본으로 보이게 하려면 모델 파일을 `models/`에 넣습니다.

기본으로 포함한 두 파일은 이전 D46에서 이미 학습했던 같은 CNN의 state_dict와 TorchScript입니다. 새 네 가지 모델은 Colab 노트북을 실행하면 생성됩니다. `samples/`는 로컬에 이미 있던 MNIST 테스트 이미지 0~9 각 한 장입니다.

지원 모델:

- 이 노트북이 저장한 세 구조(4실험)의 state_dict 패키지.
- 기존 D46 `SimpleClassifier`의 raw state_dict.
- 직접 입력한 `nn.Sequential` 구조의 state_dict. 파일 업로드 후 **Sequential 코드 직접 입력**을 켜고 학습 때 사용한 구조를 붙여 넣은 뒤 **선택한 파일 추가**를 누릅니다.
- 같은 MNIST 전처리, `[B,1,28,28]` 입력, `[B,10]` logits 출력의 TorchScript.

Sequential 입력 예시:

```python
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(16 * 14 * 14, 10),
)
```

학습 코드의 층 순서·크기·이름을 그대로 입력합니다. `nn.Linear(784, ...)`로 시작하는 MLP는 **모델 입력 형태 → 펼친 이미지 [B, 784]**를 선택합니다. `OrderedDict`로 이름 붙인 층과 여러 Sequential 블록도 지원합니다. `model.`·`module.` 같은 공통 접두어는 전체 가중치 이름·크기가 맞을 때 자동으로 연결합니다. 같은 가중치라도 구조를 다르게 입력하면 별도 모델로 등록됩니다. 등록된 모델의 코드는 **선택한 모델의 학습 기록**에서 확인할 수 있습니다.

입력은 기존 MNIST 정규화(mean=0.1307, std=0.3081), 출력은 숫자 10개의 점수입니다. 지원 층은 웹의 **코드 입력 도움말**에 표시합니다. 입력한 코드는 층 정의와 인수를 AST로 해석합니다. state_dict는 `weights_only=True`로 읽습니다. 웹은 별도 API 서버·Google Drive 인증·사용자 API 키를 요구하지 않습니다.

## Streamlit Community Cloud에서 공유

GitHub 저장소에 앱 코드, `mnist_lab/`, `drawing_component/`, `requirements.txt`, `.streamlit/config.toml`, 필요한 `models/`와 `samples/`를 넣습니다. Community Cloud에서 이 저장소와 `streamlit_app.py`, Python 3.12를 선택합니다. Secrets 입력은 필요 없습니다. 업로드 파일과 접속 세션은 서비스 재시작 후 초기화될 수 있으므로 상시 모델은 저장소의 `models/`에 둡니다.

2026-11-06 Streamlit Community Cloud에 공개 배포했습니다. Python 3.12와 CPU용 PyTorch를 사용합니다. 웹에는 기존 수업 CNN의 두 저장 형식이 기본 등록되어 있습니다. Colab에서 네 실험의 전체 학습을 실행하면 새 모델을 생성할 수 있습니다.

## 개발·검증

```bash
pip install -r requirements-training.txt ipykernel
python scripts/build_notebook.py
python -m unittest discover -s tests -v
python scripts/verify_notebook.py --data-root /path/to/existing/MNIST-parent
```

`verify_notebook.py`는 임시 복제본에서 소규모 학습·평가·두 형식 저장을 실행합니다. 배포 노트북과 기존 수업 파일을 변경하지 않습니다.

## Based-On

`MLDL-03-02-C041@eb0de07d0dfd`, `MLDL-03-02-C046@6ec4349227cb`, `MLDL-03-02-C048@27b3217e12e2`, `MLDL-03-02-C057@46154b169e1c`, `MLDL-03-02-C058@63f3534b8362`, `MLDL-03-03-C026@bc8cb6348092`.

이 프로젝트는 개인 실습용 project_extension입니다. 직접 기준은 D46 수업 CNN이며 LMS raw 자료는 포함하지 않습니다. 변경 범위와 검증 계획은 `PLAN.md`에 있습니다.

공식 참고: [PyTorch MNIST](https://github.com/pytorch/examples/blob/main/mnist/main.py), [저장과 복원](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html), [Streamlit](https://docs.streamlit.io/).
