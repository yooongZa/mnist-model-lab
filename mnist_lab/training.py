"""학습·평가·검증 최고 모델의 독립 스냅샷과 두 형식 내보내기."""
import copy
import json
import math
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mnist_lab.models import MNISTClassifier, PREPROCESS


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cpu_snapshot(model):
    # cpu()만으로는 CPU 모델과 메모리를 공유하므로 clone()까지 수행한다.
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -math.inf
        self.bad_epochs = 0

    def update(self, accuracy):
        if accuracy > self.best + self.min_delta:
            self.best = accuracy
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = count = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("유한하지 않은 손실이 발생했습니다.")
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
    if count == 0:
        raise ValueError("데이터가 비어 있습니다.")
    return {"loss": total_loss / count, "accuracy": correct / count,
            "correct": correct, "count": count}


def fit_experiment(spec, train_loader, val_loader, test_loader, device, seed=42, on_epoch=print):
    seed_everything(seed)
    model = MNISTClassifier(spec.architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.learning_rate)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=1, min_lr=1e-5
    ) if spec.scheduler else None)
    stopper = EarlyStopping(spec.patience, spec.min_delta)
    best_score = (-math.inf, -math.inf)
    best_state = None
    best_epoch = 0
    history = []
    started = time.perf_counter()
    stop_reason = "max_epochs"
    for epoch in range(1, spec.epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        train = run_epoch(model, train_loader, device, optimizer)
        val = run_epoch(model, val_loader, device)
        row = {"epoch": epoch, "learning_rate": learning_rate,
               "train_loss": train["loss"], "train_accuracy": train["accuracy"],
               "val_loss": val["loss"], "val_accuracy": val["accuracy"]}
        history.append(row)
        score = (val["accuracy"], -val["loss"])
        if score > best_score:
            best_score, best_epoch, best_state = score, epoch, cpu_snapshot(model)
        if on_epoch:
            on_epoch(f"{spec.name} | {epoch:02d}/{spec.epochs} | "
                     f"train {train['accuracy']:.2%} | val {val['accuracy']:.2%} | "
                     f"val loss {val['loss']:.4f} | lr {learning_rate:.6f}")
        if scheduler is not None:
            scheduler.step(val["loss"])
        if spec.early_stopping and stopper.update(val["accuracy"]):
            stop_reason = "early_stopping"
            break
    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)
    test = run_epoch(model, test_loader, device)
    model.cpu().eval()
    result = {
        **spec.to_dict(), "seed": seed, "best_epoch": best_epoch,
        "epochs_ran": len(history), "stop_reason": stop_reason,
        "best_val_accuracy": best_score[0], "test": test,
        "training_seconds": elapsed,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "history": history, "torch_version": str(torch.__version__),
        "preprocess": dict(PREPROCESS),
        "selection": "validation accuracy; ties use lower validation loss",
    }
    return model, result


def export_model(model, metadata, run_directory):
    """검증이 끝난 묶음만 고유 디렉터리로 이동. 기존 실험 덮어쓰기 없음."""
    run_directory = Path(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    model_id = metadata["id"]
    if not model_id or Path(model_id).name != model_id or model_id in {".", ".."}:
        raise ValueError("잘못된 실험 이름입니다.")
    destination = run_directory / model_id
    if destination.exists():
        raise FileExistsError(f"기존 결과를 보존합니다. 새 실행 폴더를 사용하세요: {destination}")
    export_copy = copy.deepcopy(model).cpu().eval()
    metadata = {**metadata, "format_version": 1, "preprocess": dict(PREPROCESS)}
    with tempfile.TemporaryDirectory(prefix=".export-", dir=run_directory) as temporary:
        bundle = Path(temporary) / model_id
        bundle.mkdir()
        state_path = bundle / f"{model_id}_state_dict.pth"
        script_path = bundle / f"{model_id}_torchscript.pt"
        torch.save({"state_dict": cpu_snapshot(export_copy), "metadata": metadata}, state_path)
        probe = torch.zeros(1, 1, 28, 28)
        traced = torch.jit.trace(export_copy, probe, check_inputs=[(torch.ones(8, 1, 28, 28),)])
        torch.jit.save(traced, str(script_path), _extra_files={"metadata.json": json.dumps(metadata, ensure_ascii=False)})
        restored = MNISTClassifier(metadata["architecture"])
        restored.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True)["state_dict"])
        restored.eval()
        scripted = torch.jit.load(str(script_path), map_location="cpu").eval()
        with torch.inference_mode():
            for batch_size in (1, 8):
                generator = torch.Generator().manual_seed(71 + batch_size)
                inputs = torch.randn(batch_size, 1, 28, 28, generator=generator)
                original = export_copy(inputs)
                torch.testing.assert_close(restored(inputs), original, rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(scripted(inputs), original, rtol=1e-4, atol=1e-5)
        metadata["export_check"] = "state_dict and TorchScript match original; batch sizes 1,8"
        (bundle / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        bundle.rename(destination)
    return destination
