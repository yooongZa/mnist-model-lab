"""배포 노트북의 임시 복제본만 새 커널에서 소규모 실행한다."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--data-root", required=True)
args = parser.parse_args()
source = ROOT / "MNIST_4모델_Colab.ipynb"
before = hashlib.sha256(source.read_bytes()).hexdigest()
notebook = nbformat.read(source, as_version=4)
nbformat.validate(notebook)
qa = ROOT / "qa"
qa.mkdir(exist_ok=True)
for cell in notebook.cells:
    if cell.cell_type != "code":
        continue
    ast.parse(cell.source)
    if "configuration" in cell.metadata.get("tags", []):
        cell.source = "\n".join([
            "MOUNT_DRIVE = False", "SMOKE_TEST = True", "DOWNLOAD_DATA = False",
            "SEED = 42", "BATCH_SIZE = 64",
            f"DATA_ROOT = {str(Path(args.data_root).resolve())!r}",
            f"OUTPUT_ROOT = {str(qa)!r}",
        ])
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv[0] = sys.executable
client = NotebookClient(notebook, km=manager, timeout=180,
                        resources={"metadata": {"path": str(ROOT)}},
                        allow_errors=False)
try:
    client.execute()
finally:
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
executed = qa / "notebook-smoke.ipynb"
nbformat.write(notebook, executed)
assert before == hashlib.sha256(source.read_bytes()).hexdigest()
runs = sorted(qa.glob("smoke-*/results.json"), key=lambda path: path.stat().st_mtime)
result = json.loads(runs[-1].read_text())
assert len(result) == 4
assert all(row["smoke_test"] and row["epochs_ran"] <= 2 for row in result)
assert len(list(runs[-1].parent.glob("*/*.pth"))) == 4
assert len(list(runs[-1].parent.glob("*/*.pt"))) == 4
report = {"notebook": str(source), "source_sha256": before, "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
          "error_outputs": 0, "mode": "smoke", "data_download": False,
          "run_directory": str(runs[-1].parent), "models": len(result), "state_dict_files": 4,
          "torchscript_files": 4, "google_drive": "not_executed", "full_training": "not_executed"}
(qa / "notebook-verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False, indent=2))
