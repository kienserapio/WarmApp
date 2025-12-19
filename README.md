# WorkoutBuddy
A simple pushup form detector that uses the OpenCV and MediaPipe frameworks.
# WorkoutBuddy

Lightweight toolkit for running the hybrid workout posture detection demos and metrics utilities used in this project.

## Quick overview

- Main detection demos: `main.py`, `main_hybrid.py`, `main_with_yolo.py`, `main_yolo_primary.py`.
- Metrics and analysis: `calculate_metrics.py`, `extract_model_metrics.py`, `test_model_structure.py`.
- Models are stored under the project `models/` directory (each model folder contains `.pt`, `metadata.yaml`, `model.json`).

## Requirements

- Python 3.11 (3.10+ generally works; 3.11 recommended)
- Git (optional)

This repo was developed and tested with the following notable packages (versions used in analysis):

- `ultralytics` (YOLOv8) ~8.3.240
- `mediapipe` ~0.10.21
- `opencv-python` ~4.12
- `torch` (install following official PyTorch instructions for your platform)
- `numpy`, `pandas`, `matplotlib`, `seaborn`

## Install dependencies (recommended)

1. Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

2. Install core Python packages. There are two recommended approaches depending on your platform:

- macOS (Apple Silicon / MPS) or CPU-only systems — follow the official PyTorch instructions at https://pytorch.org/get-started/locally/ to install the best `torch` wheel for your machine, then install the remaining packages:

```bash
# Example (after you install torch per PyTorch site):
pip install ultralytics==8.3.240 mediapipe==0.10.21 opencv-python numpy pandas matplotlib seaborn
```

- Linux / CUDA GPU systems — install CUDA-compatible `torch` via the PyTorch selector at https://pytorch.org/get-started/locally/, then install remaining packages:

```bash
# Example (after you install torch per PyTorch site):
pip install ultralytics mediapipe opencv-python numpy pandas matplotlib seaborn
```

Notes:
- Installing `torch` should use the platform-specific instructions from PyTorch; using the recommended wheel ensures CUDA or MPS acceleration when available.
- If you prefer a single command and are not using GPU/CUDA, you can try:

```bash
pip install torch torchvision torchaudio
pip install ultralytics mediapipe opencv-python numpy pandas matplotlib seaborn
```

## Running the demos

- Run the primary hybrid demo (MediaPipe + YOLO fusion):

```bash
python WorkoutBuddy/main_hybrid.py
```

- Run the YOLO-only demo:

```bash
python WorkoutBuddy/main_yolo_primary.py
```

- Run the simple main demo:

```bash
python WorkoutBuddy/main.py
```

## Metrics and evaluation

- Calculate confusion-matrix style metrics and generate F1-confidence curves (requires a labelled validation / test dataset):

```bash
python WorkoutBuddy/calculate_metrics.py --test_dir /path/to/test/images --generate_curves
```

- Extract model-level metadata and attempt to locate training history:

```bash
python WorkoutBuddy/extract_model_metrics.py
```

Recommended configuration (used in the paper and analysis):

- `confidence threshold = 0.7`
- `iou threshold = 0.45`

## Notes about models and metrics

- The `models/` folders include model weights and architecture artifacts but may not include training history CSVs (e.g. `results.csv`). If training logs were not exported from the training environment, run `calculate_metrics.py` on a held-out labelled dataset to compute precise TP/FP/FN/TN numbers.

## Troubleshooting

- If you see `ImportError` for `mediapipe` or `torch`, ensure your Python version and platform match the wheel you installed (especially on macOS Apple Silicon).
- For performance on macOS with Apple Silicon, prefer the `mps`-enabled PyTorch wheel recommended on PyTorch.org.

## Contact / Next steps

If you'd like, I can:

- add a `requirements.txt` or `pyproject.toml` with pinned versions,
- add a small sample dataset and a quick test harness to run `calculate_metrics.py`, or
- update any of the demo CLI options for easier configuration.

— End of README

Make sure to run the following commands to install the dependencies:

`pip install mediapipe`

`pip install opencv-python`


Run the python file to start the program. Press the Escape key to exit the program.