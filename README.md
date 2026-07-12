# APIZIT Heavy ML API

[![CI](https://github.com/chipsi44/apizit-heavy-ml-api/actions/workflows/ci.yml/badge.svg)](https://github.com/chipsi44/apizit-heavy-ml-api/actions/workflows/ci.yml)

A standalone Flask API in the same spirit as a small data API, but with real
machine-learning dependencies and pretrained models. The repository contains
only application code, dependency files, tests, and documentation.

The dependency set is intentionally heavy: CPU PyTorch, torchvision,
Transformers, Sentence Transformers, scikit-learn, SciPy, NumPy, OpenCV, and
Pillow. The application actually uses every major dependency for inference and
image processing; none are included only to inflate the package.

## Features

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Endpoint index |
| `GET` | `/health` | Lightweight health check |
| `GET` | `/ready` | Load and verify both ML models |
| `GET` | `/info` | Non-sensitive Python and library versions |
| `POST` | `/text/embedding` | Real 384-value text embedding |
| `POST` | `/text/similarity` | Semantic cosine similarity |
| `POST` | `/image/analyze` | Five ImageNet predictions |
| `POST` | `/image/embedding` | Real 2048-value image embedding |

Models:

- [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- torchvision ResNet50 with `IMAGENET1K_V2` weights

Models are loaded lazily, cached under `models/`, and reused for later requests.
The first model-backed request therefore downloads the required weights.

## Install

Python 3.12 is required. Install the CPU-only PyTorch wheels before the other
dependencies so no CUDA packages are selected.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-torch.txt
python -m pip install -r requirements-dev.txt
```

## Run

```bash
python run.py
```

The API listens on `http://localhost:5000`. Debug mode is disabled by default.

## Examples

Text embedding:

```bash
curl -X POST http://localhost:5000/text/embedding \
  -H "Content-Type: application/json" \
  -d '{"text":"APIZIT deploys Python APIs."}'
```

Semantic similarity:

```bash
curl -X POST http://localhost:5000/text/similarity \
  -H "Content-Type: application/json" \
  -d '{"left":"Deploy a Flask API","right":"Publish a Python web service"}'
```

Image analysis and embedding:

```bash
curl -X POST http://localhost:5000/image/analyze -F "file=@sample.jpg"
curl -X POST http://localhost:5000/image/embedding -F "file=@sample.jpg"
```

JPEG, PNG, and WebP are supported. Images are decoded with Pillow, enhanced
with OpenCV CLAHE, transformed with torchvision, and executed through the real
ResNet50 model.

Errors use a homogeneous JSON format:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The field 'text' is required."
  }
}
```

## Tests

Fast API tests use controlled model stubs and never download weights:

```bash
pytest -q
ruff check .
ruff format --check .
```

The opt-in integration test downloads and executes both real models:

```bash
pytest -q -m model
```

## Limits

- Maximum request size: 4 MiB.
- Maximum text length: 5,000 characters.
- Maximum decoded image area: 20 million pixels.
- CPU inference only.
- Initial model downloads and cold loading can take time.
- Model caches and local `.env` files are ignored by Git.

This repository deliberately stops at the Flask application. Packaging,
deployment, and APIZIT integration belong outside this project.

## License

Project code is MIT licensed. Downloaded models and third-party packages retain
their own upstream licenses.
