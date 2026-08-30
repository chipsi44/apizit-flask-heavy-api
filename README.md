# APIZIT Flask Heavy API

A standalone Flask API with real CPU machine-learning and image-processing
dependencies. It is intentionally expensive to install and build so APIZIT can
exercise large-package scanning, Expanded launches, cold starts, and predictable
long responses.

The application uses CPU PyTorch, torchvision, Transformers, Sentence
Transformers, scikit-learn, SciPy, NumPy, OpenCV, and Pillow. Models load lazily
and are cached outside the repository.

## Routes

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Immediate model-free health response |
| `GET` | `/info` | Framework, profile, runtime, and library metadata |
| `POST` | `/echo` | JSON request and response |
| `GET` | `/items/<item_id>?include_details=true` | Path and query parameters |
| `GET` | `/slow` | Intentional 80-second response |
| `GET` | `/ready` | Load and verify both models |
| `POST` | `/text/embedding` | Real text embedding |
| `POST` | `/text/similarity` | Semantic cosine similarity |
| `POST` | `/image/analyze` | Five ImageNet predictions |
| `POST` | `/image/embedding` | Real image embedding |

`/slow` is a timeout probe. Never configure it as a health check. The first
model-backed request may download public model weights and take substantially
longer than later requests.

## Run locally

Python 3.12 is required. Install the CPU-only PyTorch wheels first.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-torch.txt
python -m pip install -r requirements-dev.txt
python run.py
```

The API is available at `http://127.0.0.1:5000`.

```bash
curl http://127.0.0.1:5000/health
curl "http://127.0.0.1:5000/items/7?include_details=true"
curl -X POST http://127.0.0.1:5000/echo -H "Content-Type: application/json" -d '{"message":"hello","count":2}'
curl -X POST http://127.0.0.1:5000/text/similarity -H "Content-Type: application/json" -d '{"left":"Launch an API","right":"Publish a Python service"}'
curl -X POST http://127.0.0.1:5000/image/analyze -F "file=@sample.jpg"
```

## Verify

Normal tests use controlled model services and download no weights.

```bash
ruff check .
ruff format --check .
pytest -q
python -c "from app import app; assert app is not None"
```

The opt-in integration test loads both real public models:

```bash
pytest -q -m model
```

Inputs are bounded to 4 MiB uploads, 5,000-character text, and 20-million-pixel
decoded images. This is a controlled beta reference project, not a production
application.
