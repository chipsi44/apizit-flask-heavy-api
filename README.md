# APIZIT Heavy ML API

[![CI](https://github.com/chipsi44/apizit-heavy-ml-api/actions/workflows/ci.yml/badge.svg)](https://github.com/chipsi44/apizit-heavy-ml-api/actions/workflows/ci.yml)

A deliberately heavy, multimodal Flask API used to validate APIZIT deployments
that need AWS Lambda container images instead of classic ZIP packages. The API
runs real CPU inference: it does not return mocked embeddings or predictions.

The repository contains the application and a Lambda-compatible Dockerfile.
It intentionally creates **no AWS resources**: no ECR repository, Lambda,
API Gateway, IAM role, CloudFormation stack, Terraform state, or AWS secret.
Infrastructure and image publication are left to APIZIT or to the operator.

## Why this project is intentionally heavy

The runtime combines PyTorch, torchvision, Transformers, Sentence Transformers,
scikit-learn, SciPy, NumPy, OpenCV, Pillow, and two real pretrained models. The
measured local image is approximately **1.961 GiB uncompressed**
(`2,105,728,291` bytes). The initial uncached build took **73.1 seconds** on the
development machine on 2026-07-12.

That footprint is far beyond Lambda's 250 MB unzipped ZIP limit but below the
10 GB container-image limit. See the official
[AWS Lambda quotas](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html)
and [Python container-image guide](https://docs.aws.amazon.com/lambda/latest/dg/python-image.html).

## Models

- Text: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
  producing normalized 384-value embeddings. The model is Apache-2.0 licensed.
- Image: torchvision ResNet50 with `IMAGENET1K_V2` weights, producing ImageNet
  predictions and a 2048-value `avgpool` embedding. See the
  [torchvision ResNet documentation](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html).

Model binaries are not committed to Git. `scripts.download_models` downloads
them during the Docker build and writes them under `/opt/models`. Runtime
loading then uses local paths with Hugging Face and torchvision offline modes.

## Endpoints

| Method | Path | Purpose | Loads models |
| --- | --- | --- | --- |
| `GET` | `/health` | Lightweight liveness response | No |
| `GET` | `/ready` | Load and verify both models | Yes |
| `GET` | `/info` | Allowlisted runtime and library versions | No weights |
| `POST` | `/text/embedding` | Generate a real text embedding | Text |
| `POST` | `/text/similarity` | Cosine similarity between generated embeddings | Text |
| `POST` | `/image/analyze` | Top five ImageNet predictions | Image |
| `POST` | `/image/embedding` | ResNet50 `avgpool` feature vector | Image |

### Health

```json
{
  "status": "ok",
  "service": "apizit-heavy-ml-api"
}
```

### Text embedding

```bash
curl -X POST http://localhost:5000/text/embedding \
  -H "Content-Type: application/json" \
  -d '{"text":"APIZIT deploys Python APIs without exposing AWS infrastructure."}'
```

The response contains the complete real vector:

```json
{
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "dimension": 384,
  "embedding": [0.0123, -0.0456]
}
```

### Text similarity

```bash
curl -X POST http://localhost:5000/text/similarity \
  -H "Content-Type: application/json" \
  -d '{"left":"Deploy a Flask API","right":"Publish a Python web service"}'
```

### Image analysis and embedding

```bash
curl -X POST http://localhost:5000/image/analyze -F "file=@sample.jpg"
curl -X POST http://localhost:5000/image/embedding -F "file=@sample.jpg"
```

JPEG, PNG, and WebP are accepted. The declared MIME type and the encoded image
format are both checked. Pillow validates the file, OpenCV applies CLAHE to the
lightness channel, and torchvision performs ResNet50's expected resize,
center-crop, tensor conversion, and normalization.

Errors always use the same shape and never expose a stack trace:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The field 'text' is required."
  }
}
```

## Local Python setup

Python 3.12 is required. Install the CPU-only PyTorch wheels first so pip never
resolves a CUDA build:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-torch.txt
python -m pip install -r requirements-dev.txt
python -m scripts.download_models
python run.py
```

The Flask development server listens on `http://localhost:5000`. Debug mode is
off by default and can be enabled only with `APP_DEBUG=true`.

Run the fast suite and real model integration separately:

```bash
pytest -q
pytest -q -m model
python -m scripts.smoke_models
ruff check .
ruff format --check .
```

Fast tests mock expensive inference but cover routing, validation, errors,
multipart base64 conversion, and thread-safe registry reuse. The `model` test
and smoke script load and execute the real downloaded models.

## Docker and Lambda runtime

Build the official Python 3.12 Lambda base image for x86-64:

```bash
docker build --platform linux/amd64 -t apizit-heavy-ml-api .
```

The Docker build performs both model downloads and a real inference smoke test.
The final runtime uses these coherent locations and offline settings:

```text
TEXT_MODEL_PATH=/opt/models/text/all-MiniLM-L6-v2
IMAGE_MODEL_PATH=/opt/models/torchvision/resnet50-imagenet1k-v2.pth
HF_HOME=/tmp/huggingface
TORCH_HOME=/tmp/torch
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
TORCH_MODEL_OFFLINE=1
```

Prove that inference does not need a network:

```bash
docker run --rm --network none --entrypoint python \
  apizit-heavy-ml-api -m scripts.smoke_models
```

Run the Lambda Runtime Interface Emulator locally:

```bash
docker run --rm -p 9000:8080 apizit-heavy-ml-api
```

Invoke the included API Gateway v2 events from another terminal:

```bash
curl -X POST http://localhost:9000/2015-03-31/functions/function/invocations \
  -H "Content-Type: application/json" \
  -d @examples/events/health.json

curl -X POST http://localhost:9000/2015-03-31/functions/function/invocations \
  -H "Content-Type: application/json" \
  -d @examples/events/text-embedding.json

curl -X POST http://localhost:9000/2015-03-31/functions/function/invocations \
  -H "Content-Type: application/json" \
  -d @examples/events/image-analyze.json
```

`lambda_handler.handler` uses `serverless-wsgi` to translate API Gateway v1/v2
and ALB proxy events into WSGI requests. Base64-encoded multipart bodies are
decoded before Flask parses the upload.

For a normal local web container with Gunicorn:

```bash
docker compose up --build
```

## Configuration and limits

Copy `.env.example` for local values; never commit a real `.env` file.

- Maximum text length: 5,000 characters. MiniLM truncates long inputs to its
  model token limit.
- Maximum request/upload size: 4 MiB.
- Maximum decoded image area: 20 million pixels.
- Device: CPU only; CUDA is neither installed nor selected.
- One process-local, thread-safe registry reuses model instances across warm
  Lambda invocations.

The conservative 4 MiB upload cap accounts for base64 expansion and event
metadata under Lambda's synchronous invocation payload limit. API Gateway and
Lambda limits still apply before Flask sees the request. For larger media,
uploading to object storage and passing a reference is a better architecture.

## Operational limitations

- Cold starts are materially longer than for a small Flask API.
- A memory allocation around 3 GB or higher is a practical starting point and
  must be measured under the target workload.
- Model loading and inference are CPU-bound.
- Lambda timeout, memory, concurrency, IAM, networking, ECR lifecycle, and API
  Gateway binary-media configuration belong to the external infrastructure.
- For larger models, GPU inference, long jobs, or high sustained throughput,
  ECS/Fargate, SageMaker, or another dedicated inference platform may be more
  appropriate than Lambda.

## Repository structure

```text
src/routes/              Flask HTTP contracts
src/services/            Model registry and inference services
scripts/download_models.py  Build-time model acquisition
scripts/smoke_models.py     Real offline inference smoke test
tests/                   Fast tests plus opt-in model integration
examples/events/         API Gateway v2 invocation fixtures
lambda_handler.py        Lambda-to-Flask WSGI adapter
Dockerfile               AWS Lambda Python 3.12 image
docker-compose.yml       Local Gunicorn mode
```

## AWS infrastructure boundary

This repository stops at a tested container image and valid Lambda handler. It
does not authenticate to AWS, create ECR resources, push images, create a
function, or expose an API Gateway. In an APIZIT flow, those operations are
expected to be automated by the platform; otherwise they remain the operator's
responsibility.

## License

Project code is MIT licensed. Third-party libraries and downloaded model
weights retain their own licenses and notices; review the linked model cards
and upstream package licenses for the intended deployment context.
