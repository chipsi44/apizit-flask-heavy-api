# Contributing

Keep this project a real, heavy, standalone Flask API that remains comparable with the FastAPI and
APIZIT Linking Heavy references.

Before opening a pull request, preserve all ten routes, Python 3.12, pinned CPU dependencies, lazy
model loading, bounded input validation, and model-free normal CI. Run:

```bash
ruff check .
ruff format --check .
pytest -q
python -c "from app import app; assert app is not None"
```

Run `pytest -q -m model` only when real public model downloads are intended. The 80-second `/slow`
route is a deliberate timeout probe and its unit test must patch the wait.
