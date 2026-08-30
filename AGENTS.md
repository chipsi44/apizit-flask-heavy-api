# Reference API repository guidelines

This repository is the Flask Heavy member of the APIZIT reference API suite.

- Keep Python 3.12 compatibility and exactly ten detected public routes.
- Preserve the five shared routes and five documented ML routes.
- Keep /health model-free, model loading lazy, and /slow at exactly 80 seconds.
- Every heavy dependency must be used by real application code.
- Normal CI must not download model weights; keep real-model tests opt-in with the model marker.
- Do not add cloud infrastructure, Dockerfiles, generated handlers, secrets, credentials, or APIZIT-internal imports.
- Update tests, README.md, and CONTRIBUTING.md whenever the HTTP contract changes.
- Run Ruff, pytest, an import check, and an APIZIT scan before publication.
- Use a codex/ branch, a Conventional Commit, and a reviewed pull request.
