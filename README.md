# pretty-reco-ml

Python/ML service for the Pretty Ballerinas recommendation system.

Python: 3.12.10

This repository is independent from `pretty-crm-api`. It does **not** access MySQL. Communication with `pretty-crm-api` will go through a defined API/job contract.

## Service boundary

```text
pretty-crm-api
        |
        | model/image payload
        v
pretty-reco-ml
        |
        | embedding result
        v
pretty-crm-api
        |
        v
MySQL
```

`pretty-crm-api` owns:

- database access
- model CSV import
- model IDs
- writing embeddings to MySQL
- orchestration

`pretty-reco-ml` owns:

- image loading
- image preprocessing
- vision encoding
- embedding generation
- later model training
- later evaluation
- later inference

## Local setup

Use the same **major.minor.patch** Python version locally and in production: **3.12.10**.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux:

```bash
source .venv/bin/activate
```

Confirm the interpreter is the venv copy:

```bash
python --version
where python
```

Linux:

```bash
which python
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The first run downloads SigLIP weights into the local Hugging Face cache (outside Git).

Smoke test:

```bash
python -m embeddings.worker --version
```

## HTTP service

Local / production process (loopback only):

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --env-file .env
```

Do not publish port 8000. On **adler**, Nginx should proxy `https://ai.adler-backend.com` to `http://127.0.0.1:8000`.

| Endpoint | Auth | Purpose |
| --- | --- | --- |
| `GET /health` | none | `{"status":"ok"}` |
| `POST /embeddings/models` | header `X-API-Key` | same JSON contract as the CLI worker |

Set `RECO_API_KEY` in the environment (see `.env.example`). Never commit the key.

```bash
curl http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/embeddings/models ^
  -H "Content-Type: application/json" ^
  -H "X-API-Key: %RECO_API_KEY%" ^
  --data @examples/40724_001.json
```

Nginx/DNS/SSL for `ai.adler-backend.com` is a separate ops step.

## Image embeddings

Packshot URLs in the JSON payload are encoded independently per view (`main`, `pers`, `side`). Views are not averaged. The worker does not invent URLs; callers pass explicit addresses. App product images use:

```text
https://media.adler.co.il/app/products/{model}.jpg
https://media.adler.co.il/app/products/{model}_pers.jpg
https://media.adler.co.il/app/products/{model}_side.jpg
```

| Field | Value |
| --- | --- |
| Model name | `google/siglip-base-patch16-224` (SigLIP) |
| Source / library | Hugging Face `transformers` (`AutoModel.get_image_features`, `SiglipImageProcessorPil`) |
| Embedding dimension | 768 |
| License | Apache 2.0 |
| Device | CUDA if available, otherwise CPU |
| Python packages | `torch`, `transformers`, `Pillow`, `requests`, `numpy` |

**Why SigLIP:** it is the preferred candidate in the task brief, has a stable open-source checkpoint, is commercially usable, runs on CPU for development, and produces a 768-d vector sized for product similarity without pulling in a giant So400m checkpoint.

**Normalization:** SigLIP `get_image_features` does not return unit-length vectors. `VisionEncoder` L2-normalizes every embedding so cosine similarity equals a dot product. All images use the same preprocessing and normalization.

Batch size defaults to 12 and is configurable with `--batch-size` or `EMBEDDING_BATCH_SIZE`.

### Manual validation

From the repo root, with `.venv` active:

```bash
python -m embeddings.worker examples/40724_001.json
```

Expected: three embeddings for `40724_001` (`main`, `pers`, `side`), exit code 0, and a single JSON object on stdout. Logs such as `vision device: cpu` go to stderr.

If a view is missing on the CDN (CloudFront often returns HTTP 403 for absent objects), that view is recorded as `IMAGE_NOT_FOUND` and other views still succeed.

Tests (downloads `40724_001` at runtime; the image is not committed):

```bash
python -m pytest
```

## Production

Deploy with an isolated environment. Do **not** replace the server's system Python, change Python used by other applications, modify global packages, or reuse existing system-level virtualenvs.

Example layout on **adler** (`3.71.237.152`):

```text
/home/ubuntu/pretty-reco-ml/
    .venv/
```

This sits next to `pretty-crm-api`, `pretty-db-connect`, and `rrai-backend`. Do not deploy this repo to **lunara** (`3.22.1.120`).

System Python on adler remains 3.8.10. This app uses an isolated 3.12.10 interpreter (via uv) and `.venv`. On CPU hosts, install `torch` from the PyTorch CPU index so pip does not pull CUDA wheels.

## Current scope

Image embeddings and the HTTP service are implemented. Recommendation training, evaluation, inference, Weaviate, and database access remain out of scope.
