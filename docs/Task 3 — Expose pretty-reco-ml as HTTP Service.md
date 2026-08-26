# Task 3 — Expose pretty-reco-ml as HTTP Service

## Objective

Expose `pretty-reco-ml` through a simple internal FastAPI service so Node applications can call the Python ML functionality over HTTP.

Production architecture:

```text
pretty-crm-api
    ↓ HTTPS
ai.adler-backend.com
    ↓
Nginx
    ↓
127.0.0.1:8000
    ↓
FastAPI / pretty-reco-ml
```

Both `pretty-crm-api` and `pretty-reco-ml` currently run on the same AWS EC2 server (**adler**).

## 1. Add FastAPI

Add required dependencies to `requirements.txt`:

```text
fastapi
uvicorn
```

Pin versions consistently with the existing requirements.

## 2. Application

Create:

```text
app.py
```

This is the HTTP entry point for `pretty-reco-ml`.

It must reuse the existing embedding implementation. Do not duplicate vision/embedding logic inside the API layer.

## 3. Endpoints

Implement:

```text
GET /health
```

Response:

```json
{"status":"ok"}
```

Implement:

```text
POST /embeddings/models
```

This endpoint calls the existing image embedding pipeline and returns its result.

Use the input/output contract already implemented by the embedding task.

## 4. Authentication

Require an API key header:

```text
X-API-Key
```

Read the expected key from environment:

```text
RECO_API_KEY
```

Do not hardcode secrets.

`/health` may remain unauthenticated.

## 5. Uvicorn

Production service must bind only to:

```text
127.0.0.1:8000
```

Example:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Do not expose port 8000 directly to the internet.

## 6. Production URL

The public service URL will be:

```text
https://ai.adler-backend.com
```

Nginx will proxy:

```text
ai.adler-backend.com
→
http://127.0.0.1:8000
```

Nginx/DNS/SSL configuration is outside this coding task unless explicitly requested.

## 7. Local Development

Locally the service should run as:

```text
http://127.0.0.1:8000
```

Test:

```text
GET /health
```

and verify the embedding endpoint using an existing PB model.

## 8. Architecture Rule

`pretty-reco-ml` must remain independent of MySQL.

```text
Node
= database access + orchestration

Python
= ML processing
```

Python receives data through HTTP and returns results through HTTP.

No database credentials or database access should be added to this repository.

## Acceptance

Task is complete when:

- FastAPI starts successfully.
- `/health` returns `{"status":"ok"}`.
- `/embeddings/models` uses the existing encoder.
- API-key authentication works.
- Uvicorn binds to `127.0.0.1:8000`.
- No database access has been introduced.
