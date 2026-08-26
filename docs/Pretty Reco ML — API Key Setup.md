# Pretty Reco ML — API Key Setup

## Objective

Secure communication between:

```text
pretty-crm-api
        ↓ HTTPS
ai.adler-backend.com
        ↓
pretty-reco-ml
```

Use a simple 256-bit service API key.

No JWT or OAuth is required for this service-to-service connection.

## 1. Generate API Key

Generate the key once on the server:

```bash
openssl rand -hex 32
```

This generates a random 256-bit key represented as 64 hexadecimal characters.

## 2. pretty-crm-api

Add to `.env`:

```env
RECO_API_URL=https://ai.adler-backend.com
RECO_API_KEY=<generated-key>
```

Node sends the key with every request:

```http
X-API-Key: <generated-key>
```

The API key must never be exposed to the frontend.

## 3. pretty-reco-ml

Add the same key to `.env`:

```env
RECO_API_KEY=<same-generated-key>
```

FastAPI reads the expected key from the environment and validates:

```http
X-API-Key
```

for protected endpoints.

Do not hardcode the key in Python.

## 4. Public Health Endpoint

The following endpoint may remain unauthenticated:

```http
GET /health
```

ML endpoints must require authentication:

```http
POST /embeddings/models
```

Future training and inference endpoints should also require authentication.

## 5. Security Rules

- Never commit API keys to Git.
- `.env` must be included in `.gitignore`.
- Never send the key to `pretty-app` or any browser/frontend application.
- Only backend services may know the key.
- All production communication must use HTTPS.
- Port `8000` remains bound to `127.0.0.1`.
- Nginx is the public entry point.
- Do not log the API key.
- Generate a new key if the existing key is exposed.

## 6. Key Rotation

To rotate the key:

```bash
openssl rand -hex 32
```

Update:

```text
pretty-reco-ml RECO_API_KEY
pretty-crm-api RECO_API_KEY
```

Then restart both affected services.

## 7. Future Extension

Currently one shared service key is sufficient:

```text
pretty-crm-api → pretty-reco-ml
```

If additional backend services later access `pretty-reco-ml`, move to one API key per calling service so individual clients can be identified and revoked independently.

## Acceptance Criteria

- A 256-bit API key is generated using `openssl rand -hex 32`.
- Both backend services receive the key through environment variables.
- Protected FastAPI endpoints reject requests without a valid key.
- `/health` remains available without authentication.
- No API key exists in source code or Git.
- No API key is exposed to frontend applications.
