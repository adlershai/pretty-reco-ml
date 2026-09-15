# API Contract — Query Image Embeddings

## Endpoint

```http
POST /embeddings/query
```

Header:

```http
Content-Type: application/json
X-API-Key: <RECO_API_KEY>
```

Production URL: `https://ai.adler-backend.com/embeddings/query` (Nginx → `127.0.0.1:8000`).

Callers send **image bytes** (base64). Python does not fetch WATI or S3 URLs.

## Request

```json
{
  "image_base64": "<base64 or data:image/jpeg;base64,...>"
}
```

## Successful Response

```json
{
  "embedding": [0.0124, -0.0831, 0.0417],
  "embedding_model": "google/siglip-base-patch16-224",
  "embedding_dimension": 768,
  "relevance": "footwear",
  "scores": {
    "footwear": 0.31,
    "irrelevant": 0.08
  }
}
```

The real `embedding` array is 768-d, L2-normalized, same encoder as `POST /embeddings/models`.

`relevance` is `footwear` or `irrelevant`. The gate is generic footwear vs junk (screenshot, selfie, document, meme). It is **not** a catalog match and is **not** ballet-flat specific.

When `irrelevant`, the caller should skip catalog nearest-neighbor. The embedding is still returned.

## HTTP Status

```text
200 - encoded
400 - invalid request/payload or INVALID_IMAGE
401 - missing or invalid API key
500 - encoder/service-level failure
```

## Responsibility Boundary

Python: decode bytes → SigLIP image vector → zero-shot text gate → return.

Node: download WATI media → upload S3 → `POST /match/image` for catalog identification. This query endpoint remains for callers that only need the vector + gate.
