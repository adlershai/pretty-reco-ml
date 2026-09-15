# POST /embeddings/models

```http
POST https://ai.adler-backend.com/embeddings/models
Content-Type: application/json
X-API-Key: <RECO_API_KEY>
```

## Request

```json
{
  "models": [
    {
      "model_id": 123,
      "model": "40724_001",
      "images": {
        "main": "https://media.adler.co.il/app/products/40724_001.jpg",
        "pers": "https://media.adler.co.il/app/products/40724_001_pers.jpg",
        "side": "https://media.adler.co.il/app/products/40724_001_side.jpg"
      }
    }
  ]
}
```

| Field | Required | Notes |
| --- | --- | --- |
| `models` | yes | Array of one or more models |
| `model_id` | yes | Opaque id; returned unchanged |
| `model` | yes | Model code string; returned unchanged |
| `images.main` | if that view exists | Full URL |
| `images.pers` | if that view exists | Full URL |
| `images.side` | if that view exists | Full URL |

Omit image keys that have no file. Do not send empty strings. URLs must be explicit; the API does not invent them.

## Success (HTTP 200)

Processed. Individual image failures belong in `errors`, not in the HTTP status.

```json
{
  "results": [
    {
      "model_id": 123,
      "model": "40724_001",
      "image_type": "main",
      "embedding_model": "google/siglip-base-patch16-224",
      "embedding_dimension": 768,
      "embedding": [0.0124, -0.0831, 0.0417],
      "image_hash": "a64-character-sha256-hash"
    }
  ],
  "errors": []
}
```

`embedding` is the full 768-float vector. `image_hash` is SHA-256 of the original image bytes (64 hex chars). `image_type` is `main`, `pers`, or `side`.

## Partial failure (still HTTP 200)

```json
{
  "results": [
    {
      "model_id": 123,
      "model": "40724_001",
      "image_type": "main",
      "embedding_model": "google/siglip-base-patch16-224",
      "embedding_dimension": 768,
      "embedding": [],
      "image_hash": "..."
    }
  ],
  "errors": [
    {
      "model_id": 123,
      "model": "40724_001",
      "image_type": "side",
      "error": "IMAGE_NOT_FOUND"
    }
  ]
}
```

## HTTP status

| Code | Meaning |
| --- | --- |
| 200 | Request processed; see `errors[]` for per-image failures |
| 400 | Invalid payload |
| 401 | Missing or invalid `X-API-Key` |
| 500 | Encoder / service failure |

`GET https://ai.adler-backend.com/health` is unauthenticated and returns `{"status":"ok"}`.
