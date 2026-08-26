# API Contract — Model Image Embeddings

## Endpoint

```http
POST /embeddings/models
```

Header:

```http
Content-Type: application/json
X-API-Key: <RECO_API_KEY>
```

Production URL: `https://ai.adler-backend.com/embeddings/models` (Nginx → `127.0.0.1:8000`).

## Request

Node provides the model identity and explicit URLs. Python does not construct URLs.

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
    },
    {
      "model_id": 124,
      "model": "40725_002",
      "images": {
        "main": "https://media.adler.co.il/app/products/40725_002.jpg",
        "pers": "https://media.adler.co.il/app/products/40725_002_pers.jpg",
        "side": "https://media.adler.co.il/app/products/40725_002_side.jpg"
      }
    }
  ]
}
```

`model_id` and `model` are identifiers supplied by Node. Python must return them unchanged.

A request may contain one or multiple models. Views may be omitted; missing secondary images are not fatal.

## Successful Response

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

The real `embedding` array must contain the complete 768-d vector. Short arrays in examples are illustrative only.

## Partial Failure

Failure of one image must not fail the complete request.

```json
{
  "results": [],
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

HTTP status remains 200 when the payload was valid and processing completed.

## HTTP Status

```text
200 - request processed; individual image failures may exist in errors[]
400 - invalid request/payload
401 - missing or invalid API key
500 - encoder/service-level failure
```

A missing or invalid individual image is **not** an HTTP 500.

## Responsibility Boundary

Python:

```text
receive URLs
→ download images
→ calculate image hashes
→ generate embeddings
→ return results
```

Node:

```text
read models
→ construct/request image URLs
→ call reco API
→ receive results
→ write/update reco_model_image_embeddings
```

Python must not read or write the Pretty Ballerinas database.
