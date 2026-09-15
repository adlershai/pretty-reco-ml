# API Contract — Visual Image Match

## Endpoint

```http
POST /match/image
```

Header:

```http
Content-Type: application/json
X-API-Key: <RECO_API_KEY>
```

Production URL: `https://ai.adler-backend.com/match/image`.

Callers send **image bytes** (base64). Python does not fetch WATI or S3 URLs. The 768-d query vector is not returned.

## Request

```json
{
  "image_base64": "<base64 or data:image/jpeg;base64,...>",
  "top": 10
}
```

`top` is optional (default 10, max 50): number of catalog models to return.

## Successful Response

```json
{
  "match": {
    "model": "52792_006",
    "score": 0.91,
    "best_image_type": "side",
    "model_id": 2937
  },
  "candidates": [
    {
      "model": "52792_006",
      "score": 0.91,
      "best_image_type": "side",
      "model_id": 2937
    }
  ],
  "preprocessing": {
    "shoe_isolated": true,
    "crop": [8, 373, 937, 1325],
    "reason": "studio_panel",
    "image_size": [945, 2048]
  },
  "relevance": "footwear",
  "scores": {
    "footwear": 0.31,
    "irrelevant": 0.08
  },
  "embedding_model": "google/siglip-base-patch16-224"
}
```

`crop` is `[left, top, right, bottom]` in pixel coordinates on the original image.

Matching uses `google/siglip-base-patch16-224` (768-d, L2-normalized). Each catalog view (`main`, `pers`, `side`) is an independent candidate. Model score is `MAX(view similarity)`. The 64-d recommendation Model Tower is not used.

When `relevance` is `irrelevant`, `match` is `null` and `candidates` is empty. The crop diagnostics are still returned.

## HTTP Status

```text
200 - ranked
400 - invalid request/payload or INVALID_IMAGE
401 - missing or invalid API key
503 - catalog is not loaded
500 - encoder/service-level failure
```

## Responsibility Boundary

Python: decode bytes → isolate shoe region → SigLIP → per-view catalog search → candidates + scores + crop diagnostics.

Node: download WATI media → upload S3 → call this endpoint → apply business threshold → write `wati_image_matches`.

## Stage 3 reranker

Not implemented. A 3-case eval (`python -m evaluation.match_benchmark`) compared full-frame SigLIP to isolated-shoe SigLIP:

| split | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| original | 0.67 | 0.67 | 1.00 | 1.00 |
| isolated | 1.00 | 1.00 | 1.00 | 1.00 |

`845.jpg` (WATI screenshot) went from rank 4 (`52160_001`) to rank 1 (`52792_006`) after automatic studio-panel crop `[8, 373, 937, 1325]`. Packshot self-matches stayed rank 1 with the full frame. A fine-grained identity reranker is not justified until a larger labeled set shows residual errors.
