# API Contract — Visual Image Match

## Endpoints

```http
POST /match/image
POST /match/image/decide
```

Header:

```http
Content-Type: application/json
X-API-Key: <RECO_API_KEY>
```

Production URL: `https://ai.adler-backend.com`. Callers send **image bytes** (base64). Python does not call OpenAI and does not fetch WATI or S3 URLs. The 768-d query vector is not returned.

pretty-crm-api owns `OPENAI_API_KEY` and `lib/openai.createResponse`. ML owns isolation, SigLIP retrieval, the verifier prompt/schema, and MATCH vs UNCERTAIN.

## Request (`POST /match/image`)

```json
{
  "image_base64": "<base64 or data:image/jpeg;base64,...>",
  "top": 10,
  "verify_top": 10
}
```

`top` is SigLIP retrieval size (default 10, max 50). `verify_top` is how many of those candidates CRM should send to the vision verifier (default 10, or env `MATCH_VERIFY_TOP`). Retrieval is raised to at least `verify_top`.

## Retrieval response

Footwear:

```json
{
  "status": "needs_verification",
  "match": null,
  "candidates": [{"model": "52792_006", "score": 0.85, "best_image_type": "side", "model_id": 2937}],
  "preprocessing": {
    "shoe_isolated": true,
    "crop": [8, 373, 937, 1325],
    "reason": "studio_panel",
    "image_size": [945, 2048],
    "crop_jpeg_base64": "<isolated crop jpeg>"
  },
  "verifier": {
    "requestType": "watiImageIdentity",
    "verify_top": 10,
    "jsonSchema": {},
    "prompt": [],
    "images": [{"slot": "customer", "kind": "crop"}]
  },
  "relevance": "footwear",
  "scores": {"footwear": 0.31, "irrelevant": 0.08},
  "embedding_model": "google/siglip-base-patch16-224"
}
```

`match` is null until `/match/image/decide`. SigLIP cosine on `candidates` is retrieval similarity, not identity confidence.

When `relevance` is `irrelevant`, `status` is `irrelevant`, `verifier` is null, and `candidates` is empty.

## Decide (`POST /match/image/decide`)

CRM fills `verifier.images` (customer crop + packshots), calls OpenAI, then posts the raw output:

```json
{
  "candidates": [{"model": "51604_A", "score": 0.78, "best_image_type": "pers", "model_id": 1}],
  "openai_output": {"candidates": []}
}
```

```json
{
  "status": "match",
  "match": {"model": "51604_A", "score": 0.78, "best_image_type": "pers", "model_id": 1},
  "confidence": 0.94,
  "reason": "unique_same_model",
  "verification": [],
  "candidates": []
}
```

Or `"status": "uncertain"` with `"match": null`. Confidence is diagnostic, not a calibrated probability.

Acceptance: exactly one verifier `same` without an exclusionary shape/pattern contradiction. Two sames, zero sames, or a failed OpenAI payload → `uncertain`.

## HTTP Status

```text
200 - ranked or decided
400 - invalid request/payload or INVALID_IMAGE
401 - missing or invalid API key
503 - catalog is not loaded
500 - encoder/service-level failure
```

## Responsibility Boundary

Python: isolate → SigLIP Top-N → verifier prompt/schema → MATCH/UNCERTAIN policy.

Node: WATI download → S3 → `/match/image` → OpenAI Responses (`watiImageIdentity`) → `/match/image/decide` → persist `matched` / `uncertain` / `irrelevant`.

Do not use SigLIP cosine as a business identity cutoff. Two-tower `like_score` is unchanged.
