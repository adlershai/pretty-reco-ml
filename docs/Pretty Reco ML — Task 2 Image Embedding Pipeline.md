# Pretty Reco ML — Task 2: Image Embedding Pipeline

## Objective

Implement the first production-ready ML capability in `pretty-reco-ml`:

```text
model packshot images
→ vision encoder
→ image embeddings
```

This task must remain completely independent from database access.

The service receives model/image data as input and returns embeddings as output.

---

# 1. Scope

Implement:

```text
embeddings/
    vision_encoder.py
    worker.py
```

Optionally add:

```text
tests/
    test_vision_encoder.py
```

Do not implement:

- MySQL access
- Weaviate
- recommendation ranking
- customer embeddings
- model training
- CSV import
- CRM orchestration

---

# 2. Vision Model Selection

Evaluate a suitable image embedding model for fashion/product similarity.

Primary candidates:

```text
SigLIP
CLIP
```

Preference:

```text
SigLIP
```

unless implementation constraints justify another choice.

Selection criteria:

- strong visual similarity performance
- works well on product/fashion imagery
- stable open-source implementation
- CPU-compatible for development
- GPU-compatible for future production
- deterministic embedding generation
- manageable model size
- commercially usable license

Document the selected model in:

```text
README.md
```

including:

```text
model name
source/library
embedding dimension
required Python packages
```

---

# 3. Image Input

Each Pretty Ballerinas model has a model code such as:

```text
40724_001
```

Expected image URLs:

```text
https://media.adler.co.il/app/products/40724_001.jpg
https://media.adler.co.il/app/products/40724_001_pers.jpg
https://media.adler.co.il/app/products/40724_001_side.jpg
```

Image types:

```text
main
pers
side
```

The worker must not construct URLs implicitly from DB data.

It should receive explicit image URLs in the input payload.

---

# 4. Input Contract

The first worker should accept JSON.

Example:

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

`model_id` is opaque to the ML service.

The ML service must return it unchanged.

---

# 5. Output Contract

For each successfully encoded image return:

```json
{
  "model_id": 123,
  "model": "40724_001",
  "image_type": "main",
  "embedding_model": "selected-model-name",
  "embedding_dimension": 768,
  "embedding": [],
  "image_hash": "sha256..."
}
```

Full response:

```json
{
  "results": [],
  "errors": []
}
```

Errors must not prevent successful images/models from being returned.

Example:

```json
{
  "results": [
    {
      "model_id": 123,
      "model": "40724_001",
      "image_type": "main",
      "embedding_model": "selected-model-name",
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

---

# 6. Image Hash

Calculate SHA-256 from the original downloaded image bytes.

The hash will later be used by `pretty-crm-api` to determine whether an image has changed.

Return:

```text
image_hash
```

for every successful image.

Do not persist hashes locally.

---

# 7. Vision Encoder

Implement a reusable class/function in:

```text
embeddings/vision_encoder.py
```

Conceptual interface:

```python
encoder = VisionEncoder()

embedding = encoder.encode(image)
```

Requirements:

- load model once per process
- do not reload the vision model for every image
- normalize preprocessing consistently
- return a plain Python list or NumPy array
- deterministic output for identical input
- expose:
  - embedding model name
  - embedding dimension

---

# 8. Embedding Normalization

Normalize output vectors to unit length unless the selected embedding model/library already guarantees equivalent normalized embeddings.

Use consistent normalization for all images.

Document the decision.

---

# 9. Worker

Implement:

```text
embeddings/worker.py
```

Initial execution model:

```bash
python -m embeddings.worker input.json
```

or:

```bash
python embeddings/worker.py input.json
```

Either is acceptable, but choose one consistent approach and document it.

The worker should:

```text
read JSON input
validate payload
load VisionEncoder once
download images
hash image bytes
decode image
generate embedding
collect results/errors
write JSON output
```

Output must go to stdout as clean JSON.

Logs must go to stderr.

This is important because Node will later consume stdout programmatically.

---

# 10. HTTP Download Requirements

For image downloads:

- set a reasonable timeout
- follow redirects
- verify successful HTTP status
- handle missing images gracefully
- support JPEG/PNG/WebP if encountered
- reject invalid/unreadable image payloads
- do not retry indefinitely

Reasonable retry policy:

```text
maximum 2 attempts
```

for transient network failures.

Do not retry a confirmed 404.

---

# 11. Partial Image Availability

Not every future model should be assumed to have all three views.

Valid scenarios:

```text
main only
main + pers
main + side
all three
```

The worker must process whatever images are available.

A missing secondary image must not fail the entire model.

---

# 12. Multiple Image Embeddings

Generate a separate embedding for:

```text
main
pers
side
```

Do not combine or average them in this task.

The database currently supports separate rows by:

```text
model_id
image_type
embedding_model
```

Combining multiple views into a model-level vector will be handled later during model feature construction/training.

---

# 13. Batch Processing

The worker must support multiple models in one request.

Example:

```text
30 models
× up to 3 images
= up to 90 image embeddings
```

Do not optimize aggressively yet, but avoid loading the vision model more than once.

If batching through the encoder is simple and reliable, implement modest batching.

Suggested initial batch size:

```text
8–16 images
```

Make batch size configurable.

---

# 14. CPU / GPU

The encoder must automatically choose:

```text
CUDA if available
otherwise CPU
```

Do not require a GPU.

Log selected device to stderr.

Example:

```text
vision device: cuda
```

or:

```text
vision device: cpu
```

Do not include device information in the JSON result unless needed for diagnostics.

---

# 15. Dependency Pinning

Update:

```text
requirements.txt
```

with exact versions required by this implementation.

Likely dependencies:

```text
torch
transformers
Pillow
requests
numpy
```

Only add dependencies actually used.

Do not install globally.

---

# 16. Model Cache

The Hugging Face / model cache must remain outside Git.

Do not commit downloaded model weights.

Ensure `.gitignore` excludes any repo-local model/cache folders if used.

The service must work after dependency installation and normal first-time model download.

---

# 17. Basic Test

Add a simple test using a known PB model image:

```text
40724_001
```

At minimum verify:

- image downloads successfully
- embedding is generated
- vector dimension matches expected dimension
- vector contains finite numeric values
- normalized vector magnitude is approximately 1 if normalization is enabled
- image hash is 64-character SHA-256
- repeated encoding produces equivalent output

Do not commit the downloaded test image unless necessary.

---

# 18. Manual Validation

Add a command/example to README that processes:

```text
40724_001
```

with:

```text
main
pers
side
```

Expected result:

```text
up to 3 successful embeddings
0 fatal process errors
clean JSON on stdout
```

---

# 19. Failure Behaviour

The process should exit:

```text
0
```

if the payload was valid and processing completed, even if individual images failed.

Exit non-zero only for fatal conditions such as:

- malformed input JSON
- unsupported payload structure
- encoder cannot initialize
- unrecoverable application failure

Per-image failures belong in:

```json
"errors": []
```

---

# 20. Code Quality

Keep responsibilities separate:

```text
vision_encoder.py
    model loading
    preprocessing
    vector generation

worker.py
    input/output
    downloads
    hashing
    orchestration
```

Do not put everything into one file.

Use type hints for public functions.

Use clear exceptions rather than silent failures.

---

# 21. Acceptance Criteria

Task is complete when:

- selected vision model is documented
- Python dependencies are pinned
- encoder loads successfully on local machine
- `40724_001` can be encoded
- main/pers/side are processed independently
- SHA-256 is returned for each image
- embeddings have consistent dimensions
- multiple models can be processed in one invocation
- successful results and per-image errors use the defined JSON contract
- stdout contains only machine-readable JSON
- no DB access exists anywhere in `pretty-reco-ml`
- no Weaviate integration exists yet

---

# Next Task

After this is complete:

```text
pretty-crm-api
→ sends newly imported model images to pretty-reco-ml
→ receives embeddings
→ writes them into reco_model_image_embeddings
```

That CRM/ML integration is a separate task.