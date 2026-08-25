# Pretty Reco ML — Initial Setup

## Objective

Create a new standalone repository:

```text
pretty-reco-ml
```

This repository contains all Python/ML functionality for the Pretty Ballerinas recommendation system.

It must remain independent from `pretty-crm-api`.

`pretty-reco-ml` must **not access MySQL directly**.

Communication with `pretty-crm-api` will later happen through a defined API/job contract.

---

# 1. Python Version

Before creating any Python environment or installing dependencies, determine the Python version that will be used for `pretty-reco-ml` in production.

The same **major.minor.patch** Python version must be used locally and in production.

Example:

```text
3.12.10
```

Do not assume the local Python version is correct.

## Local check

Run:

```bash
python --version
python3 --version
```

Also check for:

```bash
py --version
pyenv versions
```

where applicable.

---

# 2. Production Python

The Python version requirement applies **only to the `pretty-reco-ml` application**.

Do NOT:

- replace the production server's system Python
- change Python used by other applications
- modify global Python packages
- modify existing system-level virtual environments

`pretty-reco-ml` must have its own isolated Python environment.

Example:

```text
/opt/pretty-reco-ml/
    .venv/
```

or equivalent deployment directory.

If the required Python version is not installed on production, install it alongside the existing system Python rather than replacing the system version.

---

# 3. Version Pinning

Once the production-compatible Python version is confirmed, create:

```text
.python-version
```

containing the exact version:

```text
3.12.x
```

Use the actual confirmed patch version rather than `x`.

Also document the version in:

```text
README.md
```

Example:

```text
Python: 3.12.10
```

---

# 4. Virtual Environment

Create a project-local virtual environment:

```bash
python -m venv .venv
```

Activate it before installing dependencies.

Windows:

```bash
.venv\Scripts\activate
```

Linux:

```bash
source .venv/bin/activate
```

Verify:

```bash
python --version
which python
```

On Windows:

```bash
where python
```

The returned Python executable must belong to `.venv`.

---

# 5. Initial Repository Structure

Create:

```text
pretty-reco-ml/
│
├── .python-version
├── .gitignore
├── README.md
├── requirements.txt
│
├── embeddings/
│   ├── __init__.py
│   ├── vision_encoder.py
│   └── worker.py
│
├── training/
│   └── __init__.py
│
├── evaluation/
│   └── __init__.py
│
└── inference/
    └── __init__.py
```

Do not build recommendation logic yet.

The first implementation target is image vectorization.

---

# 6. Image Embedding Responsibility

The first ML capability will convert model packshots into image embeddings.

A model identifier looks like:

```text
40724_001
```

Its images are:

```text
https://media.adler.co.il/app/products/40724_001.jpg
https://media.adler.co.il/app/products/40724_001_pers.jpg
https://media.adler.co.il/app/products/40724_001_side.jpg
```

The ML service will eventually receive a payload containing:

```text
model_id
model
image URLs
```

and return embeddings.

It must not query the CRM database itself.

---

# 7. Service Boundary

Architecture:

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

---

# 8. Dependency Management

Initially use:

```text
requirements.txt
```

Do not install Python dependencies globally.

All packages must be installed into `.venv`.

Example:

```bash
pip install -r requirements.txt
```

Once dependencies are selected, pin their versions.

Do not add ML libraries until the vision encoder implementation requires them.

---

# 9. Git Ignore

At minimum ignore:

```text
.venv/
__pycache__/
*.pyc
.env
.env.*
.pytest_cache/
.DS_Store
```

Do not commit model caches, downloaded images, or generated embeddings unless explicitly required later.

---

# 10. First Milestone

Stop after the following are complete:

1. Repository structure exists.
2. Exact Python version is confirmed.
3. Local `.venv` uses that version.
4. `.python-version` is committed.
5. Production deployment approach preserves the same Python version without changing system Python.
6. `python embeddings/worker.py --version` or an equivalent minimal test command runs successfully.

Do not implement database access.

Do not implement recommendation training yet.

Do not implement Weaviate yet.

The next task after this milestone will be selecting and implementing the image embedding model.