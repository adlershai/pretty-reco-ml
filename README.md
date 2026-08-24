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

Install dependencies (none pinned yet):

```bash
pip install -r requirements.txt
```

Smoke test:

```bash
python embeddings/worker.py --version
```

## Production

Deploy with an isolated environment. Do **not** replace the server's system Python, change Python used by other applications, modify global packages, or reuse existing system-level virtualenvs.

Example layout:

```text
/opt/pretty-reco-ml/
    .venv/
```

If 3.12.10 is not installed on the host, install it **alongside** system Python, then create `/opt/pretty-reco-ml/.venv` with that interpreter.

## Current scope

The first ML capability is converting model packshots into image embeddings. Recommendation training, evaluation, inference, Weaviate, and database access are out of scope until that encoder is in place.

Selecting and implementing the image embedding model is the next task.
