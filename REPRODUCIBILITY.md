# Reproducibility Guide

Use this checklist for each experiment that contributes to a table or figure.

## Environment

Record:

```bash
git rev-parse HEAD
python --version
python -m pip freeze > artifacts/<run-id>/requirements.txt
```

Also record the operating system, GPU model, CUDA version, and available GPU
memory. The root `requirements.txt` is the original pinned experiment
environment; `pyproject.toml` defines a portable development installation.

## Inputs

Keep datasets and model outputs outside Git. For every input, record:

- source URL and access date
- dataset split and version
- license or usage restrictions
- preprocessing command
- SHA-256 checksum of the processed file

Example:

```bash
python -c "import hashlib, pathlib; p=pathlib.Path('data/processed/input.jsonl'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

## Models And Prompts

Record the full model identifier and revision, prompt strategy, exact prompt
template, generation parameters, random seed, and provider API version. Hosted
model aliases can change over time, so preserve raw responses and request
metadata where permitted.

## Evaluation

Store each run in a unique artifact directory containing:

- the exact command
- stdout and stderr logs
- metric outputs
- dependency snapshot
- Git commit SHA
- input checksums

Run the test suite before evaluation:

```bash
python -m pytest
```

Do not commit API keys, licensed datasets, model weights, caches, or private
outputs. The repository `.gitignore` excludes the common local artifact paths.
