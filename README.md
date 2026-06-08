# Factual Consistency Evaluation

Reproducible code for evaluating factual consistency in automated text
summarization across prompting strategies, language models, datasets, and
factuality metrics.

This repository combines and extends:

- [FActScore](https://github.com/shmsw25/FActScore)
- [LongDocFACTScore](https://github.com/jbshp/LongDocFACTScore)

## Paper

This code accompanies:

> Md Moinul Islam and Mourad Oussalah. "A Framework for Evaluating Factual
> Consistency in Automated Text Summarization with Large Language Models and
> Prompting Strategies." *Neural Networks*, 2026, 109160.
> <https://doi.org/10.1016/j.neunet.2026.109160>

GitHub can also generate citation metadata directly from
[`CITATION.cff`](CITATION.cff).

## Setup

Python 3.10 or newer is required. Create an isolated environment and install
the package:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e .
```

For an exact snapshot of the original experiment environment, use
`requirements.txt`. It includes CUDA 11.8 PyTorch builds and may require the
matching PyTorch package index and compatible NVIDIA drivers.

OpenAI-backed experiments read `OPENAI_API_KEY` from the environment. Copy
`.env.example` to `.env` and set the value locally. Never commit `.env` or API
keys.

## Data And Artifacts

Datasets, model checkpoints, generated JSONL files, retrieval databases, and
caches are intentionally excluded from Git. Place them in local directories
such as `data/`, `models/`, or `.cache/`; these paths are covered by
`.gitignore`.

Record the following for every reported run:

- Git commit SHA
- Python and dependency versions
- dataset name, source URL, version, and preprocessing command
- model identifier and revision
- prompt strategy and decoding parameters
- random seed, hardware, and CUDA version
- exact evaluation command and output artifact

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full run checklist.

## Prepare FActScore Inputs

Convert a generated-summary CSV into FActScore JSONL:

```bash
python -m factscore.preprocess_gendata \
  --input-path data/generated/longsciverify.csv \
  --output-path data/processed/longsciverify.jsonl \
  --topic-column article_id \
  --summary-column llama2chat7b_summary
```

Prepare LongSciVerify:

```bash
python -m factscore.preprocess_longsciverify \
  --arxiv-path data/raw/arxiv_test.json \
  --pubmed-path data/raw/pubmed_test.json \
  --output-dir data/processed/longsciverify \
  --register
```

Register any corpus containing `title` and `text` fields:

```bash
python -m factscore.preprocess \
  --name bbc_corpus \
  --data-path data/processed/bbc_corpus.jsonl
```

## Run FActScore

Download the required FActScore assets and, optionally, recover the local
Llama model:

```bash
huggingface-cli login
python -m factscore.download_data --llama_7B_HF_path meta-llama/Llama-2-7b-chat-hf
```

Run an evaluation:

```bash
python -m factscore.factscorer \
  --input_path data/processed/ChatGPT_bbc.jsonl \
  --model_name retrieval+llama+npm \
  --knowledge_source bbc_corpus \
  --n_samples 50
```

## LongDocFACTScore

The original LongDocFACTScore implementation and evaluation scripts are under
`LongDocFACTScore/`.

```bash
python -m pip install -r LongDocFACTScore/evaluation_scripts/requirements.txt
python LongDocFACTScore/evaluation_scripts/run_evaluation_BBC.py
```

Some evaluation scripts require third-party repositories or datasets. Record
their exact revisions alongside experiment outputs.

## Tests

```bash
python -m pytest
```

## Citation

```bibtex
@article{ISLAM2026109160,
  title = {A Framework for Evaluating Factual Consistency in Automated Text Summarization with Large Language Models and Prompting Strategies},
  journal = {Neural Networks},
  pages = {109160},
  year = {2026},
  issn = {0893-6080},
  doi = {10.1016/j.neunet.2026.109160},
  url = {https://www.sciencedirect.com/science/article/pii/S0893608026006210},
  author = {Md Moinul Islam and Mourad Oussalah},
  keywords = {factual consistency, text summarization, factuality metrics, large language model, prompting strategy, information extraction}
}
```

Please also cite the original FActScore and LongDocFACTScore papers when using
their respective implementations.
