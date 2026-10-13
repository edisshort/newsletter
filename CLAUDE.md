# AI News Digest — Claude Code instructions

## What this project is

An internal NSE tool that produces a daily **Top 10 AI initiatives** digest (new open-source LLMs, embedding models, agent frameworks, research breakthroughs, Indic AI, AI regulation). Pipeline: **Fetch → Process → Generate → (Disseminate, later)**. Current goal: a POC demo within 1–2 days for a manager go-ahead. Optimise for a working, explainable end-to-end run, not for scale.

Build order and acceptance checks are in `PLAN.md`. Work one phase at a time and **stop for review at the end of each phase**.

## Hard constraints (never violate)

- **Airgapped target.** Processing, generation and rendering run on internal Linux servers (aimldev5 / aimluat1) with no internet. Only fetch ever touches the network.
- **Packages come from an internal PyPI mirror (Artifactory).** Use only pure-pip packages listed in `requirements*.txt`. No packages that download things at runtime (no `playwright install`, no `nltk.download`, no HF Hub auto-download). **Ask before adding any new dependency.**
- **Models are local paths.** Load the embedding model from `EMBED_MODEL_PATH`; set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` in code before importing sentence-transformers.
- **LLM is an internal OpenAI-compatible endpoint** (vLLM/Ollama). Configure via env: `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` (may be a dummy). Never call external APIs.
- **No databases, no Docker, no servers for the POC.** Files only: JSONL/JSON in `data/`.
- The fetch step must run standalone on a separate internet-connected machine with only `requirements-fetch.txt` installed. It must not import anything from process/generate/render.
- Python 3.10+, standard library first. Type hints everywhere. `argparse` for CLIs.

## Content rules

- Every digest item must carry at least one source URL from the input data. **Never invent URLs.**
- Ranking is computed in code (trend score), not by the LLM. The LLM only writes summaries for already-selected clusters.
- The LLM sees only the text of its own cluster. Every model/product name and number in a summary must appear in that cluster's source text (grounding check in `generate/validate.py`).
- If generation fails validation after retries, fall back to an extractive summary (first sentences of the top source). The pipeline must always produce 10 items when ≥10 clusters exist.

## Code conventions

- Package lives in `src/digest/`; thin CLI entry points in `scripts/`.
- All tunables (sources, weights, anchors, thresholds, paths) live in `config/*.yaml`. No magic numbers in code.
- Data contracts are Pydantic models in `src/digest/models.py`. Every stage reads and writes those models, so stages can be run and tested independently.
- Logging via `logging` (INFO by default), never `print` in library code.
- One source failing must not stop the fetch run: log a warning and continue.
- Tests in `tests/` with pytest. Network tests are marked `@pytest.mark.network` and skipped by default. All other tests run offline against `tests/fixtures/`.

## Commands

```bash
pip install -r requirements-fetch.txt          # internet machine only
pip install -r requirements.txt                # airgapped server
python scripts/fetch.py    --days 7 --out data/raw/raw_articles.jsonl
python scripts/process.py  --in data/raw/raw_articles.jsonl --out data/processed/ranked_clusters.json
python scripts/generate.py --in data/processed/ranked_clusters.json --out data/output/top10.json
python scripts/render.py   --in data/output/top10.json --out data/output/digest.html
bash   scripts/run_all.sh                      # process → generate → render
pytest -q                                      # offline tests
```

## Out of scope for now (do not build)

Email sending, scheduling/cron, Postgres/pgvector, Docker, dashboards, user auth, Hindi sources.
