# AI News Digest (POC)

An internal tool that produces the **AI Weekly Digest**: new LLMs, embedding models, agent frameworks, research breakthroughs, Indic AI and AI regulation. The output is a two-page A4 report with the NSE masthead (logo, title, date line and colour bar from the Macro Review) followed by the Top 10 as short numbered points (category, title, What, Why it matters, source links, published date, trend score) and an "Also noted" list. It is a single self-contained HTML file; open it in a browser and use Print → Save as PDF to get the PDF.

Ranking is **computed in code** (a transparent trend score). The internal LLM only writes the two-sentence summaries for clusters that have already been selected. Every summary is checked against its source text, and every link comes from the fetched data.

## How data flows (airgap)

```
 internet machine                      │  airgapped server (aimldev5 / aimluat1)
                                       │
 fetch.py ──► raw_articles.jsonl ──────┼──► process.py ──► ranked_clusters.json
 (HF, arXiv, GitHub, HN, lab blogs)    │      embed (local bge-small) · cluster · score
                                       │    generate.py ──► top10.json + grounding_report.json
          copy the JSONL file across   │      internal LLM (vLLM/Ollama), grounding check, fallback
                                       │    render.py ──► digest.html
```

- **Fetch** is the only step that touches the network. It runs on a separate internet-connected machine with only `requirements-fetch.txt` installed. If one source fails, the run logs a warning and continues.
- **Process, generate and render** run fully offline. The embedding model is loaded from a local folder (Hugging Face offline mode is forced in code), and the LLM is an internal OpenAI-compatible endpoint.
- Stages exchange plain JSON/JSONL files (contracts in `src/digest/models.py`). There is no database, server or Docker. Each stage can be re-run and inspected on its own.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate

# internet machine
pip install -r requirements-fetch.txt && pip install -e .

# airgapped server (from the internal PyPI mirror)
pip install -r requirements.txt && pip install -e .
```

Copy the `bge-small-en-v1.5` model folder to the server out-of-band and point `EMBED_MODEL_PATH` at it.

The NSE logo lives at `assets/nse_logo.png` (PNG, SVG or JPG; path set by `render.logo_path`). The current file is a cleaned copy of `images.png`: the painted-in checkerboard was made transparent and the padding cropped. It is embedded into the HTML, so the report stays a single file. If the file is missing, the masthead shows the text "NSE" and the render step logs a warning.

### Environment variables

| Variable | Where | Purpose |
|---|---|---|
| `EMBED_MODEL_PATH` | server | Local folder holding `bge-small-en-v1.5` (required) |
| `LLM_BASE_URL` | server | Internal OpenAI-compatible endpoint, e.g. `http://llm.internal:8000/v1` |
| `LLM_MODEL` | server | Model name served by that endpoint |
| `LLM_API_KEY` | server | API key; can be a dummy for vLLM/Ollama |
| `LLM_REASONING_EFFORT` | server | Optional; set to `none` for reasoning models that otherwise spend all their tokens thinking |
| `HTTPS_PROXY` | fetch machine | Optional outbound proxy |
| `PYTHON` | server | Interpreter used by `run_all.sh` (default `python`) |

## Commands

```bash
# 1. internet machine: fetch the last 7 days
python scripts/fetch.py --days 7 --out data/raw/raw_articles.jsonl

# 2. airgapped server: one command for process -> generate -> render
bash scripts/run_all.sh                                   # uses data/raw/raw_articles.jsonl
bash scripts/run_all.sh path/to/snapshot.jsonl            # a specific snapshot
bash scripts/run_all.sh --dry-run tests/fixtures/sample_raw.jsonl --out-dir /tmp/demo   # offline, no LLM

# or the stages one at a time
python scripts/process.py  --in data/raw/raw_articles.jsonl --out data/processed/ranked_clusters.json
python scripts/generate.py --in data/processed/ranked_clusters.json --out data/output/top10.json [--dry-run]
python scripts/render.py   --in data/output/top10.json --out data/output/digest.html

pytest -q                                                 # offline tests
```

`run_all.sh` exits non-zero if any stage fails or fewer than 10 items are produced. At the end it prints the runtime and the path to `digest.html`.

**Review before forwarding:** open `data/output/digest.html` and check `data/output/grounding_report.json`. The report records, for each item, whether the summary was written by the LLM or extracted from the source, how many attempts it took, and any rejected names, numbers or URLs.

## How ranking works

1. **Embed** each item (title + first 512 chars) with bge-small-en-v1.5.
2. **Cluster** near-duplicates (cosine agglomerative clustering). Clusters that share a model or repo name such as `Qwen3-14B` or `owner/repo` are also merged, so one launch reported by HF, GitHub and HN becomes a single entry.
3. **Relevance:** similarity of each cluster to topic anchors ("new open-weight LLM", "embedding model", "Indic NLP", "AI regulation", …). Off-topic clusters are dropped.
4. **Trend score** = 0.40·coverage (number of distinct sources) + 0.30·engagement (likes/stars/points/upvotes) + 0.20·relevance + 0.10·recency (48 h half-life). Each component is normalised to 0–1.
5. **Diversity caps:** at most 3 clusters per topic anchor and at most 4 clusters made up only of papers among the 15 candidates.
6. **Summaries:** one LLM call per cluster, using only that cluster's text. The output is validated for word limits, for links that are a subset of the cluster's URLs, and for grounding (every model name and number must appear in the source text). A failed summary is retried up to 2 times, then the tool falls back to an extractive summary (marked `extract` in the HTML).

## Tuning

All settings are in `config/` and none are hard-coded:

- `config/sources.yaml`: feeds, GitHub repos, HN queries, trust weights. Set `enabled: false` with a `note:` to switch a source off.
- `config/settings.yaml`:
  - `cluster.cluster_threshold`: lower values give stricter, smaller clusters (0.15 by default; 0.25 merged unrelated papers).
  - `relevance.relevance_min` and `relevance.anchors`: what counts as on-topic.
  - `score.weights`, `score.max_per_anchor`, `score.max_paper_clusters`: what rises to the top.
  - `generate.*`: word limits, retries, response format (`json_schema` or `json_object`).
  - `render.*`: title, logo path, brand colours, category badge colours.
  - `generate.quick_hits` (0 hides "Also noted") and `generate.write_overview` (weekly lead paragraph; off in the points layout).

`process.py --threshold X --relevance-min Y` lets you try new values without editing the YAML. It prints the top 20 with each score component.

## Current limitations

- **Semantic errors can still get through.** The checks catch names, numbers, qualifiers ("under" vs. "average"), licences and repeated text, but not a sentence that misstates *who did what*. Example from 2026-10-12: "GPT-6 Astra, a frontier model trained with Ironclad". A human read before forwarding is required. A larger LLM on the server should reduce this.
- **Consumer feature rollouts rank low.** "GPT-6 and Intelligent UI for everyone" ranks around 85th because its text is about everyday use rather than AI topics.
- **arXiv RSS is often empty.** It only carries the latest announcement batch (empty around weekends), so Hugging Face daily papers is the main paper source.

- **Papers dominate the input.** About 900 of the roughly 1,000 items in a 7-day window are arXiv/HF papers. The paper cap keeps them in check, but single-source papers can still outrank smaller product launches.
- **Indic AI coverage is thin.** The AI4Bharat blog was unreachable on 2026-10-09 and is disabled. No other Indic-specific source is configured yet.
- **No regulation sources.** Regulation items only appear when HN or a lab blog covers them.
- **English only.** Hindi sources are out of scope.
- The grounding check confirms that names and numbers are *present* in the source. It cannot prove that a sentence *means* the same thing, so a human read is still needed before forwarding.
- Runs are manual (no scheduling), and email delivery is not built.

## Proposed next steps (for approval)

1. **Daily scheduled run:** a cron job on the fetch machine, an automatic file transfer to the server, and `run_all.sh` on a timer.
2. **Email dissemination:** send `digest.html` through the internal SMTP relay to a pilot distribution list.
3. **Coverage:** add Indic AI sources (AI4Bharat via a reachable mirror, Sarvam, BharatGen, IndiaAI) and regulation sources (MeitY, SEBI, EU AI Act updates).
4. **Ranking quality:** use the authority weight (already computed, currently 0), add a "launch vs. paper" balance, and collect reader feedback to tune the weights.
5. **History:** keep past digests so the same item is not repeated on consecutive days.
