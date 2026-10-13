# Build plan — AI News Digest POC

Read CLAUDE.md first. Do the phases in order. At the end of each phase, run its acceptance checks, summarise what was built, and stop for review before starting the next phase.

## Phase 0 — Scaffold (≈30 min)

Create the layout (config/, src/digest/{fetch,process,generate,render}, scripts/, data/{raw,processed,output}, tests/).

Data contracts in `src/digest/models.py`: `RawItem`, `Cluster`, `Initiative`, `Digest`.

**Acceptance:** `pip install -r requirements.txt` works; `python -c "import digest"` works; `pytest -q` runs (even with zero tests).

## Phase 1 — Fetch (≈3 h)

Each fetcher is `fetch(cfg, since: datetime) -> list[RawItem]`. Sources in `config/sources.yaml` (verify each URL with a real request; set `enabled: false` and log the reason if it fails):

| Source | Endpoint | Maps to |
|---|---|---|
| HF trending models | `https://huggingface.co/api/models?sort=trendingScore&limit=100` (fall back to `sort=likes`) | type=model, metrics: likes, downloads |
| HF daily papers | `https://huggingface.co/api/daily_papers` | type=paper, metrics: upvotes |
| arXiv | `https://rss.arxiv.org/rss/cs.CL`, `cs.LG`, `cs.AI` | type=paper |
| GitHub releases | `https://github.com/{owner}/{repo}/releases.atom` for repos listed in YAML | type=release |
| Hacker News | Algolia `search_by_date`, `tags=story`, `numericFilters=created_at_i>{since},points>30`; queries: LLM, open source model, embedding, AI agent | type=discussion, metrics: points, comments |
| Lab / org blogs | RSS feeds in YAML (HF blog, Google DeepMind, OpenAI news, NVIDIA developer, AI4Bharat) | type=news |

Rules:
- httpx, timeout 20 s, 2 retries with backoff, clear User-Agent, optional proxy from `HTTPS_PROXY`.
- `extract.py`: for news items with short text, fetch the page and run `trafilatura.extract`; cap text at 4,000 chars.
- Dedup by id (sha1 of URL with query string and trailing slash stripped).
- `scripts/fetch.py --days 7 --out ...` writes JSONL and prints per-source counts.
- Build `tests/fixtures/sample_raw.jsonl` (~60 realistic items across all sources, including 2–3 deliberate near-duplicates of the same launch).

**Acceptance:** a live run returns ≥300 items with ≥5 sources non-empty; one source failing (bad URL) does not stop the run; output validates against `RawItem`.

## Phase 2 — Process (≈3 h)

1. **Embed:** bge-small-en-v1.5 from `EMBED_MODEL_PATH`; input title + first 512 chars of text; normalised; batch 64; CPU.
2. **Cluster:** `AgglomerativeClustering(metric="cosine", linkage="average", distance_threshold=settings.cluster_threshold)` (start 0.25). Also merge clusters sharing an extracted entity (regex for model/repo names like `Qwen3-14B`, `owner/repo`).
3. **Relevance:** max cosine similarity of cluster centroid to topic anchors in settings. Drop clusters below `relevance_min` (start 0.45).
4. **Score**, every component min-max normalised to 0–1:
   `score = 0.40·coverage + 0.30·engagement + 0.20·relevance + 0.10·recency` (recency: exp decay, half-life 48 h, from newest item). `authority` computed and stored, weight 0 for now. Weights from settings.
5. **Diversity:** at most `max_per_anchor` (3) per `best_anchor` until `top_n_candidates` (15).
6. Write `ranked_clusters.json` and print a table of the top 20.

**Acceptance:** on the fixture, deliberate near-duplicates land in one cluster; top 15 has no obvious off-topic items; re-running gives identical output.

## Phase 3 — Generate (≈3 h)

- `llm.py`: `openai.OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)`, temperature 0.2; `json_schema` response format, fallback to `json_object` + schema in prompt; mode is a config flag.
- `prompts.py`: one call per cluster. Concise, factual, no hype, provided text only. Input: headline + up to 5 items (title, source, url, first 1,500 chars). Output: one Initiative minus rank/trend_score.
- `validate.py`: Pydantic + word limits; sources ⊆ cluster URLs (else top item URL); grounding check of model-name tokens and numbers; up to 2 retries with error appended, then extractive fallback.
- Take first 10 valid initiatives in score order; write `top10.json` (Digest). `--dry-run` skips the LLM.

**Acceptance:** exactly 10 schema-valid items with input source URLs; zero unresolved grounding misses for llm items; `--dry-run` works offline.

## Phase 4 — Render + demo run (≈2 h)

- `render.py` + `digest.html.j2`: single self-contained, table-based HTML (Outlook-friendly). Per item: rank, category badge, title, what/why, OSS/license tag, source links, trend score. Header: date, window, "Generated automatically — verify before forwarding". Footer: sources scanned + counts.
- `scripts/run_all.sh`: process → generate → render, non-zero exit on failure, prints output path and runtime.
- README: purpose, airgap data flow, setup, env vars, commands, tuning, limitations, next steps.

**Acceptance:** `bash scripts/run_all.sh` on the fixture produces a 10-item `digest.html`; human spot-check on the real snapshot finds no factual errors.

## Definition of done

- One command turns a fetched JSONL snapshot into a reviewed Top-10 HTML digest on the airgapped server.
- All offline tests pass; fetch has been run live at least once.
- README explains the next steps the manager is being asked to approve.

## Stop and ask before

- Adding any dependency not listed above.
- Changing the data contracts in `models.py` after Phase 1.
- Anything that sends data outside the server or requires a new network endpoint.

## Phase 5 — Accuracy + ranking fixes (added 2026-10-12 after the Phase 4 spot-check failed)

Spot-check of the 2026-10-09 snapshot found: HF tag lists pasted as summaries, HN items summarised from a bare title, engagement numbers misattributed ("63 points on GitHub"), unsupported "Open source" tags, paper clusters linking unrelated papers, robotics/ML papers labelled LLM, near-identical what/why, and community fine-tunes ranked as launches.

- Fetch: run full-page extraction for `discussion` items too (HN links to the real page).
- Generate input: strip HF tag lists; no engagement numbers in the prompt or the grounding text; paper clusters send only their top paper.
- Validate: `open_source`/`license` kept only if the cluster text supports them; paper clusters forced to `Research` and linked to the top paper only; reject what/why that repeat each other.
- Process: penalise clusters led by a derivative HF model (fine-tune/quant of another owner's base model); skip clusters with too little text to summarise; at most 3 paper clusters among candidates.
- Re-fetch live, re-run with the LLM, redo the spot-check.

**Acceptance:** human spot-check of all 10 items against their sources finds no factual errors; no tag dumps; every item has readable source text.

## Phase 6 — Sectioned briefing format

Replace the flat Top 10 with an edited briefing: TL;DR (3 grounded bullets), one lead story, sections by topic (Models & Releases, Agents & Tools, Research, India / Indic AI, Policy), "Quick hits" one-liners, and a computed "Trends this week". Empty sections say so instead of being padded. Needs a `models.py` contract change (approved 2026-10-12).

**Acceptance:** fixture and real snapshot render all sections; every bullet carries a source URL; grounding check covers the TL;DR.

**Update 2026-10-12:** the sectioned briefing was built, then replaced at review. The final design uses the NSE masthead from the Macro Review (logo, title, date/issue line, colour bar) over the short numbered points from Phase 4 (category, title, What, Why it matters, links, published date, trend score), plus "Also noted" and a sources footer. The weekly overview stays in code behind `generate.write_overview` (off).
