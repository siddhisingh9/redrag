# RedRAG

**An automated red-teaming agent for RAG (Retrieval-Augmented Generation) applications.**

RedRAG attacks a RAG chatbot, detects when it leaks data or obeys injected
instructions, scores each attempt with a judge pipeline (rules → fine-tuned
small classifier → Gemini as backup), and measures how much each defense helps.

> **Status:** early development, milestone **M0 (project setup)**.
> Commands other than `show-config` are placeholders for now.

Everything runs on free tools: local models through Ollama, and the Gemini API
free tier as the only remote model.

---

## Safety by design

- **No real secrets.** Every "secret" in the target app is a fake canary string
  such as `CANARY-8F3A21`, planted so leaks are easy to detect.
- **Benign attacks only.** An attack "succeeds" when the chatbot leaks a canary,
  says a harmless marker word (e.g. `PINEAPPLE-7`), or reveals its system prompt.
  RedRAG never tries to make a model produce harmful content.
- **Your own system only.** RedRAG is meant to test the bundled target app or
  apps you own. It is not for attacking third-party systems.

---

## Setup

### 1. Prerequisites

- **Python 3.11+** (3.10 also works for now)
- **Ollama** for local models: <https://ollama.com/download>
- A GPU is optional. The defaults fit a laptop GPU with **4 GB VRAM**
  (e.g. RTX 3050). Embeddings and the judge classifier run on CPU.

### 2. Create a virtual environment and install dependencies

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# then, on any OS
pip install -r requirements.txt
pip install -e .          # registers the `redrag` command
```

`requirements.txt` includes PyTorch (pulled in by `sentence-transformers`), so
the first install is a large download.

### 3. Pull a small local model with Ollama

Install and start Ollama, then pull the default model (about 2 GB, fits 4 GB VRAM):

```bash
ollama pull llama3.2:3b
```

Any small instruct model works. To use another one, set `OLLAMA_VICTIM_MODEL` /
`OLLAMA_ATTACKER_MODEL` in `.env`.

### 4. Get a free Gemini API key (optional but recommended)

Gemini is used only as the backup judge and for report summaries.

1. Go to **Google AI Studio**: <https://aistudio.google.com/apikey>
2. Sign in and click **Create API key** (free tier, no credit card).
3. Copy the env template and paste your key:

```bash
cp .env.example .env      # Windows: copy .env.example .env
# then edit .env and set GEMINI_API_KEY=...
```

Check in AI Studio which **Flash / Flash-Lite** model is on the free tier and set
`GEMINI_MODEL` if it differs from the default. RedRAG caches every Gemini answer
on disk and stops at `GEMINI_DAILY_BUDGET` requests per day, so it stays inside
the free quota.

`.env` is gitignored. Never commit it.

### 5. Check that everything is connected

```bash
python scripts/smoke_llm.py
```

Each model reports `OK` or `SKIP` (with a hint on how to fix it). `SKIP` is fine:
everything except a real scan works without Ollama or Gemini.

---

## Usage

```bash
redrag --help              # or: python -m redrag.cli --help
redrag show-config         # effective settings (API key hidden)
```

Coming in later milestones:

| Command | What it does | Milestone |
|---|---|---|
| `redrag serve-target` | Start the vulnerable "Campus Handbook Assistant" RAG app | M1 |
| `redrag scan` | Run the attack suite against the target | M2 |
| `redrag report` | Build a Markdown/HTML report from a scan | M2 |
| `redrag mcp` | Start the MCP server (stdio) | M6 |

---

## Configuration

Defaults live in [`config/default.yaml`](config/default.yaml). Environment
variables (or a `.env` file) override them. See [`.env.example`](.env.example)
for every variable. To use a different YAML file, set `REDRAG_CONFIG`.

---

## Development

```bash
pytest          # all tests run offline: no internet, Ollama or Gemini key needed
ruff check .    # lint
ruff format .   # format
```

Tests use `MockClient`, a deterministic fake LLM, so they are fast and repeatable.

---

## Project layout

```
config/default.yaml        settings (models, top_k, defenses, judge thresholds, paths)
redrag/
  config.py                loads YAML + env overrides
  llm/                     LLM clients: Ollama, Gemini (cache + budget + backoff), Mock
  target/                  vulnerable RAG app (FastAPI), RAG pipeline, defenses     [M1]
  attacks/                 attack schema, seed attacks, poisoned-doc builder         [M2]
  agent/                   LangGraph adaptive attacker                               [M3]
  judge/                   rules → fine-tuned classifier → Gemini judge              [M2/M4]
  eval/                    metrics (ASR, FRR, judge agreement) and reports           [M2/M5]
  runner.py                orchestrates a scan                                       [M2]
  cli.py                   `redrag` command
  mcp_server.py            MCP server                                                [M6]
notebooks/                 judge fine-tuning notebook (Kaggle/Colab)                 [M4]
dashboard/                 Streamlit dashboard                                       [M7]
scripts/smoke_llm.py       checks Ollama + Gemini connectivity
tests/                     offline pytest suite
```

---

## Roadmap

| Milestone | Scope | Status |
|---|---|---|
| M0 | Project setup, config, LLM clients | ✅ in review |
| M1 | Vulnerable target RAG app + defenses | ⏳ |
| M2 | Attack library, rule judge, basic scan + report | ⏳ |
| M3 | LangGraph adaptive attacker | ⏳ |
| M4 | Fine-tuned judge (DeBERTa-v3-small) + Gemini backup judge | ⏳ |
| M5 | Defense evaluation (ASR/FRR), HTML report, tracing | ⏳ |
| M6 | MCP server + GitHub Actions gate | ⏳ |
| M7 | Dashboard + README polish | ⏳ |

### Baseline tools (optional, for comparison only)

[garak](https://github.com/NVIDIA/garak) and [promptfoo](https://www.promptfoo.dev/)
are not part of `requirements.txt`. If you want to compare against them, install
them in a separate environment.
