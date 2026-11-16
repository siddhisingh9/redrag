# RedRAG

Automated red-teaming for RAG chatbots. RedRAG runs prompt-injection, data-exfiltration
and system-prompt-extraction attacks against a RAG app, judges each response, and
measures how much each defense reduces attacks and how many normal questions it
blocks along the way.

Everything runs locally on free tools (Ollama with a 3B model). Gemini's free tier is
an optional backup judge.

![Dashboard overview](docs/screenshots/overview.png)

## Results

Scans of the bundled target app with `llama3.2:3b`. ASR is the share of attacks that
got through; FRR is the share of harmless questions a defense wrongly blocked.

| Defenses | ASR | FRR |
|---|---|---|
| none | 85% | 0% |
| input filter | 65% | 6% |
| context sanitiser | 65% | 0% |
| output leak filter | 50% | 0% |
| judge guard (fine-tuned classifier) | 0% | 0% |
| all four | 0% | 6% |

20 attacks and 49 harmless questions per configuration. All 16 combinations are in the
full report.

With the adaptive attacker rewriting failed attacks (up to 4 tries per seed, 55 seeds):

| Defenses | As written | After rewrites |
|---|---|---|
| input filter + context sanitiser + output leak filter | 20% | 55% |
| judge guard | 0% | 2% |

The judge's classifier agrees with held-out hand labels 90% of the time (Cohen's
kappa 0.76, 72 labels). The judge guard was trained on the same attack families, so the
rewrite test is the more meaningful one. The one attack that got past it leaked a canary
the output leak filter would have caught, so the two work best together.

## How it works

```mermaid
flowchart LR
    seeds["Seed attacks"] --> agent["Attacker agent<br/>(LangGraph)"]
    agent --> target["Target RAG app<br/>(FastAPI + Chroma + Ollama)"]
    target --> judge["Judge<br/>rules → classifier → Gemini"]
    judge --> agent
    judge --> results[("runs/")]
    results --> out["Reports · Dashboard · MCP"]
```

- **Attacks:** 55 seed attacks across five categories (direct and indirect injection,
  system-prompt extraction, data exfiltration, multi-turn escalation). In agent mode, a
  LangGraph loop rewrites failed attacks based on why they failed.
- **Target:** a deliberately vulnerable handbook chatbot with fake secrets (canary
  strings) and four defenses that can be switched per request.
- **Judge:** rules catch clear leaks; a fine-tuned DeBERTa-v3-small classifier handles
  the rest; Gemini is consulted only when the classifier is unsure.

All secrets are fake and attacks are benign (marker words, canaries, prompt leaks).
Only use RedRAG on systems you own.

## Setup

Requires Python 3.10+ and [Ollama](https://ollama.com/download).

```bash
git clone https://github.com/siddhisingh9/redrag.git
cd redrag
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
ollama pull llama3.2:3b
```

Optional: copy `.env.example` to `.env` and add a `GEMINI_API_KEY` from
[Google AI Studio](https://aistudio.google.com/apikey). Responses are cached and capped
at a daily budget.

To try it without Ollama, scan the built-in mock target:

```bash
redrag scan --mock-target
streamlit run dashboard/app.py
```

## Usage

```bash
redrag serve-target                    # start the target app (http://127.0.0.1:8000)

redrag scan                            # all attacks, no defenses vs all defenses
redrag scan --all-combinations         # every combination of defenses
redrag scan --agent                    # adaptive attacker
redrag scan -d none -d output_leak_filter --categories data_exfiltration

redrag report                          # rebuild the latest report
streamlit run dashboard/app.py         # browse results
```

Each scan writes `runs/<timestamp>/` with the raw results and `report.html`.

| Defense | What it does |
|---|---|
| `input_filter` | blocks messages with known injection phrases |
| `context_sanitiser` | strips instructions from retrieved documents |
| `output_leak_filter` | blocks answers containing secrets or the system prompt |
| `judge_guard` | blocks answers the trained classifier flags (needs a trained model) |

**Training the judge.** Label results with `redrag label` (or the dashboard's Label
page), export with `redrag export-dataset`, train with
`notebooks/finetune_judge.ipynb` on a free Kaggle or Colab GPU, put the output in
`models/judge-deberta/`, then check agreement with `redrag judge-eval`.

**Tracing.** Run `phoenix serve`, then add `--trace` to `serve-target` and `scan` to see
every attack as a trace in [Arize Phoenix](https://github.com/Arize-ai/phoenix).

**MCP.** `redrag mcp` exposes `scan_rag_app`, `list_attacks` and `get_report` to MCP
clients, for example:

```bash
claude mcp add redrag -- /path/to/redrag/.venv/Scripts/redrag mcp
```

**CI.** `.github/workflows/redteam.yml` runs lint, tests and a deterministic scan
against a mock target on every pull request. It fails if ASR with all defenses rises
above the threshold in `config/ci_baseline.json`. Run it locally with
`redrag ci-check --mock-target`.

Settings live in `config/default.yaml`; environment variables in `.env` override them.

| Attacks | Lineage |
|---|---|
| ![Attacks page](docs/screenshots/attacks.png) | ![Lineage page](docs/screenshots/lineage.png) |

## Development

```bash
pytest           # runs offline, no Ollama or API key needed
ruff check .
```

```
redrag/
  target/        vulnerable RAG app and defenses
  attacks/       attack schema and seed attacks
  agent/         LangGraph attacker
  judge/         rules, classifier, Gemini judge
  eval/          metrics, reports, labelling
  cli.py         command-line interface
  mcp_server.py  MCP server
dashboard/       Streamlit app
notebooks/       judge fine-tuning
```
