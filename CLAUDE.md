# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TradingAgents is a multi-agent LLM financial trading/research framework built on LangGraph. A `StateGraph` chains analyst → researcher → trader → risk-management → portfolio-manager agent nodes; each node is an LLM call (via a provider-agnostic client layer) optionally bound to tools that fetch market/fundamentals/news/macro data from pluggable vendors. It ships both as a Python package (`tradingagents`) and an interactive CLI (`cli`, entry point `tradingagents`).

## Commands

```bash
pip install -e ".[dev]"       # editable install with dev extras (ruff, pytest)
pytest -q                     # run the full test suite
pytest tests/test_backtest.py                      # run one test file
pytest tests/test_backtest.py::test_name -q         # run one test
pytest -m unit                # marker-filtered run (markers: unit, integration, smoke)
ruff check .                  # lint (strict, whole repo — this is what CI runs)
tradingagents                 # launch the interactive CLI (or: python -m cli.main)
tradingagents backtest NVDA,AAPL --start 2026-06-01 --end 2026-08-01 --every 7
```

CI (`.github/workflows/*`) runs `pytest -q` on Python 3.10–3.13, a clean-install smoke import (`pip install .` with no dev extras, then `import tradingagents, cli.main`), and `ruff check .` on the full repo. Keep new code importable without dev/optional extras, and lint-clean under the strict ruff select in `pyproject.toml` (`E, W, F, I, B, UP, C4, SIM`).

Tests auto-mock API keys and reset global config per test (see `tests/conftest.py`) — no real credentials or network calls are needed to run the suite.

## Architecture

### Graph pipeline (`tradingagents/graph/`)

`TradingAgentsGraph` (`trading_graph.py`) is the orchestrator. Construction: build two LLM clients (`deep_think_llm` for judge/manager nodes, `quick_think_llm` for analyst/debate nodes) via `llm_clients.create_llm_client`, build per-category `ToolNode`s, then hand everything to `GraphSetup.setup_graph()` (`setup.py`), which wires analyst nodes → clear-message node → Bull/Bear Researcher debate → Research Manager → Trader → Aggressive/Conservative/Neutral risk debate → Portfolio Manager → END, using `ConditionalLogic` (`conditional_logic.py`) for the debate/tool-loop routing. Every conditional edge must map onto a *complete* path-map dict (see `DEBATE_PATH_MAP`/`RISK_ANALYSIS_PATH_MAP`) so a router fall-through can never hit a missing key mid-run.

Which analysts run is configurable (`selected_analysts`, default `market, social, news, fundamentals`); `analyst_execution.py` builds the execution plan/node sequence from that list, so the graph shape itself depends on the selection.

Key entry points on `TradingAgentsGraph`:
- `propagate(ticker, trade_date, asset_type="stock", portfolio=None)` — the main call; wraps `checkpoint_scope` and returns `(final_state, signal)`.
- `create_run_state(...)` — builds the initial graph state; resolves pending decision-log entries, injects point-in-time memory context and resolved instrument identity. Any alternate entry point must go through this (or duplicate it) to keep the decision log correct.
- Checkpointing (`checkpointer.py`) is opt-in (`checkpoint_enabled`), keyed by a `_run_signature()` that folds in analyst selection, debate/risk depth, asset type, and portfolio fingerprint — so a resume under a *different* graph shape starts fresh instead of silently continuing stale state.
- `signal_processing.py` extracts one of 5 ratings (Buy/Overweight/Hold/Underweight/Sell) or `"REVIEW"` from the final decision text; always check `rating.is_review()` before mapping to `PortfolioRating`.

### Agents (`tradingagents/agents/`)

Organized by role: `analysts/` (market, fundamentals, news, sentiment/social), `researchers/` (bull/bear), `managers/` (research manager, portfolio manager), `risk_mgmt/` (aggressive/conservative/neutral debators), `trader/`. Each `create_*` factory closes over an LLM and returns a LangGraph node function. Shared state shape lives in `agents/utils/agent_states.py`; structured-output schemas (for Research Manager, Trader, Portfolio Manager) live in `agents/schemas.py` / `agents/utils/structured.py`. Tool functions the analysts call are defined in `agents/utils/*_tools.py` and bound into `ToolNode`s in `trading_graph.py::_create_tool_nodes`.

### Data layer (`tradingagents/dataflows/`)

`interface.py` is the vendor-routing layer: `TOOLS_CATEGORIES` maps a data category (core stock, technical indicators, fundamentals, news, macro, prediction markets) to its tool names, and `get_vendor(category, method)` resolves which vendor(s) to try, in order, from `config["data_vendors"]` (category-level) and `config["tool_vendors"]` (per-tool override, takes precedence). The chain is walked in order and falls through on `VendorNotConfiguredError`; it does **not** silently try vendors outside the configured chain. Vendor implementations: `y_finance.py`/`yfinance_news.py` (Yahoo Finance, default for most categories), `alpha_vantage*.py` (Alpha Vantage), `sec_edgar.py` (SEC EDGAR — point-in-time fundamentals as originally filed, no key needed), `fred.py` (macro, needs `FRED_API_KEY`), `polymarket.py` (prediction markets, keyless), `reddit.py`/`stocktwits.py` (social sentiment).

Point-in-time integrity is a first-class concern here: `date_window.py` and lookahead-guarding logic throughout this package ensure a run dated in the past only sees data that would actually have been available on that date (e.g. SEC filings not yet filed, or a stat later restated). Several test files (`test_*_lookahead.py`, `test_fundamentals_lookahead.py`, `test_date_boundaries.py`) exist specifically to pin this behavior — preserve it when touching any vendor or the date-window logic.

`symbol_utils.py` normalizes/aliases tickers (e.g. exchange suffixes, crypto, commodities like `XAUUSD` → `GC=F`) so the same instrument is used consistently across pricing, benchmark lookup, and identity resolution. `market_data_validator.py` backs `get_verified_market_snapshot`, a deterministic tool bound into the market analyst's tool node specifically so exact price/indicator claims are grounded rather than hallucinated.

### LLM providers (`tradingagents/llm_clients/`)

`factory.py::create_llm_client(provider, model, base_url, **kwargs)` is the single construction point. Anthropic, Google, Azure, and Bedrock have dedicated native clients; every other provider name (OpenAI, DeepSeek, Qwen/DashScope, GLM/Zhipu, MiniMax, OpenRouter, Mistral, Kimi/Moonshot, Groq, NVIDIA, Ollama, or a custom `openai_compatible` endpoint) is routed through `openai_client.py`'s OpenAI-compatible path, keyed off a shared provider registry — that registry, not this factory, is the place to add a new OpenAI-compatible provider. `model_catalog.py` holds the curated model lists the CLI presents per provider (any model ID is still accepted even if unlisted). `capabilities.py`/`validators.py` gate provider-specific kwargs (e.g. Anthropic `effort`, OpenAI `reasoning_effort`, Google `thinking_level` — see `TradingAgentsGraph._get_provider_kwargs`) so an option only reaches providers that support it. `api_key_env.py` maps provider → expected API-key env var for auto-detection in the CLI.

### Persistence

Two independent, opt-in-vs-always-on systems, both under `~/.tradingagents/` by default (overridable via `TRADINGAGENTS_*` env vars — see `default_config.py`):
- **Decision log** (`agents/utils/memory.py::TradingMemoryLog`, always on) — every completed run appends its decision; the *next* run for the same ticker fetches realized/alpha return, generates a reflection, and injects recent same-ticker + cross-ticker lessons into the Portfolio Manager prompt. `_memory_as_of()` in `trading_graph.py` filters injected lessons to what was already resolved by the run's trade date for historical/backtest runs, so backtests can't see future outcomes.
- **Checkpoint resume** (`graph/checkpointer.py`, opt-in via `checkpoint_enabled`/`--checkpoint`) — per-ticker SQLite (LangGraph `SqliteSaver`) so a crashed run resumes from the last completed node; cleared automatically on success.

### Backtesting (`tradingagents/backtest.py`)

`iter_grid()` + `run_backtest()` run the full pipeline over a ticker × date grid into a separate decision log (never the user's live one), scoring cells whose holding window has since traded; `run_id` reuse skips already-completed cells so an interrupted sweep resumes.

### Config

`default_config.py::DEFAULT_CONFIG` is the single source of defaults, with `TRADINGAGENTS_*` env-var overrides applied automatically (see `_ENV_OVERRIDES` table — add a row there to expose a new key, no other plumbing needed). Coercion is driven by the *type of the existing default*, so a wrong-typed env value fails loudly at startup rather than silently misconfiguring a run. `dataflows/config.py::set_config`/`get_config` hold the process-global config used by vendor routing; it *merges* on `set_config`, which is why tests must reset it between runs (see `tests/conftest.py::_isolate_config`).

## Conventions worth knowing

- Provider SDKs are imported lazily inside client/factory functions (not at module top) so importing the package or running test collection never requires every LLM SDK to be installed, and never fails on a missing API key.
- Config values that come from env vars are strings; code that consumes them (temperature, max_tokens, retries, booleans) must coerce explicitly rather than assume a Python type.
- `results_dir` writes are ticker-path-sanitized via `dataflows/utils.py::safe_ticker_component` — never join a raw user-supplied ticker into a filesystem path.
- When adding a conditional LangGraph edge, give it a *complete* path-map (every node the router function can return), not just the paths currently expected — see the `#1088` comments in `graph/setup.py`.
