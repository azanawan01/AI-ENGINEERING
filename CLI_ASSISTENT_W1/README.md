# CLI AI Research & Data Assistant

A terminal assistant built on a hand-written agent loop: one LLM decides which
approved Python tools to call, and Python validates every request before
anything runs. No agent frameworks are used, so every step of the loop is
visible in the code.

## Status

| Stage | Scope | State |
|---|---|---|
| Day 1 | Config, provider layer, streaming chat, token usage | Done |
| Day 2 | Tool registry, Pydantic schemas, validation pipeline | Done |
| Day 3 | Streaming tool calls & multi-turn agent loop | Done |
| Day 4 | 4 Real tools (clean_csv, fix_grammar, translate_text, natural_language_database) | Done |

## Tools Overview

1. **`clean_csv`**:
   - Identifies & drops duplicate rows (with optional column subsets).
   - Handles empty/missing rows (`drop_any_empty`, `drop_all_empty`, `fill_defaults`).
   - Generates terminal ASCII charts (horizontal category bar charts & numeric histograms).
   - Features added: IQR outlier detection, type inference, summary statistics, header normalization, and cleaned CSV export.
2. **`fix_grammar`**:
   - Analyzes and fixes punctuation, sentence structure, verb tenses, and spelling.
   - Returns validated JSON with categorized corrections, explanations, and quality score.
3. **`translate_text`**:
   - User decides input text, target language, and optional tone (neutral, formal, informal, business).
   - Returns structured JSON with detected source language, translated text, and linguistic notes.
4. **`natural_language_database`**:
   - 4-stage pipeline: **SQL generation -> Validation -> Execution -> Result**.
   - Enforces read-only safety (blocks INSERT, UPDATE, DELETE, DROP, etc. and semicolon injections).
   - Validates SQL syntax and schema with SQLite `EXPLAIN`.
   - Returns structured rows and an ASCII table rendered directly in the terminal.

## Architecture

```text
main.py                       bootstraps config, tools, logging and starts CLI
app/cli.py                    terminal I/O and tool invocation indicators
app/agent/session.py          agent loop: streaming completions & tool dispatch
app/agent/conversation.py     message history, trimmed to a bounded window
app/agent/prompts.py          system prompt guiding tool usage
app/llm/base.py               provider-neutral types, protocols, tool models
app/llm/openai_compatible.py  OpenAI SDK provider with streaming tool calling
app/tools/base.py             Tool protocol and ToolRegistry
app/tools/clean_csv.py        clean_csv implementation + ASCII visualizations
app/tools/grammar.py          fix_grammar implementation
app/tools/translate.py        translate_text implementation
app/tools/database.py         natural_language_database (sql->validate->exec->result)
app/config.py                 environment-based, validated settings
workspace/sample_data.csv     sample dirty dataset with duplicates, empties, outliers
workspace/company.db          sample SQLite database with departments, employees, projects
```

The agent depends on `LLMProvider`, not on a vendor SDK. The single provider
uses the OpenAI SDK against any OpenAI-compatible server, which covers both a
local Ollama instance and OpenAI itself through configuration alone.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally, with a model that supports
  tool calling (look for the "tools" tag in Ollama's model library)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env               # then set LLM_MODEL
ollama pull <your-model>
```

## Running

```bash
python main.py
```

Commands inside the chat: `/reset` clears the conversation, `/exit` quits.
Diagnostics are written to `logs/app.log`, not to the terminal.

## Configuration

All settings are environment variables; see `.env.example` for the full list
and defaults. `LLM_MODEL` is the only required value. Invalid values stop the
app at startup with a message naming the variable to fix.

## Testing and checks

```bash
pytest
ruff check . && ruff format --check .
mypy
```

Unit tests need neither Ollama nor network access: the provider is tested with
a fake SDK client, and everything above it with a scripted fake provider.

## Known limitations

- History is capped by message count, not tokens. Exact token counts need the
  model's tokenizer, which the OpenAI-compatible API does not expose.
- An HTTP 404 is reported as a missing model. A wrong `LLM_BASE_URL` path also
  produces a 404 and will show the same message.
