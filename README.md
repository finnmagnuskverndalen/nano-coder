# nano-coding-agent

> A coding agent built for tiny and small language models (≤2B parameters), with a live streaming CLI interface powered by Ollama.

Inspired by [rasbt/mini-coding-agent](https://github.com/rasbt/mini-coding-agent), this project rethinks the agent loop from the ground up for models that struggle with long prompts, complex JSON, and multi-step reasoning.

---

## Why nano?

Most coding agents are designed for 7B+ models. At ≤2B parameters, the usual assumptions break down:

- Long system prompts eat most of the context window before the task even starts
- JSON tool-call syntax is hard to produce reliably at this scale
- Multi-step reasoning chains degrade quickly without error recovery
- Format compliance is inconsistent — the parser must be forgiving

`nano-coding-agent` addresses all of these with a stripped-down prompt engine, XML-first tool syntax, a noise-tolerant parser, and a live streaming terminal UI so you can watch the model think in real time.

---

## Recommended models

| Model | Size | Notes |
|---|---|---|
| `qwen2.5-coder:1.5b` | 1.5B | Best overall for code at this size |
| `smollm2:1.7b` | 1.7B | Good instruction following |
| `deepseek-coder:1.3b` | 1.3B | Strong at Python/JS |
| `tinyllama:1.1b` | 1.1B | Minimal, works with reduced tool set |

---

## Features

- **Nano-optimized prompt engine** — compressed system prompt (~200 tokens), XML-only tool format, minimal examples
- **Noise-tolerant parser** — handles markdown-wrapped output, missing tags, partial JSON, and plain-text responses
- **Live streaming CLI** — real-time token streaming from Ollama with a split-pane terminal view (model output on the left, tool results on the right)
- **Reduced tool set** — 4 core tools sized for small context windows, with optional extras
- **Aggressive context trimming** — rolling history window, deduped file reads, clipped tool output
- **Session persistence** — save and resume sessions as JSON
- **Approval gating** — ask / auto / never modes for risky operations
- **Zero runtime dependencies** — standard library only (Python 3.10+)

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running
- A pulled model (`ollama pull qwen2.5-coder:1.5b`)

Optional: `uv` for environment management.

---

## Install

```bash
git clone https://github.com/your-user/nano-coding-agent.git
cd nano-coding-agent
```

No Python dependencies to install. Run directly:

```bash
python nano_coding_agent.py
```

Or with `uv`:

```bash
uv run nano-coding-agent
```

---

## Usage

```bash
# Interactive REPL (default)
python nano_coding_agent.py

# One-shot prompt
python nano_coding_agent.py "write a fizzbuzz function and test it"

# Choose model
python nano_coding_agent.py --model deepseek-coder:1.3b

# Set workspace directory
python nano_coding_agent.py --cwd /path/to/project

# Auto-approve all risky operations
python nano_coding_agent.py --approval auto

# Resume last session
python nano_coding_agent.py --resume latest
```

---

## CLI Interface

The terminal is split into two live panels:

```
┌─────────────────────────────────┬─────────────────────────────┐
│  model output                   │  tool results               │
│                                 │                             │
│  > thinking about the task...   │  [list_files] .             │
│    <tool name="read_file"        │  [F] main.py                │
│          path="main.py">        │  [F] tests/test_main.py     │
│    </tool>                      │                             │
│                                 │  [read_file] main.py        │
│    <tool name="write_file"       │     1: def add(a, b):       │
│          path="main.py">        │     2:     return a + b     │
│      <content>                  │                             │
│        def add(a, b):           │                             │
│          return a + b           │                             │
│      </content>                 │                             │
│    </tool>                      │                             │
│                                 │                             │
│    <final>Done. Wrote main.py   │                             │
│    with add() function.</final> │                             │
└─────────────────────────────────┴─────────────────────────────┘
nano-coding-agent [qwen2.5-coder:1.5b] > _
```

Tokens stream live as the model generates them. Tool calls execute immediately when the closing tag is detected mid-stream, so you see results appear in the right pane while the model is still writing.

---

## Interactive commands

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/memory` | Show distilled session memory |
| `/session` | Show current session file path |
| `/tools` | List available tools |
| `/reset` | Clear session history and memory |
| `/exit` or `/quit` | Exit |

---

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--cwd` | `.` | Workspace directory |
| `--model` | `qwen2.5-coder:1.5b` | Ollama model name |
| `--host` | `http://127.0.0.1:11434` | Ollama server URL |
| `--approval` | `ask` | Risky tool policy: `ask`, `auto`, `never` |
| `--resume` | — | Session ID or `latest` |
| `--max-steps` | `5` | Max tool/model turns per request |
| `--max-new-tokens` | `384` | Max output tokens per step |
| `--temperature` | `0.1` | Sampling temperature |
| `--tool-set` | `core` | `core` (4 tools) or `full` (6 tools) |
| `--no-stream` | — | Disable live streaming (batch mode) |

---

## Tools

### Core tool set (default, ≤1.5B recommended)

| Tool | Risky | Description |
|---|---|---|
| `list_files` | No | List files in the workspace |
| `read_file` | No | Read a file by line range |
| `write_file` | Yes | Write or overwrite a file |
| `run_shell` | Yes | Run a shell command |

### Full tool set (`--tool-set full`, 1.5B–2B)

Adds `search` (ripgrep/fallback) and `patch_file` (targeted line replacement).

---

## Tool call format

The agent uses XML-style tool calls exclusively. This is more reliable than JSON for small models:

```xml
<tool name="read_file" path="main.py" start="1" end="40"></tool>

<tool name="write_file" path="utils.py">
<content>
def clamp(value, lo, hi):
    return max(lo, min(hi, value))
</content>
</tool>

<tool name="run_shell" command="python -m pytest -q" timeout="20"></tool>

<final>Done. All tests pass.</final>
```

---

## Project layout

```
nano-coding-agent/
├── nano_coding_agent.py   # Single-file agent (no runtime deps)
├── tests/
│   ├── test_parser.py     # Parser unit tests
│   ├── test_tools.py      # Tool execution tests
│   └── test_agent.py      # Agent loop integration tests (FakeModelClient)
├── pyproject.toml
└── README.md
```

---

## Design differences from mini-coding-agent

| | mini-coding-agent | nano-coding-agent |
|---|---|---|
| Target model size | 4B–14B | ≤2B |
| Tool format | JSON + XML fallback | XML only |
| System prompt size | ~600 tokens | ~200 tokens |
| Tool count | 7 | 4 (core) / 6 (full) |
| CLI | Simple REPL | Live streaming split-pane |
| Delegation | Yes (subagents) | No |
| Default max_new_tokens | 512 | 384 |
| Default temperature | 0.2 | 0.1 |

---

## Development roadmap

See [PROJECT_PLAN.md](./PROJECT_PLAN.md) for the full phased build plan.

---

## License

Apache 2.0
