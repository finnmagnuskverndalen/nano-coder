# nano-coder

```
 ________________
< nano-coder     >
 ----------------
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
```

> A coding agent built for tiny and small language models (≤2B parameters), with live streaming CLI and split-pane terminal UI.

Inspired by [rasbt/mini-coding-agent](https://github.com/rasbt/mini-coding-agent), this project rethinks the agent loop from the ground up for models that struggle with long prompts, complex JSON, and multi-step reasoning.

---

## Why nano?

Most coding agents are designed for 7B+ models. At ≤2B parameters, the usual assumptions break down:

- Long system prompts eat most of the context window before the task even starts
- JSON tool-call syntax is unreliable at this scale — XML attributes are much easier to generate
- Multi-step reasoning chains degrade quickly without aggressive error recovery
- Ollama defaults to only **2048 tokens** of context — nano-coder overrides this to **16K** (the model supports 32K)

`nano-coder` addresses all of these with a stripped-down prompt engine (~200 tokens static), XML-first tool syntax, a noise-tolerant parser, explicit `num_ctx` override, and a live streaming terminal UI.

---

## Recommended models

| Model | Size | Notes |
|---|---|---|
| `qwen2.5-coder:1.5b` | 1.5B | Best overall — 32K context, strong code instruct |
| `qwen2.5-coder:0.5b` | 0.5B | Minimal footprint, surprisingly capable |
| `smollm2:1.7b` | 1.7B | Good instruction following |
| `deepseek-coder:1.3b` | 1.3B | Strong at Python/JS |
| `tinyllama:1.1b` | 1.1B | Works with reduced tool set (`--tool-set core`) |

---

## Features

- **Cowsay welcome screen** — because every good tool deserves a cow
- **Interactive workspace picker** — choose or create a project folder at startup
- **Two modes** — `Ask` (confirm risky tools) and `Auto-Accept` (fully autonomous), toggle with `/mode`
- **Black & white terminal UI** — clean greyscale palette, no colour distractions
- **16K context by default** — overrides Ollama's 2048-token default via `num_ctx`; configurable up to the model's 32K max
- **Nano-optimized prompt engine** — ~200 token static prefix, XML-only tool format
- **Live streaming split-pane** — model tokens stream left, tool results appear right, in real time
- **Noise-tolerant parser** — handles markdown fences, `<think>` blocks, wrong case, missing tags
- **Session persistence** — save and resume sessions as JSON
- **Zero runtime dependencies** — standard library only (Python 3.10+)

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running
- A pulled model: `ollama pull qwen2.5-coder:1.5b`

---

## Install

```bash
git clone https://github.com/finnmagnuskverndalen/nano-coder.git
cd nano-coder
```

No Python dependencies to install. Run directly:

```bash
python nano_coding_agent.py
```

Or with `uv`:

```bash
uv run nano-coder
```

---

## Usage

### Interactive (default)

```bash
python nano_coding_agent.py
```

On startup you'll see the workspace picker — choose a directory, type a path, or create a new folder:

```
  Choose a project folder
──────────────────────────────────────────────────────────────────────
  [0]  (current dir)  /home/finn/projects
  [1]  my-app/          /home/finn/projects/my-app
  [2]  nano-coder/      /home/finn/projects/nano-coder
  [3]  enter a path…
  [4]  create a new folder
```

### One-shot

```bash
python nano_coding_agent.py "write a binary search function and test it"
```

### Common flags

```bash
# Skip the workspace picker
python nano_coding_agent.py --cwd /path/to/project

# Fully autonomous mode (no approval prompts)
python nano_coding_agent.py --mode auto

# Use a different model
python nano_coding_agent.py --model deepseek-coder:1.3b

# Set context window (default 16K, max 32K for qwen2.5-coder)
python nano_coding_agent.py --num-ctx 32768

# Enable full tool set (adds search + patch_file)
python nano_coding_agent.py --tool-set full

# Resume last session
python nano_coding_agent.py --resume latest
```

---

## Context window

Ollama defaults to **2048 tokens** regardless of what a model supports. `nano-coder` sets `num_ctx=16384` in every API call, giving the agent a real working window:

| `--num-ctx` | RAM impact | Good for |
|---|---|---|
| `4096` | minimal | very constrained hardware |
| `16384` (default) | ~1–2 GB extra | most laptops, good balance |
| `32768` | ~2–4 GB extra | full model capacity, best results |

The qwen2.5-coder family (all sizes 0.5B–32B) supports up to **32K tokens**. Setting `--num-ctx 32768` unlocks the full window.

---

## Modes

| Mode | Flag | `/mode` toggle | Description |
|---|---|---|---|
| **Ask** | `--mode ask` (default) | → Auto-Accept | Prompts before every risky tool (`write_file`, `run_shell`, `patch_file`) |
| **Auto-Accept** | `--mode auto` | → Ask | Fully autonomous — no prompts, agent runs freely |

The current mode is always visible in the prompt line and the split-pane header.

---

## CLI interface

```
═══════════════════════════════════════════════════════════════════════════
                  ________________
                 < nano-coder     >
                  ----------------
                         \   ^__^
                          \  (oo)\_______
                             (__)\       )\/\
                                 ||----w |
                                 ||     ||
──────────────────────────────────────────────────────────────────────────
  workspace    /home/finn/my-project
  model        qwen2.5-coder:1.5b        branch    main
  ctx          16,384 tokens             mode      ASK (confirm risky tools)
  session      20260417-142310-a3f1bb
═══════════════════════════════════════════════════════════════════════════

[20260417-14]  [ ASK ]  nano-coder > write a fizzbuzz function

┌─ qwen2.5-coder:1.5b  [ ASK ] ──────┬─ read_file ──────────────────────┐
│ <tool name="read_file"              │ ▶ list_files  path=.              │
│       path="main.py"                │   [F] main.py                    │
│       start="1" end="20">           │   [F] tests/test_main.py         │
│ </tool>                             │                                  │
│                                     │ ▶ write_file  path=fizzbuzz.py   │
│ <tool name="write_file"             │   wrote fizzbuzz.py (180 chars)  │
│       path="fizzbuzz.py">           │                                  │
│   <content>                         │                                  │
│   def fizzbuzz(n): ...              │                                  │
│   </content>                        │                                  │
│ </tool>                             │                                  │
└─────────────────────────────────────┴──────────────────────────────────┘
  Done. fizzbuzz.py written with fizzbuzz(n) function.
```

---

## Interactive commands

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/mode` | Toggle between Ask and Auto-Accept |
| `/memory` | Show distilled session memory |
| `/session` | Show path to current session file |
| `/tools` | List active tools with risk level |
| `/reset` | Clear session history and memory |
| `/exit` | Exit |

---

## Tools

### Core (default)

| Tool | Risky | Description |
|---|---|---|
| `list_files` | — | List files in workspace |
| `read_file` | — | Read file by line range (default 60 lines) |
| `write_file` | ⚠ | Write or overwrite a file |
| `run_shell` | ⚠ | Run a shell command in workspace root |

### Full (`--tool-set full`)

Adds `search` (ripgrep/fallback) and `patch_file` (targeted replacement). Recommended for 1.5B+ models.

---

## All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--cwd` | (picker) | Workspace directory — skips the picker |
| `--model` | `qwen2.5-coder:1.5b` | Ollama model name |
| `--host` | `http://127.0.0.1:11434` | Ollama server URL |
| `--mode` | `ask` | `ask` or `auto` |
| `--num-ctx` | `16384` | Context window tokens (model max: 32768) |
| `--max-steps` | `5` | Max tool/model turns per request |
| `--max-new-tokens` | `384` | Max output tokens per step |
| `--temperature` | `0.1` | Sampling temperature |
| `--top-p` | `0.9` | Top-p nucleus sampling |
| `--tool-set` | `core` | `core` (4 tools) or `full` (6 tools) |
| `--resume` | — | Session ID or `latest` |
| `--no-stream` | — | Disable live streaming (plain output) |
| `--ollama-timeout` | `300` | Ollama request timeout in seconds |

---

## Project layout

```
nano-coder/
├── nano_coding_agent.py   # Complete agent — single file, zero runtime deps
├── tests/
│   └── test_agent.py      # 42 tests: parser, tools, agent loop, streaming
├── PROJECT_PLAN.md
├── pyproject.toml
└── README.md
```

---

## Design vs mini-coding-agent

| | mini-coding-agent | nano-coder |
|---|---|---|
| Target size | 4B–14B | ≤2B |
| Tool format | JSON + XML fallback | XML only |
| System prompt | ~600 tokens | ~200 tokens |
| Context window | Ollama default (2048) | **16K default, up to 32K** |
| Tool count | 7 | 4 core / 6 full |
| CLI | Simple REPL | Live streaming split-pane |
| Workspace | `--cwd` flag only | Interactive picker + `--cwd` |
| Modes | ask / auto / never | **Ask / Auto-Accept** (toggleable live) |
| Default temperature | 0.2 | 0.1 |

---

## License

Apache 2.0
