# nano-coding-agent — Project Plan

> Phased build plan for a coding agent optimized for ≤2B parameter models, with live streaming CLI.

---

## Overview

**Goal:** A single-file Python coding agent (no runtime deps) that works reliably with Ollama models under 2B parameters, featuring a live streaming terminal interface.

**Non-goals:** Multi-agent orchestration, cloud model support, GUI, plugin system.

**Success criteria:** A `qwen2.5-coder:1.5b` model can reliably read a file, write a corrected version, and run its tests — completing the full loop in under 30 seconds on a consumer laptop.

---

## Phase 1 — Core Agent Loop (no UI)

**Goal:** Get a working agent loop that nano models can follow reliably.

### 1.1 Nano-optimized prompt engine

The biggest lever for small model performance is prompt compression. Every token in the system prompt is a token the model can't use for reasoning.

**Tasks:**
- Write a `build_prefix()` that targets ≤200 tokens for the static section
- Rules list: max 6 lines, imperative and concrete
- Tool schema: one line per tool, attribute-style (`name`, `path`, `start`, `end`)
- Examples: one example total, XML format only — no JSON
- Strip workspace context to just `cwd` and `git status --short`; omit commit log and doc snippets by default

**Design note:** The workspace context should be injected per-request, not baked into the prefix. This allows the prefix to be cached/reused and keeps the per-request overhead predictable.

**Estimated prompt budget:**
```
rules:         ~80 tokens
tool schema:   ~60 tokens
one example:   ~40 tokens
               ───────────
total prefix:  ~180 tokens
```

### 1.2 XML-only tool format

JSON tool calls require the model to balance braces, quote keys, and escape strings. XML attributes are much easier to generate at 1B scale.

**Chosen format:**
```xml
<!-- Read -->
<tool name="read_file" path="main.py" start="1" end="40"></tool>

<!-- Write (multiline content in body) -->
<tool name="write_file" path="utils.py">
<content>
def hello():
    return "hi"
</content>
</tool>

<!-- Shell -->
<tool name="run_shell" command="python -m pytest -q" timeout="20"></tool>

<!-- Final answer -->
<final>Done.</final>
```

**Tasks:**
- Implement `parse_tool(raw: str) -> tuple[str, dict] | None`
- Parse XML attributes from `<tool ...>` opener
- Extract `<content>`, `<command>`, `<old_text>`, `<new_text>` from body
- Implement `parse_final(raw: str) -> str | None`

### 1.3 Noise-tolerant parser

Small models produce noisy output. The parser must handle:

| Pattern | Recovery strategy |
|---|---|
| Tool wrapped in ` ```xml ... ``` ` | Strip markdown fence before parsing |
| `<tool>` with JSON body (mini-coding-agent format) | Detect and parse JSON fallback |
| Missing closing `</tool>` | Treat end-of-output as implicit close |
| Plain text without tags | Treat as `<final>` if last step; retry otherwise |
| Partial attribute quoting | Permissive regex, not strict XML parser |
| `<TOOL>` or `<Tool>` (wrong case) | Lowercase normalization before parse |

**Tasks:**
- Implement `normalize(raw: str) -> str` — strip fences, normalize case, trim leading `<think>...</think>` blocks (Qwen 3.x emits these)
- Write unit tests for each noise pattern
- `parse()` should never raise; always returns `("tool"|"final"|"retry", payload)`

### 1.4 Core tool implementations

Start with 4 tools only. Each tool must be sandboxed to the workspace root.

**`list_files(path=".")`**
- Returns `[D]`/`[F]` prefixed entries, max 150 lines
- Skips `.git`, `__pycache__`, `.venv`, `node_modules`

**`read_file(path, start=1, end=60)`**
- Default window is 60 lines (not 200 — saves context)
- Returns `# path\n   1: line\n   2: line...`

**`write_file(path, content)`**
- Creates parent dirs
- Approval gated

**`run_shell(command, timeout=20)`**
- Returns `exit_code`, `stdout`, `stderr`
- Clipped to 2000 chars total (not 4000 — nano models struggle with long tool results)
- Approval gated

**Tasks:**
- Implement all 4 with path sandboxing (`path.resolve()` must be under `workspace_root`)
- Implement `validate_tool(name, args)` with per-tool checks
- Implement `approve(name, args)` with ask/auto/never modes
- Write tool unit tests using `tmp_path` fixtures

### 1.5 Agent loop

**Tasks:**
- Implement `MiniAgent.ask(user_message) -> str`
- Loop: build prompt → call model → parse → execute tool or return final
- Max steps: 5 (default), configurable
- Retry on malformed output: up to `max_steps * 2` attempts total
- On repeated identical tool call: return error and break loop
- Distilled memory: track current task, last 6 touched files, last 4 notes

### 1.6 Ollama model client

**Tasks:**
- `OllamaModelClient.complete(prompt, max_new_tokens) -> str` — batch mode
- `OllamaModelClient.stream(prompt, max_new_tokens) -> Iterator[str]` — streaming mode (for phase 2)
- Set `"raw": true` and `"think": false` in payload (suppress Qwen thinking blocks at API level if possible)
- Raise `RuntimeError` with helpful message if Ollama isn't running

### 1.7 Session persistence

**Tasks:**
- `SessionStore` saves/loads JSON under `<workspace_root>/.nano-coding-agent/sessions/`
- Session format: `id`, `created_at`, `history`, `memory`
- `--resume latest` loads most recently modified session file

### 1.8 Context trimming

At ≤2B, context overflow causes silent quality degradation. Trim aggressively.

**Rules:**
- Tool output clipped to 2000 chars (configurable)
- Total history clipped to 8000 chars
- Recent 4 turns shown at full length; older turns compressed to 150 chars each
- Duplicate `read_file` calls on the same path: only the most recent result kept in history

**Tasks:**
- Implement `history_text(history) -> str` with the above rules
- Implement `clip(text, limit) -> str`

### 1.9 Basic REPL

Minimal interactive loop, no streaming yet.

**Tasks:**
- `/help`, `/memory`, `/session`, `/reset`, `/exit`, `/quit` slash commands
- One-shot mode: `python nano_coding_agent.py "some task"`
- Clean error handling for `EOFError`, `KeyboardInterrupt`

**Phase 1 deliverable:** A working agent that a `qwen2.5-coder:1.5b` model can drive to read, write, and test a file. Single Python file, no deps.

---

## Phase 2 — Live Streaming CLI

**Goal:** Real-time token streaming with a split-pane terminal UI.

### 2.1 Streaming from Ollama

**Tasks:**
- Implement `OllamaModelClient.stream()` using Ollama's `"stream": true` NDJSON API
- Parse each `{"response": "..."}` line as it arrives
- Implement `StreamParser`: stateful parser that detects complete tool tags mid-stream and fires a callback, without waiting for the full response

**StreamParser design:**
```
state machine:
  NORMAL     → accumulate text, print each token
  IN_TOOL    → accumulate tool tag, detect </tool> or <final>
  IN_CONTENT → accumulate <content> body
  DONE       → stop streaming (tool or final detected)
```

When `</tool>` is detected mid-stream, the agent can execute the tool immediately while the model is still generating (or stop generation if Ollama supports it).

### 2.2 Split-pane terminal renderer

Using only `sys.stdout` and ANSI escape codes — no `curses`, no third-party libs.

**Layout:**
```
┌──────────────────────────────────────────────────────────────────────┐
│ nano-coding-agent  model: qwen2.5-coder:1.5b  branch: main          │
├─────────────────────────────────────┬────────────────────────────────┤
│ model output                        │ tool results                   │
│                                     │                                │
│  [streaming tokens here...]         │  [tool name] args              │
│                                     │  output lines...               │
│                                     │                                │
├─────────────────────────────────────┴────────────────────────────────┤
│ nano-coding-agent [qwen2.5-coder:1.5b] > _                          │
└──────────────────────────────────────────────────────────────────────┘
```

**Tasks:**
- Implement `TerminalRenderer` class
  - `render_header(model, branch, session_id)`
  - `render_token(token)` — append to left pane
  - `render_tool_start(name, args)` — add header to right pane
  - `render_tool_result(output)` — append to right pane
  - `render_final(text)` — display final answer, styled differently
  - `render_prompt()` — draw input line at bottom
  - `clear()` — reset panes between requests
- Handle terminal resize gracefully (catch `SIGWINCH`, re-render)
- Fallback: if terminal width < 80, switch to single-column linear output
- `--no-stream` flag disables streaming and uses simple line-by-line output (for piped/non-TTY contexts)

### 2.3 Token-level streaming integration

**Tasks:**
- Wire `StreamParser` to `TerminalRenderer.render_token()`
- On tool detection: call `render_tool_start()`, execute tool, call `render_tool_result()`
- On `<final>`: call `render_final()`, exit loop
- Spinner or progress indicator while waiting for first token

**Phase 2 deliverable:** Full split-pane streaming CLI. Running the agent feels like watching the model think.

---

## Phase 3 — Full Tool Set and Polish

**Goal:** Add remaining tools, harden the agent, improve UX.

### 3.1 Optional tools (`--tool-set full`)

**`search(pattern, path=".")`**
- Use `rg` if available, fallback to `str.lower() in line.lower()` scan
- Max 150 matches, clipped

**`patch_file(path, old_text, new_text)`**
- Exact match, must occur exactly once in file
- More token-efficient than rewriting the whole file for small changes
- Good for models that can identify the exact line to change

**Tasks:**
- Implement both tools with sandbox checks
- Add to `build_tools()` when `--tool-set full`
- Add to prompt schema only when enabled (keep core prompt small)

### 3.2 `/tools` slash command

Show currently active tools with schema and risk level. Useful for users experimenting with `--tool-set`.

### 3.3 Thinking block suppression

Qwen 3.x models emit `<think>...</think>` blocks before answering. These consume tokens and confuse the parser.

**Tasks:**
- Add `"think": false` to Ollama payload (already in 1.6, verify it works)
- Add `normalize()` step that strips `<think>...</think>` if present in output
- Do not show thinking blocks in the streaming UI (or optionally show them greyed out)

### 3.4 Welcome screen

```
  /\ /\
 { `---' }
 { O O }
 ~~> V <~~
  \ \|/ /
   `-----'__

  NANO CODING AGENT
  ─────────────────────────────────────────────────
  workspace   /home/user/myproject
  model       qwen2.5-coder:1.5b    branch  main
  approval    ask                   session 20260417-142310-a3f1bb
  ─────────────────────────────────────────────────
```

### 3.5 Test suite

**Tasks:**
- `tests/test_parser.py` — unit tests for all noise patterns (JSON fallback, missing tags, fenced code, wrong case, thinking blocks)
- `tests/test_tools.py` — tool execution against `tmp_path` fixtures
- `tests/test_agent.py` — full agent loop using `FakeModelClient` with scripted response sequences
- `tests/test_context.py` — history trimming, clip behavior, context overflow scenarios

### 3.6 pyproject.toml and CLI entry point

```toml
[project]
name = "nano-coding-agent"
requires-python = ">=3.10"
dependencies = []

[project.scripts]
nano-coding-agent = "nano_coding_agent:main"
```

**Phase 3 deliverable:** Polished, tested, releasable project. Full README, working test suite, clean entry point.

---

## Phase 4 — Evaluation and Model Tuning (Optional)

**Goal:** Understand which models work best and what prompt changes improve reliability.

### 4.1 Benchmark harness

A set of scripted tasks with known correct outcomes:

- "Write a function `add(a, b)` and a test for it" → check file created, test passes
- "Fix the off-by-one error in `binary_search.py`" → check corrected output
- "List all Python files and count them" → check final answer contains correct count

Run each task against multiple models and record: success rate, steps taken, tool call validity rate.

### 4.2 Prompt ablation

Test variants of the system prompt to find the minimum token count that maintains reliability:
- Full rules vs. 3-rule vs. 6-rule versions
- With/without examples
- With/without workspace context in prefix

### 4.3 Model-specific profiles

Different models may need slightly different prompting. Add optional `--profile` flag:
- `qwen2.5-coder` — default
- `tinyllama` — even shorter prompt, 3 tools only, lower max_new_tokens
- `smollm2` — moderate

---

## File structure (final)

```
nano-coding-agent/
├── nano_coding_agent.py     # Complete agent (~900 lines, no runtime deps)
├── tests/
│   ├── test_parser.py
│   ├── test_tools.py
│   ├── test_agent.py
│   └── test_context.py
├── PROJECT_PLAN.md          # This file
├── README.md
├── EXAMPLE.md               # Walkthrough of a sample session
└── pyproject.toml
```

---

## Key design constraints

**Single file.** Everything in `nano_coding_agent.py`. No imports beyond stdlib. Copy-paste deployable.

**No runtime deps.** `argparse`, `json`, `os`, `re`, `shutil`, `subprocess`, `sys`, `textwrap`, `urllib.request`, `uuid`, `datetime`, `pathlib`. That's it.

**Prompt token budget: strict.**

| Section | Budget |
|---|---|
| Static prefix (rules + tools + example) | 200 tokens |
| Workspace context (cwd + status) | 80 tokens |
| Memory block | 100 tokens |
| History (rolling) | 800 tokens |
| Current user request | 200 tokens |
| **Total** | **~1380 tokens** |

This leaves 700+ tokens for generation in a 2048-token context window (common for 1B models), or 2600+ tokens in a 4096 window.

**XML tool format only.** No JSON tool calls in the prompt or parser primary path. JSON support kept as a silent fallback only.

**Fail gracefully.** Every tool returns a string (never raises to the agent loop). Parser never raises. Ollama errors surface as readable messages, not stack traces.

---

## Milestones

| Phase | Estimated effort | Deliverable |
|---|---|---|
| Phase 1 | 2–3 days | Working batch-mode agent, no UI |
| Phase 2 | 2–3 days | Live streaming split-pane CLI |
| Phase 3 | 1–2 days | Full tool set, tests, polish |
| Phase 4 | Ongoing | Benchmarks, model profiles |
