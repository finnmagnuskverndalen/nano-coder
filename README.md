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

**A novelty coding agent for tiny language models.** Not production software — made for tinkering, learning, and fun.

---

## What it is

A minimal local agent loop that lets you give coding tasks to small Ollama models (≤2B parameters). It reads files, writes code, runs shell commands, and streams output live in a split-pane terminal UI.

It works surprisingly well for simple tasks. It will also confidently make a mess of complex ones. That's part of the charm.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download) running locally
- A small model pulled: `ollama pull qwen2.5-coder:1.5b`

No external Python dependencies.

---

## Quickstart

```bash
git clone https://github.com/finnmagnuskverndalen/nano-coder.git
cd nano-coder
python nano_coding_agent.py
```

On startup, nano-coder runs a quick preflight — it checks that Ollama is reachable and that your model is pulled, and offers to pull it inline if it's missing. Then it prompts for a workspace folder.

```
[20260417-14]  [ ASK ]  steps:5  nano-coder > write a fizzbuzz function
```

---

## Options

```
--cwd PATH          Skip the folder picker, use this directory
--model NAME        Ollama model  (default: qwen2.5-coder:1.5b)
--mode ask|auto     Ask before risky tools, or run fully autonomous
--num-ctx N         Context window tokens  (default: 16384, max: 32768)
--tool-set core|full  4 tools or 6  (default: core)
--resume latest     Resume last session
--no-stream         Disable live token streaming
--skip-preflight    Skip the startup Ollama + model check
```

---

## CLI commands

```
/mode       Toggle Ask / Auto-Accept
/steps      Show current step limit
/steps N    Set step limit  (e.g. /steps 10)
/tools      List active tools
/memory     Show session memory
/undo       Revert the last file write or patch
/reset      Clear session
/exit       Quit
```

In Ask mode, risky tool calls show a colored diff of the change before the approval prompt. Press `y` to accept, `n` to decline, or `a` to accept the rest of this turn.

---

## Models

| Model | Size |
|---|---|
| `qwen2.5-coder:1.5b` | 1.5 GB — recommended |
| `qwen2.5-coder:0.5b` | 398 MB — fastest |
| `deepseek-coder:1.3b` | 776 MB |
| `smollm2:1.7b` | 1.0 GB |

---

## License

Apache 2.0
