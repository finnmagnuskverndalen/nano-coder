#!/usr/bin/env python3
"""
nano-coding-agent — a coding agent for tiny and small LLMs (<=2B params)
with live streaming CLI and split-pane terminal UI.

No runtime dependencies beyond the Python standard library.
Requires: Python 3.10+, Ollama running locally.
"""

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

DEFAULT_MODEL = "qwen2.5-coder:1.5b"
DEFAULT_HOST  = "http://127.0.0.1:11434"
DEFAULT_MAX_STEPS      = 5
DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_TEMPERATURE    = 0.1
DEFAULT_TOP_P          = 0.9
# Ollama defaults to only 2048 tokens context.
# qwen2.5-coder (all sizes) is trained for 32K — set num_ctx accordingly.
# 16K is a good balance: fits on 8GB RAM, gives the agent a real working window.
DEFAULT_NUM_CTX = 16384

MAX_TOOL_OUTPUT        = 2000
MAX_HISTORY            = 8000
MAX_RECENT_FULL        = 4
MAX_HISTORY_COMPRESSED = 150

IGNORED_DIRS = {".git", ".nano-coding-agent", "__pycache__", ".pytest_cache",
                ".ruff_cache", ".venv", "venv", "node_modules", ".mypy_cache"}

HELP_TEXT = textwrap.dedent("""\
  Commands:
    /help          Show this help message
    /memory        Show distilled session memory
    /session       Show path to current session file
    /tools         List active tools with risk level
    /mode          Toggle between Ask and Auto-Accept mode
    /model         Show current model
    /model <name>  Switch to a different Ollama model
    /models        List locally installed Ollama models
    /steps         Show current step limit
    /steps <n>     Set step limit  e.g. /steps 10
    /undo          Revert the last file write or patch
    /reset         Clear session history and memory
    /exit          Exit the agent
""").rstrip()

# ─────────────────────────────────────────────
# ANSI helpers  —  black & white palette
# ─────────────────────────────────────────────

ESC = "\033"

def ansi(*codes): return f"{ESC}[{';'.join(str(c) for c in codes)}m"
def _reset():   return ansi(0)
def _bold():    return ansi(1)
def _dim():     return ansi(2)
def _invert():  return ansi(7)
def _ul():      return ansi(4)
def _fg(n):     return ansi(38, 5, n)   # 256-colour grey ramp 232–255
def _bg(n):     return ansi(48, 5, n)

# Palette — pure greyscale
C_BRIGHT  = _fg(255)            # white        — model output, important text
C_MID     = _fg(250)            # light grey   — normal text
C_DIM     = _fg(240)            # mid grey     — labels, dividers
C_FAINT   = _fg(235)            # dark grey    — de-emphasised
C_INVERT  = _invert()           # reverse      — mode badge, headers
C_BOLD    = _bold()
C_RESET   = _reset()
C_UL      = _ul()

# Semantic aliases
C_HEADER   = C_BRIGHT + C_BOLD
C_MODEL    = C_BRIGHT
C_TOOL_HDR = C_BRIGHT + C_BOLD
C_TOOL_OUT = C_MID
C_FINAL    = C_BRIGHT + C_BOLD
C_ERROR    = C_BRIGHT + C_BOLD
C_ACCENT   = C_BRIGHT


def strip_ansi(text: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", text)

def term_width() -> int:
    return shutil.get_terminal_size((80, 24)).columns

def is_tty() -> bool:
    return sys.stdout.isatty() and sys.stderr.isatty()


# ─────────────────────────────────────────────
# Cowsay logo
# ─────────────────────────────────────────────

def cowsay_logo(message: str = "nano-coder") -> list[str]:
    """
    Returns lines matching real cowsay output format:
     _____________
    < nano-coder  >
     -------------
            \\   ^__^
             \\  (oo)\\_______
                (__)\\       )\\/\\
                    ||----w |
                    ||     ||
    """
    inner  = max(len(message), 14)     # minimum bubble content width
    dashes = inner + 2                 # total underscores / dashes
    top    = " " + "_" * dashes
    mid    = f"< {message:<{inner}} >"  # left-aligned, padded to inner width
    bot    = " " + "-" * dashes
    return [
        top,
        mid,
        bot,
        r"        \   ^__^",
        r"         \  (oo)\_______",
        r"            (__)\       )\/\ ",
        r"                ||----w |",
        r"                ||     ||",
    ]


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def clip(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…[truncated {len(text) - limit} chars]"

def middle(text: str, limit: int) -> str:
    text = str(text).replace("\n", " ")
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    if limit == 2:
        return text[0] + "…"
    left  = (limit - 1) // 2
    right = limit - 1 - left
    return text[:left] + "…" + text[-right:]

def remember(bucket: list, item: str, limit: int) -> None:
    if not item:
        return
    if item in bucket:
        bucket.remove(item)
    bucket.append(item)
    del bucket[:-limit]


def _render_diff(old: str, new: str, label: str, max_lines: int = 20) -> str:
    """
    Return a colored unified diff for display above the approval prompt.
    Trims to max_lines with a "(N more lines)" marker so long diffs don't
    blow out the terminal. Uses the file's greyscale palette: additions
    bright, deletions dim, hunk headers faint.
    """
    old_lines = old.splitlines(keepends=True) if old else []
    new_lines = new.splitlines(keepends=True) if new else []
    raw = list(difflib.unified_diff(old_lines, new_lines, n=2, lineterm=""))
    header = f"  {C_DIM}── {label} ──{C_RESET}"
    if not raw:
        return header + f"\n  {C_DIM}(no changes){C_RESET}"

    # Skip the first two header lines (---/+++); the label replaces them.
    body = [ln.rstrip("\n") for ln in raw if not ln.startswith(("---", "+++"))]
    shown, overflow = body[:max_lines], max(0, len(body) - max_lines)

    out = [header]
    for ln in shown:
        if ln.startswith("@@"):
            out.append(f"  {C_FAINT}{ln}{C_RESET}")
        elif ln.startswith("+"):
            out.append(f"  {C_BRIGHT}{ln}{C_RESET}")
        elif ln.startswith("-"):
            out.append(f"  {C_DIM}{ln}{C_RESET}")
        else:
            out.append(f"  {C_FAINT}{ln}{C_RESET}")
    if overflow:
        out.append(f"  {C_DIM}… ({overflow} more line{'s' if overflow != 1 else ''}){C_RESET}")
    return "\n".join(out)


# ─────────────────────────────────────────────
# Preflight — Ollama reachability + model availability
# ─────────────────────────────────────────────

def preflight_check(host: str, model: str) -> bool:
    """
    Verify Ollama is reachable and the requested model is pulled.
    Prints a 2-line status and, if the model is missing, offers to
    `ollama pull` it inline. Returns True if the agent is ready to run.
    """
    host = host.rstrip("/")
    # 1. Reachability — hit /api/tags (cheap, lists models)
    try:
        req = urllib.request.Request(host + "/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"  {C_ERROR}✗{C_RESET} {C_MID}Ollama not reachable at {host}{C_RESET}")
        print(f"    {C_DIM}{exc}{C_RESET}")
        print(f"    {C_DIM}Start it with:{C_RESET} {C_BRIGHT}ollama serve{C_RESET}")
        return False
    print(f"  {C_BRIGHT}✓{C_RESET} {C_MID}Ollama reachable at {host}{C_RESET}")

    # 2. Model availability
    installed = [m["name"] for m in data.get("models", []) if "name" in m]
    if model in installed:
        print(f"  {C_BRIGHT}✓{C_RESET} {C_MID}Model {model} available{C_RESET}\n")
        return True

    print(f"  {C_ERROR}✗{C_RESET} {C_MID}Model {C_BRIGHT}{model}{C_RESET} {C_MID}not pulled{C_RESET}")
    if not is_tty():
        print(f"    {C_DIM}Pull it with:{C_RESET} {C_BRIGHT}ollama pull {model}{C_RESET}\n")
        return False
    try:
        ans = input(f"    {C_DIM}Pull it now? [Y/n]{C_RESET} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("")
        return False
    if ans in {"", "y", "yes"}:
        print(f"    {C_DIM}running:{C_RESET} ollama pull {model}")
        try:
            r = subprocess.run(["ollama", "pull", model], check=False)
        except FileNotFoundError:
            print(f"    {C_ERROR}ollama CLI not found on PATH{C_RESET}\n")
            return False
        if r.returncode == 0:
            print(f"  {C_BRIGHT}✓{C_RESET} {C_MID}Model {model} pulled{C_RESET}\n")
            return True
        print(f"    {C_ERROR}pull failed (exit {r.returncode}){C_RESET}\n")
        return False
    print(f"    {C_DIM}Pull it later with:{C_RESET} {C_BRIGHT}ollama pull {model}{C_RESET}\n")
    return False


# ─────────────────────────────────────────────
# Project folder picker
# ─────────────────────────────────────────────

def pick_workspace(initial_cwd: str | None = None) -> str:
    """
    Interactive workspace picker shown at startup when --cwd is not given.

    Shows the current directory, up to 8 subdirectories, plus options to
    enter a custom path or create a new folder. Returns the chosen absolute path.
    """
    if not is_tty():
        return initial_cwd or "."

    cwd = Path(initial_cwd or ".").resolve()
    w   = min(term_width(), 72)
    sep = C_DIM + "─" * w + C_RESET

    print(f"\n{C_BOLD}{C_BRIGHT}  Choose a project folder{C_RESET}\n{sep}\n")

    options: list[tuple[str, Path]] = [("(current dir)  " + str(cwd), cwd)]

    subdirs = sorted(
        (d for d in cwd.iterdir() if d.is_dir() and d.name not in IGNORED_DIRS),
        key=lambda d: d.name.lower(),
    )[:8]
    for d in subdirs:
        options.append((d.name + "/", d))

    for i, (label, path) in enumerate(options):
        num   = f"{C_DIM}[{i}]{C_RESET}"
        extra = f"  {C_DIM}{path}{C_RESET}" if i > 0 else ""
        print(f"  {num}  {C_MID}{label}{C_RESET}{extra}")

    enter_idx  = len(options)
    create_idx = len(options) + 1
    print(f"  {C_DIM}[{enter_idx}]{C_RESET}  {C_MID}enter a path…{C_RESET}")
    print(f"  {C_DIM}[{create_idx}]{C_RESET}  {C_MID}create a new folder{C_RESET}")
    print()

    while True:
        try:
            raw = input(f"{C_DIM}choice [{C_RESET}0{C_DIM}–{create_idx}]{C_RESET} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return str(cwd)

        if not raw:
            return str(cwd)

        if raw.isdigit():
            n = int(raw)
            if 0 <= n < len(options):
                chosen = options[n][1]
                print(f"  {C_DIM}workspace →{C_RESET} {chosen}\n")
                return str(chosen)
            elif n == enter_idx:
                raw = _ask_path("  path: ")
                if not raw:
                    continue
            elif n == create_idx:
                name = _ask_input("  new folder name: ")
                if not name:
                    continue
                new_path = cwd / name
                new_path.mkdir(parents=True, exist_ok=True)
                print(f"  {C_DIM}created →{C_RESET} {new_path}\n")
                return str(new_path)
            else:
                print(f"  {C_DIM}invalid choice{C_RESET}")
                continue

        # Treat as a typed path
        p = Path(raw).expanduser().resolve()
        if p.is_dir():
            print(f"  {C_DIM}workspace →{C_RESET} {p}\n")
            return str(p)
        if not p.exists():
            yn = _ask_input(f"  '{p}' doesn't exist — create it? [y/N] ")
            if yn.lower() in {"y", "yes"}:
                p.mkdir(parents=True, exist_ok=True)
                print(f"  {C_DIM}created →{C_RESET} {p}\n")
                return str(p)
        else:
            print(f"  {C_DIM}not a directory: {raw}{C_RESET}")


def _ask_input(prompt: str) -> str:
    try:
        return input(f"{C_DIM}{prompt}{C_RESET}").strip()
    except (EOFError, KeyboardInterrupt):
        print("")
        return ""

def _ask_path(prompt: str) -> str:
    return _ask_input(prompt)


# ─────────────────────────────────────────────
# Terminal renderer  —  B&W split-pane
# ─────────────────────────────────────────────

class TerminalRenderer:
    """
    Split-pane renderer for TTY. Falls back to plain output for pipes.

    Mode badge (header + prompt):
      [ ASK ]           — prompts before every risky tool
      [ AUTO-ACCEPT ]   — fully autonomous, no prompts
    """

    def __init__(self, model: str, use_stream: bool = True,
                 approval_policy: str = "ask"):
        self.model           = model
        self.approval_policy = approval_policy
        self.use_stream      = use_stream and is_tty()
        self._left_lines:  list[str] = []
        self._right_lines: list[str] = []
        self._lock         = threading.Lock()
        self._current_tool = ""

    # ── public API ──────────────────────────────────────────────

    def print_welcome(self, workspace_cwd: str, branch: str,
                      session_id: str, num_ctx: int = DEFAULT_NUM_CTX) -> None:
        w     = term_width()
        inner = w - 2

        def center(text: str) -> str:
            plain = strip_ansi(text)
            pad   = max(0, (inner - len(plain)) // 2)
            return " " * pad + text

        def kv(label: str, val: str, col_w: int) -> str:
            val_w = max(1, col_w - 12)
            return f"{C_DIM}{label:<11}{C_RESET}{C_MID}{middle(val, val_w)}{C_RESET}"

        col = (inner - 4) // 2

        lines = [f"{C_DIM}{'═' * w}{C_RESET}"]
        for cow_line in cowsay_logo():
            lines.append(center(f"{C_MID}{cow_line}{C_RESET}"))
        lines.append(f"{C_DIM}{'─' * w}{C_RESET}")
        lines.append("  " + kv("workspace", workspace_cwd, inner - 2))
        lines.append("  " + kv("model",   self.model,         col) + "    " + kv("branch",  branch,          col))
        lines.append("  " + kv("ctx",     f"{num_ctx:,} tokens", col) + "    " + kv("mode", self._mode_label(), col))
        lines.append("  " + kv("session", session_id[:28],    col))
        lines.append(f"{C_DIM}{'═' * w}{C_RESET}")
        print("\n".join(lines))

    def _mode_label(self) -> str:
        return "AUTO-ACCEPT (autonomous)" if self.approval_policy == "auto" else "ASK (confirm risky tools)"

    def _mode_badge(self) -> str:
        if self.approval_policy == "auto":
            return f"{C_INVERT} AUTO-ACCEPT {C_RESET}"
        return f"{C_DIM}[ ASK ]{C_RESET}"

    def set_approval(self, policy: str) -> None:
        self.approval_policy = policy

    def set_model(self, model: str) -> None:
        self.model = model

    def announce_mode_change(self, policy: str) -> None:
        if policy == "auto":
            label = "AUTO-ACCEPT — all tools run without confirmation"
        else:
            label = "ASK — risky tools require confirmation"
        print(f"\n  {C_INVERT} MODE {C_RESET}  {C_MID}{label}{C_RESET}\n")

    def announce_model_change(self, old: str, new: str) -> None:
        print(f"\n  {C_INVERT} MODEL {C_RESET}  "
              f"{C_DIM}{old}{C_RESET} → {C_BRIGHT}{new}{C_RESET}\n")

    def start_turn(self) -> None:
        with self._lock:
            self._left_lines  = []
            self._right_lines = []
            self._current_tool = ""
        if self.use_stream:
            self._render_panes()

    def write_token(self, token: str) -> None:
        with self._lock:
            if self._left_lines:
                self._left_lines[-1] += token
            else:
                self._left_lines.append(token)
            if "\n" in self._left_lines[-1]:
                parts = self._left_lines[-1].split("\n")
                self._left_lines[-1:] = parts
        if self.use_stream:
            self._render_panes()

    def write_model_line(self, text: str) -> None:
        with self._lock:
            self._left_lines.append(text)
        if self.use_stream:
            self._render_panes()

    def write_tool_header(self, name: str, args: dict) -> None:
        self._current_tool = name
        arg_str = "  ".join(
            f"{k}={middle(str(v), 30)}"
            for k, v in args.items() if k != "content"
        )
        with self._lock:
            if self._right_lines:
                self._right_lines.append("")
            self._right_lines.append(
                f"{C_TOOL_HDR}▶ {name}{C_RESET}  {C_DIM}{arg_str}{C_RESET}"
            )
        if self.use_stream:
            self._render_panes()
        else:
            print(f"\n  {C_TOOL_HDR}▶ {name}{C_RESET}  {C_DIM}{arg_str}{C_RESET}")

    def write_tool_result(self, result: str) -> None:
        with self._lock:
            for line in result.splitlines()[:30]:
                self._right_lines.append(f"  {C_TOOL_OUT}{line}{C_RESET}")
        if self.use_stream:
            self._render_panes()
        else:
            for line in result.splitlines()[:30]:
                print(f"    {C_TOOL_OUT}{line}{C_RESET}")

    def write_final(self, text: str) -> None:
        if self.use_stream:
            self._render_panes(final=text)
        else:
            print(f"\n{C_FINAL}{text}{C_RESET}\n")

    def write_error(self, text: str) -> None:
        msg = f"error: {text}"
        if self.use_stream:
            with self._lock:
                self._right_lines.append(f"{C_ERROR}{msg}{C_RESET}")
            self._render_panes()
        else:
            print(f"{C_ERROR}{msg}{C_RESET}", file=sys.stderr)

    def prompt_line(self, session_id: str, max_steps: int = DEFAULT_MAX_STEPS) -> str:
        badge = self._mode_badge()
        sid   = session_id[:10]
        return (
            f"\n{C_DIM}[{sid}]{C_RESET} "
            f"{badge} "
            f"{C_DIM}steps:{C_RESET}{C_MID}{max_steps}{C_RESET} "
            f"{C_BRIGHT}nano-coder{C_RESET} "
            f"{C_DIM}>{C_RESET} "
        )

    # ── internal rendering ───────────────────────────────────────

    def _render_panes(self, final: str = "") -> None:
        w  = term_width()
        lw = max(20, (w - 3) // 2)
        rw = w - lw - 3

        out: list[str] = []

        # Header row
        badge       = self._mode_badge()
        badge_plain = strip_ansi(badge)
        model_str   = middle(self.model, lw - len(badge_plain) - 3)
        tool_str    = middle(self._current_tool or "—", rw - 2)
        pad         = max(0, lw - len(model_str) - len(badge_plain) - 2)
        out.append(
            f"{C_DIM}{model_str}  {C_RESET}{badge}{' ' * pad}"
            f"{C_DIM}│{C_RESET} {C_TOOL_HDR}{tool_str}{C_RESET}"
        )
        out.append(f"{C_DIM}{'─' * lw}┼{'─' * (rw + 1)}{C_RESET}")

        lv = self._visible_lines(self._left_lines,  lw)
        rv = self._visible_lines(self._right_lines, rw)
        rows = max(len(lv), len(rv), 8)

        for i in range(rows):
            l = lv[i] if i < len(lv) else ""
            r = rv[i] if i < len(rv) else ""
            l_pad = max(0, lw - len(strip_ansi(l)))
            out.append(
                f"{C_MODEL}{l}{C_RESET}{' ' * l_pad}"
                f" {C_DIM}│{C_RESET} {r}"
            )

        out.append(f"{C_DIM}{'─' * lw}┴{'─' * (rw + 1)}{C_RESET}")

        if final:
            out.append(f"{C_FINAL}  {final}{C_RESET}")
            out.append("")

        sys.stdout.write(f"{ESC}[{len(out) + 2}A{ESC}[0J" + "\n".join(out) + "\n")
        sys.stdout.flush()

    def _visible_lines(self, lines: list[str], width: int) -> list[str]:
        result = []
        for line in lines:
            plain = strip_ansi(line)
            if len(plain) <= width:
                result.append(line)
            else:
                result.append(line[:width])
                result.append(f"  {C_DIM}{plain[width:width + width - 2]}{C_RESET}")
        return result[-20:]


# ─────────────────────────────────────────────
# Workspace context
# ─────────────────────────────────────────────

class WorkspaceContext:
    def __init__(self, cwd: str, repo_root: str, branch: str, status: str):
        self.cwd       = cwd
        self.repo_root = repo_root
        self.branch    = branch
        self.status    = status

    @classmethod
    def build(cls, cwd: str) -> "WorkspaceContext":
        cwd = str(Path(cwd).resolve())

        def git(args: list[str], fallback: str = "") -> str:
            try:
                r = subprocess.run(["git", *args], cwd=cwd,
                                   capture_output=True, text=True, check=True, timeout=5)
                return r.stdout.strip() or fallback
            except Exception:
                return fallback

        repo_root = git(["rev-parse", "--show-toplevel"], cwd)
        branch    = git(["branch", "--show-current"], "-")
        status    = git(["status", "--short"], "clean") or "clean"
        return cls(cwd=cwd, repo_root=repo_root, branch=branch,
                   status=clip(status, 500))

    def text(self) -> str:
        return f"cwd: {self.cwd}\ngit status:\n{self.status}"


# ─────────────────────────────────────────────
# Session store
# ─────────────────────────────────────────────

class SessionStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, sid: str) -> Path:
        return self.root / f"{sid}.json"

    def save(self, session: dict) -> Path:
        p = self.path(session["id"])
        p.write_text(json.dumps(session, indent=2), encoding="utf-8")
        return p

    def load(self, sid: str) -> dict:
        return json.loads(self.path(sid).read_text(encoding="utf-8"))

    def latest(self) -> str | None:
        files = sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime)
        return files[-1].stem if files else None


# ─────────────────────────────────────────────
# Model clients
# ─────────────────────────────────────────────

class OllamaModelClient:
    def __init__(self, model: str, host: str, temperature: float,
                 top_p: float, timeout: int, num_ctx: int = DEFAULT_NUM_CTX):
        self.model       = model
        self.host        = host.rstrip("/")
        self.temperature = temperature
        self.top_p       = top_p
        self.timeout     = timeout
        self.num_ctx     = num_ctx

    def set_model(self, model: str) -> None:
        """Switch the active model. Caller is responsible for validating it exists."""
        self.model = model

    def list_local_models(self) -> list[str]:
        """Return sorted names of models installed locally (via Ollama /api/tags)."""
        req = urllib.request.Request(
            self.host + "/api/tags",
            headers={"Content-Type": "application/json"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot reach Ollama at {self.host}") from exc
        return sorted(m["name"] for m in data.get("models", []) if "name" in m)

    def _payload(self, prompt: str, max_new_tokens: int, stream: bool) -> bytes:
        return json.dumps({
            "model": self.model, "prompt": prompt, "stream": stream,
            "raw": False, "think": False,
            "options": {
                "num_predict": max_new_tokens,
                "num_ctx":     self.num_ctx,    # override Ollama's 2048 default
                "temperature": self.temperature,
                "top_p":       self.top_p,
            },
        }).encode("utf-8")

    def complete(self, prompt: str, max_new_tokens: int) -> str:
        req = urllib.request.Request(
            self.host + "/api/generate",
            data=self._payload(prompt, max_new_tokens, False),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Ollama HTTP {exc.code}: {exc.read().decode()}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host}\n"
                "Make sure `ollama serve` is running.\n"
                f"Model: {self.model}"
            ) from exc
        if data.get("error"):
            raise RuntimeError(f"Ollama error: {data['error']}")
        return data.get("response", "")

    def stream(self, prompt: str, max_new_tokens: int):
        req = urllib.request.Request(
            self.host + "/api/generate",
            data=self._payload(prompt, max_new_tokens, True),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("error"):
                        raise RuntimeError(f"Ollama: {chunk['error']}")
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot reach Ollama at {self.host}") from exc


class FakeModelClient:
    """For tests: replays scripted responses."""
    def __init__(self, outputs: list[str]):
        self.outputs: list[str] = list(outputs)
        self.prompts: list[str] = []
        self.model = "fake"

    def set_model(self, model: str) -> None:
        self.model = model

    def list_local_models(self) -> list[str]:
        return ["fake"]

    def complete(self, prompt: str, max_new_tokens: int) -> str:
        self.prompts.append(prompt)
        if not self.outputs:
            raise RuntimeError("FakeModelClient ran out of outputs")
        return self.outputs.pop(0)

    def stream(self, prompt: str, max_new_tokens: int):
        yield from self.complete(prompt, max_new_tokens)


# ─────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────

class Parser:

    @staticmethod
    def normalize(raw: str) -> str:
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
        raw = re.sub(r"```(?:xml|json)?\s*(<(?:tool|final)[\s>])", r"\1",
                     raw, flags=re.IGNORECASE)
        raw = re.sub(r"(</(?:tool|final)>)\s*```", r"\1", raw, flags=re.IGNORECASE)
        raw = re.sub(
            r"<(/?)(TOOL|FINAL|CONTENT|OLD_TEXT|NEW_TEXT)(\b[^>]*)>",
            lambda m: f"<{m.group(1)}{m.group(2).lower()}{m.group(3)}>", raw,
        )
        return raw.strip()

    @staticmethod
    def parse(raw: str) -> tuple[str, dict | str]:
        raw = Parser.normalize(raw)

        xml = Parser._parse_xml_tool(raw)
        if xml is not None:
            name, args = xml
            if name:
                return "tool", {"name": name, "args": args}

        if "<tool>" in raw:
            body = Parser._extract_tag(raw, "tool")
            if body is not None:
                try:
                    p = json.loads(body)
                    if isinstance(p, dict) and p.get("name"):
                        return "tool", {"name": p["name"], "args": p.get("args") or {}}
                except (json.JSONDecodeError, KeyError):
                    pass

        if "<final>" in raw:
            text = Parser._extract_tag(raw, "final")
            if text and text.strip():
                return "final", text.strip()
            return "retry", "Empty <final> — put your answer inside <final>…</final>"

        raw = raw.strip()
        if raw:
            return "final", raw
        return "retry", (
            "No valid output. Reply with:\n"
            '  <tool name="tool_name" arg="val"></tool>\n'
            "  or  <final>your answer</final>"
        )

    @staticmethod
    def _parse_xml_tool(raw: str) -> tuple[str, dict] | None:
        m = re.search(r"<tool\b([^>]*)>(.*?)</tool>", raw, re.DOTALL | re.IGNORECASE)
        if not m:
            m = re.search(r"<tool\b([^>]*?)(?:/>|>([^<]*)$)", raw, re.DOTALL | re.IGNORECASE)
            if not m:
                return None
            attr_str, body = m.group(1), (m.group(2) or "").strip()
        else:
            attr_str, body = m.group(1), m.group(2)

        attrs = Parser._parse_attrs(attr_str)
        name  = str(attrs.pop("name", "")).strip()
        args  = dict(attrs)

        for key in ("content", "old_text", "new_text", "command", "task", "pattern"):
            sub = Parser._extract_tag(body, key)
            if sub is not None:
                args[key] = sub

        if name == "write_file" and "content" not in args:
            bs = body.strip("\n")
            if bs:
                args["content"] = bs

        return name, args

    @staticmethod
    def _parse_attrs(text: str) -> dict:
        attrs = {}
        for m in re.finditer(r"""(\w+)\s*=\s*(?:"([^"]*)"|'([^']*)')""", text):
            attrs[m.group(1)] = m.group(2) if m.group(2) is not None else m.group(3)
        return attrs

    @staticmethod
    def _extract_tag(text: str, tag: str) -> str | None:
        s = text.find(f"<{tag}>")
        if s == -1:
            return None
        s += len(tag) + 2
        e = text.find(f"</{tag}>", s)
        return text[s:] if e == -1 else text[s:e]


# ─────────────────────────────────────────────
# Stream parser
# ─────────────────────────────────────────────

class StreamParser:
    def __init__(self, on_tool: callable, on_final: callable, on_token: callable):
        self.on_tool  = on_tool
        self.on_final = on_final
        self.on_token = on_token
        self._buffer  = ""
        self._done    = False

    @property
    def done(self) -> bool:
        return self._done

    def feed(self, token: str) -> None:
        if self._done:
            return
        self._buffer += token
        self.on_token(token)
        self._try_detect()

    def flush(self) -> None:
        if self._done or not self._buffer.strip():
            return
        kind, payload = Parser.parse(self._buffer)
        if kind == "tool":
            self.on_tool(payload["name"], payload["args"])
        elif kind == "final":
            self.on_final(payload)
        else:
            self.on_final(self._buffer.strip())
        self._done = True

    def _try_detect(self) -> None:
        buf = self._buffer
        if "</tool>" in buf.lower():
            kind, payload = Parser.parse(buf)
            if kind == "tool":
                self.on_tool(payload["name"], payload["args"])
                self._buffer = ""
                self._done   = True
                return
        if "</final>" in buf.lower():
            text = Parser._extract_tag(buf, "final")
            if text and text.strip():
                self.on_final(text.strip())
                self._done = True


# ─────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────

class NanoAgent:

    def __init__(
        self,
        model_client,
        workspace: WorkspaceContext,
        session_store: SessionStore,
        renderer: TerminalRenderer,
        session: dict | None = None,
        approval_policy: str = "ask",
        max_steps: int = DEFAULT_MAX_STEPS,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        tool_set: str = "core",
        use_stream: bool = True,
    ):
        self.model_client    = model_client
        self.workspace       = workspace
        self.root            = Path(workspace.repo_root)
        self.session_store   = session_store
        self.renderer        = renderer
        self.approval_policy = approval_policy
        self.max_steps       = max_steps
        self.max_new_tokens  = max_new_tokens
        self.tool_set        = tool_set
        self.use_stream      = use_stream and is_tty()

        self.session = session or {
            "id":             _new_session_id(),
            "created_at":     now_iso(),
            "workspace_root": str(workspace.repo_root),
            "history":        [],
            "memory":         {"task": "", "files": [], "notes": []},
        }
        # Single-slot undo: captures pre-state of the last file-mutating tool.
        # Shape: {"path": Path, "prev": str | None, "action": "write" | "patch"}
        # prev is None when the file did not exist before the write.
        self._undo_slot: dict | None = None
        # Resets at the start of each ask(). When True, skip approval prompts
        # for the rest of this turn (user answered 'a' to a previous prompt).
        self._approve_all_turn: bool = False
        self.tools        = self._build_tools()
        self.session_path = self.session_store.save(self.session)

    @classmethod
    def from_session(cls, model_client, workspace, session_store,
                     renderer, session_id: str, **kwargs) -> "NanoAgent":
        return cls(model_client=model_client, workspace=workspace,
                   session_store=session_store, renderer=renderer,
                   session=session_store.load(session_id), **kwargs)

    # ── public ───────────────────────────────────────────────────

    def ask(self, user_message: str) -> str:
        memory = self.session["memory"]
        if not memory["task"]:
            memory["task"] = clip(user_message.strip(), 300)
        # Fresh approval decisions each turn — 'a' (approve-all) is turn-scoped.
        self._approve_all_turn = False

        self._record({"role": "user", "content": user_message, "ts": now_iso()})
        self.renderer.start_turn()

        tool_steps   = 0
        attempts     = 0
        max_attempts = max(self.max_steps * 3, self.max_steps + 6)

        while tool_steps < self.max_steps and attempts < max_attempts:
            attempts += 1

            if self.use_stream:
                raw, tool_fired = self._run_streaming_step(user_message)
                if tool_fired:
                    tool_steps += 1
                    continue
                kind, payload = Parser.parse(raw)
            else:
                raw  = self.model_client.complete(
                    self._build_prompt(user_message), self.max_new_tokens
                )
                self.renderer.write_model_line(raw)
                kind, payload = Parser.parse(raw)

            if kind == "tool":
                tool_steps += 1
                result = self._run_tool(payload["name"], payload["args"])
                self._record({"role": "tool", "name": payload["name"],
                              "args": payload["args"], "content": result, "ts": now_iso()})
                self._note_tool(payload["name"], payload["args"], result)
                continue

            if kind == "final":
                final = str(payload).strip()
                self._record({"role": "assistant", "content": final, "ts": now_iso()})
                remember(memory["notes"], clip(final, 200), 4)
                self.renderer.write_final(final)
                return final

            self._record({"role": "assistant", "content": str(payload), "ts": now_iso()})

        final = ("Reached step limit." if tool_steps >= self.max_steps
                 else "Too many malformed responses.")
        self._record({"role": "assistant", "content": final, "ts": now_iso()})
        self.renderer.write_final(final)
        return final

    def set_approval(self, policy: str) -> None:
        self.approval_policy = policy
        self.renderer.set_approval(policy)

    def set_max_steps(self, n: int) -> None:
        self.max_steps = max(1, n)

    def set_model(self, model: str) -> None:
        """Switch the underlying model on both the client and the renderer header."""
        if not hasattr(self.model_client, "set_model"):
            raise RuntimeError("This model client does not support switching.")
        self.model_client.set_model(model)
        self.renderer.set_model(model)

    def list_local_models(self) -> list[str]:
        if not hasattr(self.model_client, "list_local_models"):
            raise RuntimeError("This model client does not support listing models.")
        return self.model_client.list_local_models()

    def reset(self) -> None:
        self.session["history"] = []
        self.session["memory"]  = {"task": "", "files": [], "notes": []}
        self._undo_slot = None
        self.session_store.save(self.session)

    def undo_last(self) -> str:
        """Revert the last file-mutating tool call (write_file or patch_file)."""
        slot = self._undo_slot
        if not slot:
            return "nothing to undo"
        path: Path = slot["path"]
        prev = slot["prev"]
        rel = path.relative_to(self.root)
        try:
            if prev is None:
                # File did not exist before — remove it.
                if path.exists():
                    path.unlink()
                msg = f"reverted {slot['action']}: removed {rel} (did not exist before)"
            else:
                path.write_text(prev, encoding="utf-8")
                msg = f"reverted {slot['action']}: restored {rel} ({len(prev)} chars)"
        except OSError as exc:
            return f"undo failed: {exc}"
        self._undo_slot = None
        return msg

    def memory_text(self) -> str:
        m     = self.session["memory"]
        notes = "\n".join(f"  - {n}" for n in m["notes"]) or "  - none"
        return (f"task:  {m['task'] or '—'}\n"
                f"files: {', '.join(m['files']) or '—'}\n"
                f"notes:\n{notes}")

    def tools_text(self) -> str:
        lines = []
        for name, tool in self.tools.items():
            risk = "⚠  approval" if tool["risky"] else "   safe"
            lines.append(f"  {C_BOLD}{name:<16}{C_RESET} {risk}  {C_DIM}{tool['desc']}{C_RESET}")
        return "\n".join(lines)

    # ── prompt ───────────────────────────────────────────────────

    def _build_prefix(self) -> str:
        tool_lines = [
            f'  <tool name="{n}" ' + "  ".join(f'{k}="{v}"' for k, v in t["schema"].items()) + "></tool>"
            for n, t in self.tools.items()
        ]
        return textwrap.dedent(f"""\
            You are nano-coder, a coding agent running on Ollama.
            Rules:
            - Use tools to inspect the workspace before writing code.
            - Emit exactly one tool call OR one final answer per response.
            - Tool call format (XML only):
              <tool name="TOOL_NAME" arg1="val1"></tool>
              For file content use a <content> tag:
              <tool name="write_file" path="x.py"><content>
              code here
              </content></tool>
            - Final answer: <final>your answer</final>
            - Keep answers short and concrete. Never invent tool results.
            Available tools:
            {chr(10).join(tool_lines)}
            Example:
              <tool name="read_file" path="main.py" start="1" end="40"></tool>
              <final>Done. I updated main.py with the fix.</final>
        """).strip()

    def _build_prompt(self, user_message: str) -> str:
        m = self.session["memory"]
        mem = f"task: {m['task'] or '—'}  files: {', '.join(m['files'][-4:]) or '—'}"
        return (f"{self._build_prefix()}\n\n"
                f"Workspace:\n{self.workspace.text()}\n\n"
                f"Memory:\n{mem}\n\n"
                f"History:\n{self._history_text()}\n\n"
                f"Task: {user_message}")

    def _history_text(self) -> str:
        history = self.session["history"]
        if not history:
            return "(empty)"
        lines      = []
        seen_reads: set[str] = set()
        recent_idx = max(0, len(history) - MAX_RECENT_FULL)
        for idx, item in enumerate(history):
            recent = idx >= recent_idx
            if item["role"] == "tool":
                if item["name"] == "read_file" and not recent:
                    key = str(item["args"].get("path", ""))
                    if key in seen_reads:
                        continue
                    seen_reads.add(key)
                limit = 800 if recent else MAX_HISTORY_COMPRESSED
                lines.append(f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True)}")
                lines.append(clip(item["content"], limit))
            else:
                limit = 800 if recent else MAX_HISTORY_COMPRESSED
                lines.append(f"[{item['role']}] {clip(item['content'], limit)}")
        return clip("\n".join(lines), MAX_HISTORY)

    # ── streaming ────────────────────────────────────────────────

    def _run_streaming_step(self, user_message: str) -> tuple[str, bool]:
        prompt     = self._build_prompt(user_message)
        full_raw   = ""
        tool_fired = False

        def on_token(token: str) -> None:
            nonlocal full_raw
            full_raw += token
            self.renderer.write_token(token)

        def on_tool(name: str, args: dict) -> str:
            nonlocal tool_fired
            tool_fired = True
            result = self._run_tool(name, args)
            self._record({"role": "tool", "name": name, "args": args,
                          "content": result, "ts": now_iso()})
            self._note_tool(name, args, result)
            return result

        sp = StreamParser(on_tool=on_tool, on_final=lambda t: None, on_token=on_token)
        try:
            for token in self.model_client.stream(prompt, self.max_new_tokens):
                sp.feed(token)
                if sp.done:
                    break
            if not sp.done:
                sp.flush()
        except RuntimeError as exc:
            self.renderer.write_error(str(exc))

        return full_raw, tool_fired

    # ── tools ────────────────────────────────────────────────────

    def _build_tools(self) -> dict:
        tools = {
            "list_files": {
                "schema": {"path": "."},
                "risky": False, "desc": "List files in the workspace.",
                "run": self._tool_list_files,
            },
            "read_file": {
                "schema": {"path": "REQUIRED", "start": "1", "end": "60"},
                "risky": False, "desc": "Read a file by line range.",
                "run": self._tool_read_file,
            },
            "write_file": {
                "schema": {"path": "REQUIRED"},
                "risky": True, "desc": "Write or overwrite a file.",
                "run": self._tool_write_file,
            },
            "run_shell": {
                "schema": {"command": "REQUIRED", "timeout": "20"},
                "risky": True, "desc": "Run a shell command.",
                "run": self._tool_run_shell,
            },
        }
        if self.tool_set == "full":
            tools["search"] = {
                "schema": {"pattern": "REQUIRED", "path": "."},
                "risky": False, "desc": "Search workspace with ripgrep or fallback.",
                "run": self._tool_search,
            }
            tools["patch_file"] = {
                "schema": {"path": "REQUIRED", "old_text": "REQUIRED", "new_text": "REQUIRED"},
                "risky": True, "desc": "Replace one exact block in a file.",
                "run": self._tool_patch_file,
            }
        return tools

    def _run_tool(self, name: str, args: dict) -> str:
        self.renderer.write_tool_header(name, args)
        tool = self.tools.get(name)
        if not tool:
            r = f"error: unknown tool '{name}'. Available: {', '.join(self.tools)}"
            self.renderer.write_tool_result(r); return r
        try:
            self._validate_tool(name, args)
        except ValueError as exc:
            r = f"error: bad args for {name}: {exc}"
            self.renderer.write_tool_result(r); return r
        if self._is_repeated(name, args):
            path_hint = f" '{args.get('path')}'" if args.get("path") else ""
            r = (f"error: {name}{path_hint} was already called with the same arguments. "
                 f"The file is already written. Stop repeating — use <final> to report what you did, "
                 f"or call a different tool.")
            self.renderer.write_tool_result(r); return r
        if tool["risky"] and not self._approve(name, args):
            r = f"error: approval denied for {name}"
            self.renderer.write_tool_result(r); return r
        try:
            r = clip(tool["run"](args))
        except Exception as exc:
            r = f"error: {name} failed: {exc}"
        self.renderer.write_tool_result(r)
        return r

    def _validate_tool(self, name: str, args: dict) -> None:
        args = args or {}
        if name == "list_files":
            if not self._safe_path(args.get("path", ".")).is_dir():
                raise ValueError("path is not a directory")
        elif name == "read_file":
            if not args.get("path"):
                raise ValueError("path required")
            if not self._safe_path(args["path"]).exists():
                raise ValueError(f"file not found: {args['path']}")
            if int(args.get("start", 1)) < 1 or int(args.get("end", 60)) < int(args.get("start", 1)):
                raise ValueError("invalid line range")
        elif name == "write_file":
            if not args.get("path"):
                raise ValueError("path required")
            if "content" not in args:
                raise ValueError("content required")
            if self._safe_path(args["path"]).is_dir():
                raise ValueError("path is a directory")
        elif name == "run_shell":
            if not str(args.get("command", "")).strip():
                raise ValueError("command required")
            t = int(args.get("timeout", 20))
            if not (1 <= t <= 120):
                raise ValueError("timeout must be 1–120")
        elif name == "search":
            if not str(args.get("pattern", "")).strip():
                raise ValueError("pattern required")
        elif name == "patch_file":
            if not args.get("path"):
                raise ValueError("path required")
            p = self._safe_path(args["path"])
            if not p.is_file():
                raise ValueError("file not found")
            old = str(args.get("old_text", ""))
            if not old:
                raise ValueError("old_text required")
            if "new_text" not in args:
                raise ValueError("new_text required")
            count = p.read_text(encoding="utf-8").count(old)
            if count != 1:
                raise ValueError(f"old_text must appear exactly once (found {count})")

    def _is_repeated(self, name: str, args: dict) -> bool:
        evts = [e for e in self.session["history"] if e["role"] == "tool"]
        if not evts:
            return False
        # Case 1: exact same call twice in a row
        if len(evts) >= 2 and all(e["name"] == name and e["args"] == args for e in evts[-2:]):
            return True
        # Case 2: same tool on same path called 2+ times in last 4 tool events
        # (catches write_file loops where content differs slightly each time)
        if name in {"write_file", "patch_file", "read_file"} and args.get("path"):
            path = args["path"]
            recent_same_path = [
                e for e in evts[-4:]
                if e["name"] == name and e["args"].get("path") == path
            ]
            if len(recent_same_path) >= 2:
                return True
        return False

    def _approve(self, name: str, args: dict) -> bool:
        if self.approval_policy == "auto":  return True
        if self.approval_policy == "never": return False
        if self._approve_all_turn:          return True

        # Render a preview appropriate for the tool.
        preview = self._build_approval_preview(name, args)
        if preview:
            print(preview)

        try:
            ans = input(
                f"  {C_INVERT} APPROVE {C_RESET} "
                f"{C_BOLD}{name}{C_RESET}  "
                f"{C_DIM}[y=yes / n=no / a=yes to all this turn]{C_RESET} "
            )
        except EOFError:
            return False
        choice = ans.strip().lower()
        if choice in {"a", "all"}:
            self._approve_all_turn = True
            return True
        return choice in {"y", "yes"}

    def _build_approval_preview(self, name: str, args: dict) -> str:
        """
        Return an ANSI-colored preview shown above the approval prompt.
        Diffs for write_file / patch_file; command text for run_shell.
        """
        if name == "write_file" and args.get("path") and "content" in args:
            try:
                path = self._safe_path(args["path"])
            except ValueError as exc:
                return f"\n  {C_ERROR}{exc}{C_RESET}\n"
            new = str(args["content"])
            old = path.read_text(encoding="utf-8") if path.is_file() else ""
            rel = path.relative_to(self.root) if path.is_absolute() else args["path"]
            label = f"write_file {rel}" + ("" if old else "  (new file)")
            return "\n" + _render_diff(old, new, label) + "\n"

        if name == "patch_file" and args.get("path") and "new_text" in args:
            try:
                path = self._safe_path(args["path"])
            except ValueError as exc:
                return f"\n  {C_ERROR}{exc}{C_RESET}\n"
            if not path.is_file():
                return ""
            old = path.read_text(encoding="utf-8")
            new = old.replace(str(args.get("old_text", "")), str(args["new_text"]), 1)
            rel = path.relative_to(self.root)
            return "\n" + _render_diff(old, new, f"patch_file {rel}") + "\n"

        if name == "run_shell" and args.get("command"):
            cmd = str(args["command"]).strip()
            return f"\n  {C_DIM}$ {C_RESET}{C_BRIGHT}{cmd}{C_RESET}\n"

        # Fallback — compact JSON.
        preview = json.dumps(args, ensure_ascii=False)[:200]
        return f"\n  {C_DIM}{preview}{C_RESET}\n"

    def _safe_path(self, raw: str) -> Path:
        p = Path(raw)
        p = p if p.is_absolute() else self.root / p
        resolved = p.resolve()
        if not str(resolved).startswith(str(self.root)):
            raise ValueError(f"path escapes workspace: {raw}")
        return resolved

    def _tool_list_files(self, args: dict) -> str:
        path = self._safe_path(args.get("path", "."))
        entries = sorted((e for e in path.iterdir() if e.name not in IGNORED_DIRS),
                         key=lambda e: (e.is_file(), e.name.lower()))
        return "\n".join(
            f"{'[F]' if e.is_file() else '[D]'} {e.relative_to(self.root)}"
            for e in entries[:150]
        ) or "(empty)"

    def _tool_read_file(self, args: dict) -> str:
        path  = self._safe_path(args["path"])
        start = int(args.get("start", 1))
        end   = int(args.get("end",   60))
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        body  = "\n".join(f"{n:>4}: {l}"
                          for n, l in enumerate(lines[start - 1:end], start=start))
        return f"# {path.relative_to(self.root)}\n{body}"

    def _tool_write_file(self, args: dict) -> str:
        path = self._safe_path(args["path"])
        content = str(args["content"])
        # Capture pre-state for /undo before mutating anything.
        prev = path.read_text(encoding="utf-8") if path.is_file() else None
        self._undo_slot = {"path": path, "prev": prev, "action": "write"}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        rel = path.relative_to(self.root)
        return f"OK: wrote {rel} ({len(content)} chars). File is saved. Do not write {rel} again."

    def _tool_run_shell(self, args: dict) -> str:
        r = subprocess.run(str(args["command"]).strip(), cwd=self.root, shell=True,
                           capture_output=True, text=True,
                           timeout=int(args.get("timeout", 20)))
        return (f"exit: {r.returncode}\nstdout:\n{clip(r.stdout.strip() or '(empty)', 800)}"
                f"\nstderr:\n{clip(r.stderr.strip() or '(empty)', 400)}")

    def _tool_search(self, args: dict) -> str:
        pattern = str(args["pattern"]).strip()
        path    = self._safe_path(args.get("path", "."))
        if shutil.which("rg"):
            r = subprocess.run(
                ["rg", "-n", "--smart-case", "--max-count", "150", pattern, str(path)],
                cwd=self.root, capture_output=True, text=True)
            return r.stdout.strip() or r.stderr.strip() or "(no matches)"
        matches = []
        files = [path] if path.is_file() else [
            f for f in path.rglob("*")
            if f.is_file() and not any(p in IGNORED_DIRS for p in f.relative_to(self.root).parts)
        ]
        for f in files:
            for n, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if pattern.lower() in line.lower():
                    matches.append(f"{f.relative_to(self.root)}:{n}:{line}")
                    if len(matches) >= 150:
                        return "\n".join(matches)
        return "\n".join(matches) or "(no matches)"

    def _tool_patch_file(self, args: dict) -> str:
        path = self._safe_path(args["path"])
        text = path.read_text(encoding="utf-8")
        # Capture pre-state for /undo before mutating.
        self._undo_slot = {"path": path, "prev": text, "action": "patch"}
        path.write_text(text.replace(str(args["old_text"]), str(args["new_text"]), 1),
                        encoding="utf-8")
        rel = path.relative_to(self.root)
        return f"OK: patched {rel}. File is saved. Do not patch the same block again."

    def _record(self, item: dict) -> None:
        self.session["history"].append(item)
        self.session_path = self.session_store.save(self.session)

    def _note_tool(self, name: str, args: dict, result: str) -> None:
        m = self.session["memory"]
        if name in {"read_file", "write_file", "patch_file"} and args.get("path"):
            remember(m["files"], str(args["path"]), 8)
        remember(m["notes"], f"{name}: {clip(str(result).replace(chr(10), ' '), 150)}", 4)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


# ─────────────────────────────────────────────
# REPL command handlers
# ─────────────────────────────────────────────

def _cmd_models(agent: "NanoAgent") -> None:
    """List locally installed Ollama models, marking the current one."""
    try:
        models = agent.list_local_models()
    except RuntimeError as exc:
        print(f"  {C_ERROR}{exc}{C_RESET}")
        return
    if not models:
        print(f"  {C_DIM}no models installed — try:{C_RESET} "
              f"{C_BRIGHT}ollama pull qwen2.5-coder:1.5b{C_RESET}")
        return
    current = agent.model_client.model
    print(f"\n  {C_BOLD}Installed Ollama models:{C_RESET}")
    for m in models:
        marker = f"{C_BRIGHT}●{C_RESET}" if m == current else " "
        label  = f"{C_BRIGHT}{m}{C_RESET}" if m == current else f"{C_MID}{m}{C_RESET}"
        print(f"   {marker} {label}")
    print(f"\n  {C_DIM}switch with:{C_RESET} /model <name>\n")


def _cmd_model(agent: "NanoAgent", user_input: str) -> None:
    """`/model` shows current; `/model <name>` switches."""
    parts = user_input.split(maxsplit=1)
    if len(parts) == 1:
        print(f"  {C_DIM}current model:{C_RESET} "
              f"{C_BRIGHT}{agent.model_client.model}{C_RESET}")
        print(f"  {C_DIM}usage:{C_RESET} /model <name>   "
              f"{C_DIM}(try /models to list installed){C_RESET}")
        return

    new_model = parts[1].strip()
    if not new_model:
        print(f"  {C_DIM}usage: /model <name>{C_RESET}")
        return

    old = agent.model_client.model
    if new_model == old:
        print(f"  {C_DIM}already using{C_RESET} {C_BRIGHT}{new_model}{C_RESET}")
        return

    # Verify the model is installed locally before switching
    try:
        available = agent.list_local_models()
    except RuntimeError as exc:
        print(f"  {C_ERROR}{exc}{C_RESET}")
        return

    if new_model not in available:
        print(f"  {C_ERROR}model '{new_model}' not found locally{C_RESET}")
        print(f"  {C_DIM}pull it first:{C_RESET} "
              f"{C_BRIGHT}ollama pull {new_model}{C_RESET}")
        if available:
            shown = ", ".join(available[:6])
            more  = f" (+{len(available) - 6} more)" if len(available) > 6 else ""
            print(f"  {C_DIM}or pick from:{C_RESET} {shown}{more}")
        return

    try:
        agent.set_model(new_model)
    except RuntimeError as exc:
        print(f"  {C_ERROR}{exc}{C_RESET}")
        return
    agent.renderer.announce_model_change(old, new_model)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nano-coder",
        description="Coding agent for tiny and small LLMs (<=2B params).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("prompt",           nargs="*",  help="One-shot prompt (optional).")
    p.add_argument("--cwd",            default=None,          help="Workspace dir (skips picker).")
    p.add_argument("--model",          default=DEFAULT_MODEL, help="Ollama model name.")
    p.add_argument("--host",           default=DEFAULT_HOST,  help="Ollama server URL.")
    p.add_argument("--ollama-timeout", type=int,   default=300,  help="Ollama timeout (s).")
    p.add_argument("--resume",         default=None,             help="Session ID or 'latest'.")
    p.add_argument("--mode",           choices=("ask", "auto"),  default="ask",
                   help="ask = confirm risky tools  |  auto = fully autonomous.")
    p.add_argument("--max-steps",      type=int,   default=DEFAULT_MAX_STEPS)
    p.add_argument("--max-new-tokens", type=int,   default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--num-ctx",        type=int,   default=DEFAULT_NUM_CTX,
                   help="Context window tokens sent to Ollama (model max: 32K for qwen2.5-coder).")
    p.add_argument("--temperature",    type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-p",          type=float, default=DEFAULT_TOP_P)
    p.add_argument("--tool-set",       choices=("core", "full"), default="core")
    p.add_argument("--no-stream",      action="store_true", help="Disable live streaming.")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Skip the Ollama + model availability check.")
    return p


def _mode_to_policy(mode: str) -> str:
    return "auto" if mode == "auto" else "ask"


def build_agent(args: argparse.Namespace, cwd: str,
                renderer: TerminalRenderer) -> NanoAgent:
    workspace = WorkspaceContext.build(cwd)
    store     = SessionStore(Path(workspace.repo_root) / ".nano-coding-agent" / "sessions")
    client    = OllamaModelClient(
        model=args.model, host=args.host,
        temperature=args.temperature, top_p=args.top_p,
        timeout=args.ollama_timeout, num_ctx=args.num_ctx,
    )
    policy = _mode_to_policy(args.mode)
    kwargs = dict(model_client=client, workspace=workspace, session_store=store,
                  renderer=renderer, approval_policy=policy,
                  max_steps=args.max_steps, max_new_tokens=args.max_new_tokens,
                  tool_set=args.tool_set, use_stream=not args.no_stream)
    sid = args.resume
    if sid == "latest":
        sid = store.latest()
    if sid:
        return NanoAgent.from_session(session_id=sid, **kwargs)
    return NanoAgent(**kwargs)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # Preflight: check Ollama + model before asking the user anything.
    if not args.skip_preflight:
        if not preflight_check(args.host, args.model):
            return 1

    # Workspace: interactive picker unless --cwd given
    cwd = str(Path(args.cwd).expanduser().resolve()) if args.cwd else pick_workspace()

    policy   = _mode_to_policy(args.mode)
    renderer = TerminalRenderer(model=args.model, use_stream=not args.no_stream,
                                approval_policy=policy)
    agent    = build_agent(args, cwd, renderer)
    ws = WorkspaceContext.build(cwd)

    renderer.print_welcome(
        workspace_cwd=ws.cwd,
        branch=ws.branch,
        session_id=agent.session["id"],
        num_ctx=args.num_ctx,
    )

    # One-shot
    if args.prompt:
        prompt = " ".join(args.prompt).strip()
        if prompt:
            try:
                agent.ask(prompt)
            except RuntimeError as exc:
                print(f"{C_ERROR}{exc}{C_RESET}", file=sys.stderr)
                return 1
        return 0

    # Interactive REPL
    while True:
        try:
            raw = input(renderer.prompt_line(agent.session["id"], agent.max_steps))
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0

        user_input = raw.strip()
        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            return 0
        if user_input == "/help":
            print(HELP_TEXT); continue
        if user_input == "/memory":
            print(agent.memory_text()); continue
        if user_input == "/session":
            print(agent.session_path); continue
        if user_input == "/tools":
            print(agent.tools_text()); continue
        if user_input == "/reset":
            agent.reset()
            print(f"  {C_DIM}Session cleared.{C_RESET}"); continue
        if user_input == "/undo":
            msg = agent.undo_last()
            print(f"  {C_DIM}{msg}{C_RESET}"); continue
        if user_input == "/mode":
            new_policy = "auto" if agent.approval_policy == "ask" else "ask"
            agent.set_approval(new_policy)
            renderer.announce_mode_change(new_policy); continue
        if user_input == "/models":
            _cmd_models(agent); continue
        if user_input == "/model" or user_input.startswith("/model "):
            _cmd_model(agent, user_input); continue
        if user_input == "/steps" or user_input.startswith("/steps "):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 1:
                print(f"  {C_DIM}step limit:{C_RESET} {C_BRIGHT}{agent.max_steps}{C_RESET}"
                      f"  {C_DIM}(set with /steps <n>){C_RESET}")
            else:
                raw_n = parts[1].strip()
                try:
                    n = int(raw_n)
                    if n < 1:
                        raise ValueError
                    agent.set_max_steps(n)
                    print(f"  {C_DIM}step limit set to{C_RESET} {C_BRIGHT}{n}{C_RESET}")
                except ValueError:
                    print(f"  {C_DIM}usage: /steps <number>  e.g. /steps 10{C_RESET}")
            continue
        if user_input.startswith("/"):
            print(f"  {C_DIM}Unknown command. /help for list.{C_RESET}"); continue

        try:
            agent.ask(user_input)
        except RuntimeError as exc:
            print(f"\n{C_ERROR}{exc}{C_RESET}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
