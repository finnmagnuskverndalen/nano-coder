#!/usr/bin/env python3
"""
nano-coding-agent — a coding agent for tiny and small LLMs (<=2B params)
with live streaming CLI and split-pane terminal UI.

No runtime dependencies beyond the Python standard library.
Requires: Python 3.10+, Ollama running locally.
"""

import argparse
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
DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MAX_STEPS = 5
DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.9

MAX_TOOL_OUTPUT = 2000   # chars — kept small for nano models
MAX_HISTORY = 8000       # chars — total rolling history budget
MAX_RECENT_FULL = 4      # last N turns shown at full length
MAX_HISTORY_COMPRESSED = 150  # chars per older turn

IGNORED_DIRS = {".git", ".nano-coding-agent", "__pycache__", ".pytest_cache",
                ".ruff_cache", ".venv", "venv", "node_modules", ".mypy_cache"}

SLASH_COMMANDS = {"/help", "/memory", "/session", "/tools", "/reset", "/exit", "/quit"}

WELCOME_ART = [
    "  /\\ /\\  ",
    " { `---' }",
    " { O   O }",
    " ~~> V <~~",
    "  \\ \\|/ / ",
    "   `-----'",
]

HELP_TEXT = textwrap.dedent("""\
  Commands:
    /help     Show this help message
    /memory   Show distilled session memory
    /session  Show path to current session file
    /tools    List available tools
    /reset    Clear session history and memory
    /exit     Exit the agent
""").rstrip()

# ─────────────────────────────────────────────
# ANSI helpers
# ─────────────────────────────────────────────

ESC = "\033"

def ansi(*codes): return f"{ESC}[{';'.join(str(c) for c in codes)}m"
def reset():      return ansi(0)
def bold():       return ansi(1)
def dim():        return ansi(2)
def fg(r, g, b):  return ansi(38, 2, r, g, b)
def bg(r, g, b):  return ansi(48, 2, r, g, b)

# Palette
C_HEADER   = fg(130, 200, 255)   # soft blue — header bar
C_MODEL    = fg(180, 230, 180)   # soft green — model output
C_TOOL_HDR = fg(255, 200, 100)   # amber — tool name
C_TOOL_OUT = fg(200, 200, 200)   # light grey — tool output
C_FINAL    = fg(150, 255, 200)   # mint — final answer
C_ERROR    = fg(255, 110, 110)   # red — errors
C_DIM      = fg(100, 100, 110)   # muted — dividers, prompts
C_ACCENT   = fg(255, 160, 80)    # orange — accents
C_THINK    = fg(120, 100, 160)   # purple — thinking blocks (if shown)

def strip_ansi(text: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", text)

def term_width() -> int:
    return shutil.get_terminal_size((80, 24)).columns

def is_tty() -> bool:
    return sys.stdout.isatty() and sys.stderr.isatty()

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
    # "…" is 1 char; left + 1 + right = limit
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

# ─────────────────────────────────────────────
# Terminal renderer (split-pane + simple fallback)
# ─────────────────────────────────────────────

class TerminalRenderer:
    """
    Split-pane renderer for TTY. Falls back to simple line output for pipes.

    Layout (TTY mode):
      ┌─ header ───────────────────────────────────────────────────────┐
      │ left pane (model tokens)    │ right pane (tool results)        │
      ├─────────────────────────────┴──────────────────────────────────┤
      │ prompt line                                                     │
      └─────────────────────────────────────────────────────────────────┘

    In non-TTY / --no-stream mode: plain sequential output, no ANSI.
    """

    def __init__(self, model: str, use_stream: bool = True):
        self.model = model
        self.use_stream = use_stream and is_tty()
        self._left_lines: list[str] = []
        self._right_lines: list[str] = []
        self._lock = threading.Lock()
        self._current_tool: str = ""

    # ── public API ──────────────────────────────────────────────

    def print_welcome(self, workspace_cwd: str, branch: str,
                      session_id: str, approval: str) -> None:
        if not self.use_stream:
            print(f"nano-coder  {self.model}  {workspace_cwd}")
            return

        w = term_width()
        inner = w - 2
        bar = f"{C_DIM}{'─' * w}{reset()}"

        def center(text: str) -> str:
            plain = strip_ansi(text)
            pad = max(0, (inner - len(plain)) // 2)
            return " " * pad + text

        def row(left_label: str, left_val: str,
                right_label: str, right_val: str) -> str:
            lw = (inner - 4) // 2
            left  = f"{C_DIM}{left_label:<10}{reset()}{middle(left_val, lw - 10)}"
            right = f"{C_DIM}{right_label:<10}{reset()}{middle(right_val, lw - 10)}"
            plain_l = strip_ansi(left)
            gap = max(1, lw - len(plain_l))
            return "  " + left + " " * gap + "  " + right

        lines = []
        lines.append(f"{C_ACCENT}{'═' * w}{reset()}")
        for art_line in WELCOME_ART:
            lines.append(center(f"{C_MODEL}{art_line}{reset()}"))
        lines.append(center(f"{bold()}{C_HEADER}NANO CODER{reset()}"))
        lines.append(bar)
        lines.append(row("workspace", workspace_cwd, "branch", branch))
        lines.append(row("model", self.model, "session", session_id[:20]))
        lines.append(row("approval", approval, "", ""))
        lines.append(f"{C_ACCENT}{'═' * w}{reset()}")
        print("\n".join(lines))

    def start_turn(self) -> None:
        """Called at the start of each agent turn (new user request)."""
        with self._lock:
            self._left_lines = []
            self._right_lines = []
            self._current_tool = ""
        if self.use_stream:
            self._render_panes()

    def write_token(self, token: str) -> None:
        """Append a streamed token to the left (model) pane."""
        with self._lock:
            if self._left_lines:
                self._left_lines[-1] += token
            else:
                self._left_lines.append(token)
            # wrap on newlines in token
            if "\n" in self._left_lines[-1]:
                parts = self._left_lines[-1].split("\n")
                self._left_lines[-1:] = parts
        if self.use_stream:
            self._render_panes()

    def write_model_line(self, text: str) -> None:
        """Append a complete line to the left pane (non-streaming)."""
        with self._lock:
            self._left_lines.append(text)
        if self.use_stream:
            self._render_panes()

    def write_tool_header(self, name: str, args: dict) -> None:
        """Show tool being called in the right pane."""
        self._current_tool = name
        arg_str = "  ".join(
            f"{k}={middle(str(v), 30)}" for k, v in args.items() if k != "content"
        )
        with self._lock:
            if self._right_lines:
                self._right_lines.append("")
            self._right_lines.append(
                f"{C_TOOL_HDR}[{name}]{reset()}  {C_DIM}{arg_str}{reset()}"
            )
        if self.use_stream:
            self._render_panes()
        elif not self.use_stream:
            print(f"\n  {C_TOOL_HDR}▶ {name}{reset()}  {C_DIM}{arg_str}{reset()}")

    def write_tool_result(self, result: str) -> None:
        """Show tool result in the right pane."""
        with self._lock:
            for line in result.splitlines()[:30]:
                self._right_lines.append(f"  {C_TOOL_OUT}{line}{reset()}")
        if self.use_stream:
            self._render_panes()
        elif not self.use_stream:
            for line in result.splitlines()[:30]:
                print(f"    {C_TOOL_OUT}{line}{reset()}")

    def write_final(self, text: str) -> None:
        """Display the final answer."""
        if self.use_stream:
            self._render_panes(final=text)
        else:
            print(f"\n{C_FINAL}{text}{reset()}\n")

    def write_error(self, text: str) -> None:
        if self.use_stream:
            with self._lock:
                self._right_lines.append(f"{C_ERROR}error: {text}{reset()}")
            self._render_panes()
        else:
            print(f"{C_ERROR}error: {text}{reset()}", file=sys.stderr)

    def write_info(self, text: str) -> None:
        if not self.use_stream:
            print(f"{C_DIM}{text}{reset()}")

    def prompt_line(self, session_id: str) -> str:
        sid = session_id[:12]
        return (
            f"\n{C_DIM}[{sid}]{reset()} "
            f"{C_ACCENT}nano-coder{reset()} "
            f"{C_DIM}({self.model}){reset()} > "
        )

    # ── internal rendering ───────────────────────────────────────

    def _render_panes(self, final: str = "") -> None:
        """Re-render the split pane to stdout."""
        w = term_width()
        lw = max(20, (w - 3) // 2)   # left pane width
        rw = w - lw - 3               # right pane width

        out_lines: list[str] = []

        # Header bar
        model_str = middle(self.model, lw - 2)
        tool_str  = middle(self._current_tool or "—", rw - 2)
        header = (
            f"{bg(20, 20, 30)}{C_HEADER} {model_str:<{lw - 1}}"
            f"{C_DIM}│{reset()}"
            f"{bg(20, 20, 30)}{C_TOOL_HDR} {tool_str:<{rw - 1}}{reset()}"
        )
        out_lines.append(header)
        out_lines.append(f"{C_DIM}{'─' * lw}┼{'─' * (rw + 1)}{reset()}")

        # Content rows — zip left and right
        left_visible  = self._visible_lines(self._left_lines,  lw)
        right_visible = self._visible_lines(self._right_lines, rw)
        max_rows = max(len(left_visible), len(right_visible), 8)

        for i in range(max_rows):
            l = left_visible[i]  if i < len(left_visible)  else ""
            r = right_visible[i] if i < len(right_visible) else ""
            l_plain = strip_ansi(l)
            r_plain = strip_ansi(r)
            l_pad = max(0, lw - len(l_plain))
            r_pad = max(0, rw - len(r_plain))
            row = (
                f"{C_MODEL}{l}{reset()}{' ' * l_pad}"
                f" {C_DIM}│{reset()} "
                f"{r}{' ' * r_pad}"
            )
            out_lines.append(row)

        out_lines.append(f"{C_DIM}{'─' * lw}┴{'─' * (rw + 1)}{reset()}")

        if final:
            out_lines.append(f"{C_FINAL}{bold()}  {final}{reset()}")
            out_lines.append("")

        # Move cursor to top-left, overwrite
        move_up = f"{ESC}[{len(out_lines) + 2}A{ESC}[0J"
        sys.stdout.write(move_up + "\n".join(out_lines) + "\n")
        sys.stdout.flush()

    def _visible_lines(self, lines: list[str], width: int) -> list[str]:
        """Word-wrap lines to fit pane width, return last N visible."""
        result = []
        for line in lines:
            plain = strip_ansi(line)
            if len(plain) <= width:
                result.append(line)
            else:
                # Hard wrap (preserve ANSI in first segment)
                result.append(line[:width])
                result.append(f"  {C_DIM}{plain[width:width + width - 2]}{reset()}")
        return result[-20:]   # keep last 20 visible lines per pane


# ─────────────────────────────────────────────
# Workspace context
# ─────────────────────────────────────────────

class WorkspaceContext:
    def __init__(self, cwd: str, repo_root: str, branch: str, status: str):
        self.cwd = cwd
        self.repo_root = repo_root
        self.branch = branch
        self.status = status

    @classmethod
    def build(cls, cwd: str) -> "WorkspaceContext":
        cwd = str(Path(cwd).resolve())

        def git(args: list[str], fallback: str = "") -> str:
            try:
                r = subprocess.run(
                    ["git", *args], cwd=cwd,
                    capture_output=True, text=True, check=True, timeout=5,
                )
                return r.stdout.strip() or fallback
            except Exception:
                return fallback

        repo_root = git(["rev-parse", "--show-toplevel"], cwd)
        branch    = git(["branch", "--show-current"], "-")
        status    = git(["status", "--short"], "clean") or "clean"

        return cls(cwd=cwd, repo_root=repo_root, branch=branch,
                   status=clip(status, 500))

    def text(self) -> str:
        return (
            f"cwd: {self.cwd}\n"
            f"git status:\n{self.status}"
        )


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
                 top_p: float, timeout: int):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout

    def _payload(self, prompt: str, max_new_tokens: int, stream: bool) -> bytes:
        return json.dumps({
            "model":  self.model,
            "prompt": prompt,
            "stream": stream,
            "raw":    False,
            "think":  False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": self.temperature,
                "top_p":       self.top_p,
            },
        }).encode("utf-8")

    def complete(self, prompt: str, max_new_tokens: int) -> str:
        req = urllib.request.Request(
            self.host + "/api/generate",
            data=self._payload(prompt, max_new_tokens, stream=False),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host}\n"
                "Make sure `ollama serve` is running and the model is pulled.\n"
                f"Model: {self.model}"
            ) from exc
        if data.get("error"):
            raise RuntimeError(f"Ollama error: {data['error']}")
        return data.get("response", "")

    def stream(self, prompt: str, max_new_tokens: int):
        """Yields tokens one by one from Ollama's streaming NDJSON API."""
        req = urllib.request.Request(
            self.host + "/api/generate",
            data=self._payload(prompt, max_new_tokens, stream=True),
            headers={"Content-Type": "application/json"},
            method="POST",
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
                        raise RuntimeError(f"Ollama stream error: {chunk['error']}")
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host}\n"
                f"Model: {self.model}"
            ) from exc


class FakeModelClient:
    """For tests: replays scripted responses."""
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def complete(self, prompt: str, max_new_tokens: int) -> str:
        self.prompts.append(prompt)
        if not self.outputs:
            raise RuntimeError("FakeModelClient ran out of outputs")
        return self.outputs.pop(0)

    def stream(self, prompt: str, max_new_tokens: int):
        response = self.complete(prompt, max_new_tokens)
        yield from response   # yield char by char


# ─────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────
# Handles noisy output from small models:
#   - Markdown fences around tool calls
#   - <think>...</think> blocks (Qwen 3.x)
#   - Missing closing tags
#   - Wrong case (<TOOL>, <Tool>)
#   - JSON tool format (mini-coding-agent compat fallback)
#   - Plain text with no tags → treated as <final>

class Parser:

    @staticmethod
    def normalize(raw: str) -> str:
        """Strip noise before parsing."""
        # Remove <think>...</think> blocks
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
        # Strip markdown fences that wrap tool/final tags
        raw = re.sub(r"```(?:xml|json)?\s*(<(?:tool|final)[\s>])", r"\1", raw, flags=re.IGNORECASE)
        raw = re.sub(r"(</(?:tool|final)>)\s*```", r"\1", raw, flags=re.IGNORECASE)
        # Normalize tag case
        raw = re.sub(r"<(/?)(TOOL|FINAL|CONTENT|OLD_TEXT|NEW_TEXT)(\b[^>]*)>",
                     lambda m: f"<{m.group(1)}{m.group(2).lower()}{m.group(3)}>",
                     raw)
        return raw.strip()

    @staticmethod
    def parse(raw: str) -> tuple[str, dict | str]:
        """
        Returns:
          ("tool",  {"name": str, "args": dict})
          ("final", str)
          ("retry", str)   — malformed, with hint message
        """
        raw = Parser.normalize(raw)

        # ── Try XML tool format (primary) ────────────────────────
        xml_result = Parser._parse_xml_tool(raw)
        if xml_result is not None:
            name, args = xml_result
            if name:
                return "tool", {"name": name, "args": args}

        # ── Try JSON tool format (fallback for compat) ────────────
        if "<tool>" in raw:
            body = Parser._extract_tag(raw, "tool")
            try:
                payload = json.loads(body)
                if isinstance(payload, dict) and payload.get("name"):
                    return "tool", {"name": payload["name"],
                                    "args": payload.get("args") or {}}
            except (json.JSONDecodeError, KeyError):
                pass

        # ── Final answer ──────────────────────────────────────────
        if "<final>" in raw:
            text = Parser._extract_tag(raw, "final").strip()
            if text:
                return "final", text
            return "retry", "Empty <final> tag — write your answer inside <final>...</final>"

        # ── Plain text fallback ───────────────────────────────────
        raw = raw.strip()
        if raw:
            # If it looks like the model just answered directly, accept it
            return "final", raw

        return "retry", (
            "No valid output detected. Reply with:\n"
            '  <tool name="tool_name" arg1="val1"></tool>\n'
            "  or\n"
            "  <final>your answer</final>"
        )

    @staticmethod
    def _parse_xml_tool(raw: str) -> tuple[str, dict] | None:
        """Parse <tool name="..." key="val"><content>...</content></tool>"""
        match = re.search(r"<tool\b([^>]*)>(.*?)</tool>", raw, re.DOTALL | re.IGNORECASE)
        if not match:
            # Try self-closing or unclosed tag
            match = re.search(r"<tool\b([^>]*?)(?:/>|>([^<]*)$)", raw, re.DOTALL | re.IGNORECASE)
            if not match:
                return None
            attr_str = match.group(1)
            body = (match.group(2) or "").strip()
        else:
            attr_str = match.group(1)
            body = match.group(2)

        attrs = Parser._parse_attrs(attr_str)
        name  = str(attrs.pop("name", "")).strip()
        args  = dict(attrs)

        # Extract named sub-tags from body
        for key in ("content", "old_text", "new_text", "command", "task", "pattern"):
            sub = Parser._extract_tag(body, key)
            if sub is not None:
                args[key] = sub

        # Body itself as content fallback (for write_file)
        if name == "write_file" and "content" not in args:
            body_stripped = body.strip("\n")
            if body_stripped:
                args["content"] = body_stripped

        return name, args

    @staticmethod
    def _parse_attrs(text: str) -> dict:
        attrs = {}
        for m in re.finditer(r"""(\w+)\s*=\s*(?:"([^"]*)"|'([^']*)')""", text):
            attrs[m.group(1)] = m.group(2) if m.group(2) is not None else m.group(3)
        return attrs

    @staticmethod
    def _extract_tag(text: str, tag: str) -> str | None:
        start_tag = f"<{tag}>"
        end_tag   = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return None
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:].strip()
        return text[start:end]


# ─────────────────────────────────────────────
# Stream parser (detects tool tags mid-stream)
# ─────────────────────────────────────────────

class StreamParser:
    """
    Stateful parser that processes a token stream and fires callbacks
    when a complete tool call or final answer is detected.
    """

    def __init__(self, on_tool: callable, on_final: callable,
                 on_token: callable):
        self.on_tool  = on_tool     # (name, args) -> str  (tool result)
        self.on_final = on_final    # (text) -> None
        self.on_token = on_token    # (token) -> None  (raw token for display)
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
        """Call after stream ends to process remaining buffer."""
        if self._done or not self._buffer.strip():
            return
        kind, payload = Parser.parse(self._buffer)
        if kind == "tool":
            result = self.on_tool(payload["name"], payload["args"])
            _ = result  # result handled by caller via on_tool return
        elif kind == "final":
            self.on_final(payload)
        else:
            # Plain text — treat as final
            self.on_final(self._buffer.strip())
        self._done = True

    def _try_detect(self) -> None:
        buf = self._buffer

        # Detect </tool> close
        if "</tool>" in buf.lower():
            kind, payload = Parser.parse(buf)
            if kind == "tool":
                self.on_tool(payload["name"], payload["args"])
                self._buffer = ""
                self._done = True
                return

        # Detect </final> close
        if "</final>" in buf.lower():
            text = Parser._extract_tag(buf.lower().replace("</final>",
                                       "</final>").replace("<final>", "<final>"), "final")
            # Re-extract from original buf to preserve case
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

        self.tools        = self._build_tools()
        self.session_path = self.session_store.save(self.session)

    @classmethod
    def from_session(cls, model_client, workspace, session_store,
                     renderer, session_id: str, **kwargs) -> "NanoAgent":
        session = session_store.load(session_id)
        return cls(model_client=model_client, workspace=workspace,
                   session_store=session_store, renderer=renderer,
                   session=session, **kwargs)

    # ── Public interface ─────────────────────────────────────────

    def ask(self, user_message: str) -> str:
        memory = self.session["memory"]
        if not memory["task"]:
            memory["task"] = clip(user_message.strip(), 300)

        self._record({"role": "user", "content": user_message, "ts": now_iso()})
        self.renderer.start_turn()

        tool_steps  = 0
        attempts    = 0
        max_attempts = max(self.max_steps * 3, self.max_steps + 6)
        last_result  = ""

        while tool_steps < self.max_steps and attempts < max_attempts:
            attempts += 1
            prompt = self._build_prompt(user_message)

            if self.use_stream:
                raw, tool_fired = self._run_streaming_step()
                if tool_fired:
                    tool_steps += 1
                    continue
                # Stream ended; check for final
                kind, payload = Parser.parse(raw)
            else:
                raw  = self.model_client.complete(prompt, self.max_new_tokens)
                self.renderer.write_model_line(raw)
                kind, payload = Parser.parse(raw)

            if kind == "tool":
                tool_steps += 1
                name   = payload["name"]
                args   = payload["args"]
                result = self._run_tool(name, args)
                last_result = result
                self._record({
                    "role": "tool", "name": name,
                    "args": args,   "content": result, "ts": now_iso(),
                })
                self._note_tool(name, args, result)
                continue

            if kind == "final":
                final = str(payload).strip()
                self._record({"role": "assistant", "content": final, "ts": now_iso()})
                remember(memory["notes"], clip(final, 200), 4)
                self.renderer.write_final(final)
                return final

            # retry
            self._record({"role": "assistant", "content": str(payload), "ts": now_iso()})
            continue

        final = (
            "Reached step limit without a final answer."
            if tool_steps >= self.max_steps
            else "Too many malformed responses."
        )
        self._record({"role": "assistant", "content": final, "ts": now_iso()})
        self.renderer.write_final(final)
        return final

    def reset(self) -> None:
        self.session["history"] = []
        self.session["memory"]  = {"task": "", "files": [], "notes": []}
        self.session_store.save(self.session)

    def memory_text(self) -> str:
        m = self.session["memory"]
        notes = "\n".join(f"  - {n}" for n in m["notes"]) or "  - none"
        return (
            f"task:  {m['task'] or '—'}\n"
            f"files: {', '.join(m['files']) or '—'}\n"
            f"notes:\n{notes}"
        )

    def tools_text(self) -> str:
        lines = []
        for name, tool in self.tools.items():
            risk = "⚠ approval" if tool["risky"] else "  safe"
            lines.append(f"  {C_TOOL_HDR}{name:<14}{reset()} {risk}  {C_DIM}{tool['desc']}{reset()}")
        return "\n".join(lines)

    # ── Prompt construction ──────────────────────────────────────

    def _build_prefix(self) -> str:
        tool_lines = []
        for name, tool in self.tools.items():
            schema_parts = "  ".join(f'{k}="{v}"' for k, v in tool["schema"].items())
            tool_lines.append(f'  <tool name="{name}" {schema_parts}></tool>')

        tool_text = "\n".join(tool_lines)

        return textwrap.dedent(f"""\
            You are nano-coder, a coding agent running on Ollama.
            Rules:
            - Use tools to inspect the workspace before writing code.
            - Emit exactly one tool call OR one final answer per response.
            - Tool call format (XML only, no JSON):
              <tool name="TOOL_NAME" arg1="val1" arg2="val2"></tool>
              For file content, put it in a <content> tag in the body:
              <tool name="write_file" path="x.py"><content>
              code here
              </content></tool>
            - Final answer format: <final>your answer</final>
            - Keep answers short and concrete.
            - Never invent tool results.
            Available tools:
            {tool_text}
            Example:
              <tool name="read_file" path="main.py" start="1" end="40"></tool>
              <final>Done. I updated main.py with the fix.</final>
        """).strip()

    def _build_prompt(self, user_message: str) -> str:
        return (
            f"{self._build_prefix()}\n\n"
            f"Workspace:\n{self.workspace.text()}\n\n"
            f"Memory:\n{self._memory_text_short()}\n\n"
            f"History:\n{self._history_text()}\n\n"
            f"Task: {user_message}"
        )

    def _memory_text_short(self) -> str:
        m = self.session["memory"]
        return (
            f"task: {m['task'] or '—'}  "
            f"files: {', '.join(m['files'][-4:]) or '—'}"
        )

    def _history_text(self) -> str:
        history = self.session["history"]
        if not history:
            return "(empty)"

        lines      = []
        seen_reads : set[str] = set()
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

    # ── Streaming step ───────────────────────────────────────────

    def _run_streaming_step(self) -> tuple[str, bool]:
        """
        Run one streaming model call. Returns (full_raw, tool_was_fired).
        tool_was_fired=True means a tool was detected and executed mid-stream.
        """
        prompt    = self._build_prompt(self.session["memory"].get("task", ""))
        full_raw  = ""
        tool_fired = False

        def on_token(token: str) -> None:
            nonlocal full_raw
            full_raw += token
            self.renderer.write_token(token)

        def on_tool(name: str, args: dict) -> str:
            nonlocal tool_fired
            tool_fired = True
            result = self._run_tool(name, args)
            self._record({
                "role": "tool", "name": name,
                "args": args,   "content": result, "ts": now_iso(),
            })
            self._note_tool(name, args, result)
            return result

        def on_final(text: str) -> None:
            pass  # handled by caller

        sp = StreamParser(on_tool=on_tool, on_final=on_final, on_token=on_token)

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

    # ── Tool execution ───────────────────────────────────────────

    def _build_tools(self) -> dict:
        tools = {
            "list_files": {
                "schema":  {"path": "."},
                "risky":   False,
                "desc":    "List files in the workspace.",
                "run":     self._tool_list_files,
            },
            "read_file": {
                "schema":  {"path": "REQUIRED", "start": "1", "end": "60"},
                "risky":   False,
                "desc":    "Read a file by line range.",
                "run":     self._tool_read_file,
            },
            "write_file": {
                "schema":  {"path": "REQUIRED"},
                "risky":   True,
                "desc":    "Write or overwrite a file.",
                "run":     self._tool_write_file,
            },
            "run_shell": {
                "schema":  {"command": "REQUIRED", "timeout": "20"},
                "risky":   True,
                "desc":    "Run a shell command in the repo root.",
                "run":     self._tool_run_shell,
            },
        }
        if self.tool_set == "full":
            tools["search"] = {
                "schema":  {"pattern": "REQUIRED", "path": "."},
                "risky":   False,
                "desc":    "Search the workspace with ripgrep or fallback.",
                "run":     self._tool_search,
            }
            tools["patch_file"] = {
                "schema":  {"path": "REQUIRED", "old_text": "REQUIRED", "new_text": "REQUIRED"},
                "risky":   True,
                "desc":    "Replace one exact block in a file.",
                "run":     self._tool_patch_file,
            }
        return tools

    def _run_tool(self, name: str, args: dict) -> str:
        self.renderer.write_tool_header(name, args)
        tool = self.tools.get(name)
        if tool is None:
            result = f"error: unknown tool '{name}'. Available: {', '.join(self.tools)}"
            self.renderer.write_tool_result(result)
            return result

        try:
            self._validate_tool(name, args)
        except ValueError as exc:
            result = f"error: invalid args for {name}: {exc}"
            self.renderer.write_tool_result(result)
            return result

        if self._is_repeated(name, args):
            result = f"error: repeated identical call to {name} — try a different approach"
            self.renderer.write_tool_result(result)
            return result

        if tool["risky"] and not self._approve(name, args):
            result = f"error: approval denied for {name}"
            self.renderer.write_tool_result(result)
            return result

        try:
            result = clip(tool["run"](args))
        except Exception as exc:
            result = f"error: {name} failed: {exc}"

        self.renderer.write_tool_result(result)
        return result

    def _validate_tool(self, name: str, args: dict) -> None:
        args = args or {}
        if name == "list_files":
            p = self._safe_path(args.get("path", "."))
            if not p.is_dir():
                raise ValueError("path is not a directory")

        elif name == "read_file":
            if not args.get("path"):
                raise ValueError("path is required")
            p = self._safe_path(args["path"])
            if not p.exists():
                raise ValueError(f"file not found: {args['path']}")
            start = int(args.get("start", 1))
            end   = int(args.get("end",   60))
            if start < 1 or end < start:
                raise ValueError("invalid line range")

        elif name == "write_file":
            if not args.get("path"):
                raise ValueError("path is required")
            if "content" not in args:
                raise ValueError("content is required")
            p = self._safe_path(args["path"])
            if p.is_dir():
                raise ValueError("path is a directory")

        elif name == "run_shell":
            if not str(args.get("command", "")).strip():
                raise ValueError("command is required")
            t = int(args.get("timeout", 20))
            if not (1 <= t <= 120):
                raise ValueError("timeout must be 1–120")

        elif name == "search":
            if not str(args.get("pattern", "")).strip():
                raise ValueError("pattern is required")

        elif name == "patch_file":
            if not args.get("path"):
                raise ValueError("path is required")
            p = self._safe_path(args["path"])
            if not p.is_file():
                raise ValueError("file not found")
            old = str(args.get("old_text", ""))
            if not old:
                raise ValueError("old_text is required")
            if "new_text" not in args:
                raise ValueError("new_text is required")
            count = p.read_text(encoding="utf-8").count(old)
            if count != 1:
                raise ValueError(f"old_text must appear exactly once (found {count})")

    def _is_repeated(self, name: str, args: dict) -> bool:
        tool_events = [e for e in self.session["history"] if e["role"] == "tool"]
        if len(tool_events) < 2:
            return False
        last2 = tool_events[-2:]
        return all(e["name"] == name and e["args"] == args for e in last2)

    def _approve(self, name: str, args: dict) -> bool:
        if self.approval_policy == "auto":
            return True
        if self.approval_policy == "never":
            return False
        preview = json.dumps(args, ensure_ascii=False)[:120]
        try:
            ans = input(
                f"\n{C_ACCENT}approve{reset()} {C_TOOL_HDR}{name}{reset()} "
                f"{C_DIM}{preview}{reset()} [y/N] "
            )
        except EOFError:
            return False
        return ans.strip().lower() in {"y", "yes"}

    # ── Tool implementations ─────────────────────────────────────

    def _safe_path(self, raw: str) -> Path:
        p = Path(raw)
        p = p if p.is_absolute() else self.root / p
        resolved = p.resolve()
        if not str(resolved).startswith(str(self.root)):
            raise ValueError(f"path escapes workspace: {raw}")
        return resolved

    def _tool_list_files(self, args: dict) -> str:
        path = self._safe_path(args.get("path", "."))
        entries = sorted(
            (e for e in path.iterdir() if e.name not in IGNORED_DIRS),
            key=lambda e: (e.is_file(), e.name.lower()),
        )
        lines = []
        for e in entries[:150]:
            kind = "[F]" if e.is_file() else "[D]"
            lines.append(f"{kind} {e.relative_to(self.root)}")
        return "\n".join(lines) or "(empty directory)"

    def _tool_read_file(self, args: dict) -> str:
        path  = self._safe_path(args["path"])
        start = int(args.get("start", 1))
        end   = int(args.get("end",   60))
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        body  = "\n".join(
            f"{n:>4}: {line}"
            for n, line in enumerate(lines[start - 1:end], start=start)
        )
        return f"# {path.relative_to(self.root)}\n{body}"

    def _tool_write_file(self, args: dict) -> str:
        path    = self._safe_path(args["path"])
        content = str(args["content"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"wrote {path.relative_to(self.root)} ({len(content)} chars)"

    def _tool_run_shell(self, args: dict) -> str:
        cmd     = str(args["command"]).strip()
        timeout = int(args.get("timeout", 20))
        result  = subprocess.run(
            cmd, cwd=self.root, shell=True,
            capture_output=True, text=True, timeout=timeout,
        )
        out = clip(result.stdout.strip() or "(empty)", 800)
        err = clip(result.stderr.strip() or "(empty)", 400)
        return f"exit: {result.returncode}\nstdout:\n{out}\nstderr:\n{err}"

    def _tool_search(self, args: dict) -> str:
        pattern = str(args["pattern"]).strip()
        path    = self._safe_path(args.get("path", "."))
        if shutil.which("rg"):
            r = subprocess.run(
                ["rg", "-n", "--smart-case", "--max-count", "150", pattern, str(path)],
                cwd=self.root, capture_output=True, text=True,
            )
            return r.stdout.strip() or r.stderr.strip() or "(no matches)"
        # Fallback
        matches = []
        files = [path] if path.is_file() else [
            f for f in path.rglob("*")
            if f.is_file()
            and not any(p in IGNORED_DIRS for p in f.relative_to(self.root).parts)
        ]
        for f in files:
            for n, line in enumerate(
                f.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                if pattern.lower() in line.lower():
                    matches.append(f"{f.relative_to(self.root)}:{n}:{line}")
                    if len(matches) >= 150:
                        return "\n".join(matches)
        return "\n".join(matches) or "(no matches)"

    def _tool_patch_file(self, args: dict) -> str:
        path     = self._safe_path(args["path"])
        old_text = str(args["old_text"])
        new_text = str(args["new_text"])
        text     = path.read_text(encoding="utf-8")
        path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"patched {path.relative_to(self.root)}"

    # ── Session helpers ──────────────────────────────────────────

    def _record(self, item: dict) -> None:
        self.session["history"].append(item)
        self.session_path = self.session_store.save(self.session)

    def _note_tool(self, name: str, args: dict, result: str) -> None:
        m    = self.session["memory"]
        path = args.get("path")
        if name in {"read_file", "write_file", "patch_file"} and path:
            remember(m["files"], str(path), 8)
        note = f"{name}: {clip(str(result).replace(chr(10), ' '), 150)}"
        remember(m["notes"], note, 4)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nano-coder",
        description="Coding agent for tiny and small LLMs (<=2B params).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("prompt", nargs="*", help="One-shot prompt (optional).")
    p.add_argument("--cwd",            default=".",                help="Workspace directory.")
    p.add_argument("--model",          default=DEFAULT_MODEL,      help="Ollama model name.")
    p.add_argument("--host",           default=DEFAULT_HOST,       help="Ollama server URL.")
    p.add_argument("--ollama-timeout", type=int, default=300,      help="Ollama timeout (s).")
    p.add_argument("--resume",         default=None,               help="Session ID or 'latest'.")
    p.add_argument("--approval",       choices=("ask","auto","never"), default="ask",
                   help="Risky tool approval policy.")
    p.add_argument("--max-steps",      type=int, default=DEFAULT_MAX_STEPS,
                   help="Max tool/model turns per request.")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                   help="Max tokens per model call.")
    p.add_argument("--temperature",    type=float, default=DEFAULT_TEMPERATURE,
                   help="Sampling temperature.")
    p.add_argument("--top-p",          type=float, default=DEFAULT_TOP_P,
                   help="Top-p sampling.")
    p.add_argument("--tool-set",       choices=("core","full"), default="core",
                   help="Tool set: core (4 tools) or full (6 tools).")
    p.add_argument("--no-stream",      action="store_true",
                   help="Disable live token streaming.")
    return p


def build_agent(args: argparse.Namespace,
                renderer: TerminalRenderer) -> NanoAgent:
    workspace = WorkspaceContext.build(args.cwd)
    store     = SessionStore(
        Path(workspace.repo_root) / ".nano-coding-agent" / "sessions"
    )
    client = OllamaModelClient(
        model=args.model, host=args.host,
        temperature=args.temperature, top_p=args.top_p,
        timeout=args.ollama_timeout,
    )
    kwargs = dict(
        model_client=client,
        workspace=workspace,
        session_store=store,
        renderer=renderer,
        approval_policy=args.approval,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        tool_set=args.tool_set,
        use_stream=not args.no_stream,
    )

    sid = args.resume
    if sid == "latest":
        sid = store.latest()
    if sid:
        return NanoAgent.from_session(session_id=sid, **kwargs)
    return NanoAgent(**kwargs)


def main(argv: list[str] | None = None) -> int:
    args     = build_arg_parser().parse_args(argv)
    renderer = TerminalRenderer(model=args.model, use_stream=not args.no_stream)
    agent    = build_agent(args, renderer)

    workspace = WorkspaceContext.build(args.cwd)
    renderer.print_welcome(
        workspace_cwd=workspace.cwd,
        branch=workspace.branch,
        session_id=agent.session["id"],
        approval=args.approval,
    )

    # One-shot mode
    if args.prompt:
        prompt = " ".join(args.prompt).strip()
        if prompt:
            try:
                agent.ask(prompt)
            except RuntimeError as exc:
                print(f"{C_ERROR}{exc}{reset()}", file=sys.stderr)
                return 1
        return 0

    # Interactive REPL
    while True:
        try:
            raw = input(renderer.prompt_line(agent.session["id"]))
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0

        user_input = raw.strip()
        if not user_input:
            continue

        if user_input in {"/exit", "/quit"}:
            return 0

        if user_input == "/help":
            print(HELP_TEXT)
            continue

        if user_input == "/memory":
            print(agent.memory_text())
            continue

        if user_input == "/session":
            print(agent.session_path)
            continue

        if user_input == "/tools":
            print(agent.tools_text())
            continue

        if user_input == "/reset":
            agent.reset()
            print(f"{C_DIM}Session cleared.{reset()}")
            continue

        if user_input.startswith("/"):
            print(f"{C_DIM}Unknown command. Type /help for available commands.{reset()}")
            continue

        try:
            agent.ask(user_input)
        except RuntimeError as exc:
            print(f"\n{C_ERROR}{exc}{reset()}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
