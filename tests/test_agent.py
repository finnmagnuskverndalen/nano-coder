"""
Tests for nano-coding-agent.
Run with: python -m pytest tests/ -v
"""

import json
import sys
import textwrap
from pathlib import Path

import pytest

# Make nano_coding_agent importable from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from nano_coding_agent import (
    FakeModelClient,
    NanoAgent,
    OllamaModelClient,
    Parser,
    SessionStore,
    StreamParser,
    TerminalRenderer,
    WorkspaceContext,
    clip,
    middle,
)


# ─────────────────────────────────────────────
# Parser tests
# ─────────────────────────────────────────────

class TestParser:

    # ── normalize ────────────────────────────

    def test_strips_think_blocks(self):
        raw = "<think>let me reason about this</think>\n<final>Done.</final>"
        assert "<think>" not in Parser.normalize(raw)
        assert "<final>Done.</final>" in Parser.normalize(raw)

    def test_strips_markdown_fence(self):
        raw = "```xml\n<tool name=\"list_files\" path=\".\"></tool>\n```"
        norm = Parser.normalize(raw)
        assert "```" not in norm
        assert "<tool" in norm

    def test_normalizes_uppercase_tags(self):
        raw = "<TOOL name=\"list_files\" path=\".\"></TOOL>"
        norm = Parser.normalize(raw)
        assert "<tool" in norm.lower()
        assert "<TOOL" not in norm

    def test_strips_think_and_fence_combined(self):
        raw = "<think>hmm</think>\n```xml\n<final>answer</final>\n```"
        norm = Parser.normalize(raw)
        assert "<think>" not in norm
        assert "```" not in norm
        assert "<final>answer</final>" in norm

    # ── parse: tool ──────────────────────────

    def test_parse_xml_tool_simple(self):
        raw = '<tool name="list_files" path="."></tool>'
        kind, payload = Parser.parse(raw)
        assert kind == "tool"
        assert payload["name"] == "list_files"
        assert payload["args"]["path"] == "."

    def test_parse_xml_tool_with_content(self):
        raw = textwrap.dedent("""\
            <tool name="write_file" path="hello.py">
            <content>
            def hello():
                return "hi"
            </content>
            </tool>
        """)
        kind, payload = Parser.parse(raw)
        assert kind == "tool"
        assert payload["name"] == "write_file"
        assert "def hello" in payload["args"]["content"]

    def test_parse_xml_tool_run_shell(self):
        raw = '<tool name="run_shell" command="python -m pytest -q" timeout="20"></tool>'
        kind, payload = Parser.parse(raw)
        assert kind == "tool"
        assert payload["name"] == "run_shell"
        assert payload["args"]["command"] == "python -m pytest -q"
        assert payload["args"]["timeout"] == "20"

    def test_parse_json_tool_fallback(self):
        raw = '<tool>{"name": "list_files", "args": {"path": "."}}</tool>'
        kind, payload = Parser.parse(raw)
        assert kind == "tool"
        assert payload["name"] == "list_files"

    def test_parse_tool_wrapped_in_fence(self):
        raw = '```xml\n<tool name="list_files" path="."></tool>\n```'
        kind, payload = Parser.parse(raw)
        assert kind == "tool"
        assert payload["name"] == "list_files"

    def test_parse_tool_uppercase(self):
        raw = '<TOOL name="list_files" path="."></TOOL>'
        kind, payload = Parser.parse(raw)
        assert kind == "tool"

    def test_parse_tool_with_think_prefix(self):
        raw = "<think>I should list files first</think>\n<tool name=\"list_files\" path=\".\"></tool>"
        kind, payload = Parser.parse(raw)
        assert kind == "tool"
        assert payload["name"] == "list_files"

    # ── parse: final ─────────────────────────

    def test_parse_final(self):
        raw = "<final>All done! The file is written.</final>"
        kind, payload = Parser.parse(raw)
        assert kind == "final"
        assert payload == "All done! The file is written."

    def test_parse_final_with_think_prefix(self):
        raw = "<think>I'm done</think>\n<final>Task complete.</final>"
        kind, payload = Parser.parse(raw)
        assert kind == "final"
        assert payload == "Task complete."

    def test_parse_plain_text_as_final(self):
        raw = "I've completed the task. The function is in main.py."
        kind, payload = Parser.parse(raw)
        assert kind == "final"
        assert "main.py" in payload

    def test_parse_empty_string_returns_retry(self):
        kind, payload = Parser.parse("")
        assert kind == "retry"

    # ── parse: patch_file ────────────────────

    def test_parse_patch_file(self):
        raw = textwrap.dedent("""\
            <tool name="patch_file" path="main.py">
            <old_text>return -1</old_text>
            <new_text>return mid</new_text>
            </tool>
        """)
        kind, payload = Parser.parse(raw)
        assert kind == "tool"
        assert payload["args"]["old_text"] == "return -1"
        assert payload["args"]["new_text"] == "return mid"


# ─────────────────────────────────────────────
# Utility tests
# ─────────────────────────────────────────────

class TestUtils:

    def test_clip_under_limit(self):
        assert clip("hello", 100) == "hello"

    def test_clip_over_limit(self):
        result = clip("a" * 200, 50)
        assert len(result) <= 80  # truncated + message
        assert "truncated" in result

    def test_middle_short(self):
        assert middle("hello", 20) == "hello"

    def test_middle_long(self):
        result = middle("a" * 100, 10)
        assert len(result) == 10
        assert "…" in result


# ─────────────────────────────────────────────
# Tool tests
# ─────────────────────────────────────────────

def make_agent(tmp_path: Path, responses: list[str], tool_set: str = "core") -> NanoAgent:
    """Create a NanoAgent with FakeModelClient for testing."""
    # Create a fake git repo
    subprocess_safe = __import__("subprocess")
    try:
        subprocess_safe.run(["git", "init"], cwd=tmp_path,
                            capture_output=True, check=True)
    except Exception:
        pass

    workspace = WorkspaceContext(
        cwd=str(tmp_path),
        repo_root=str(tmp_path),
        branch="main",
        status="clean",
    )
    store    = SessionStore(tmp_path / ".nano-coding-agent" / "sessions")
    renderer = TerminalRenderer(model="test-model", use_stream=False)
    client   = FakeModelClient(responses)

    return NanoAgent(
        model_client=client,
        workspace=workspace,
        session_store=store,
        renderer=renderer,
        approval_policy="auto",
        max_steps=5,
        max_new_tokens=256,
        tool_set=tool_set,
        use_stream=False,
    )


class TestTools:

    def test_list_files(self, tmp_path):
        (tmp_path / "hello.py").write_text("print('hi')")
        (tmp_path / "README.md").write_text("# test")

        agent = make_agent(tmp_path, [
            '<tool name="list_files" path="."></tool>',
            "<final>Listed files.</final>",
        ])
        result = agent.ask("list the files")
        # Check tool ran (files tracked in memory)
        assert "list_files" in str(agent.session["memory"]["notes"])

    def test_read_file(self, tmp_path):
        (tmp_path / "main.py").write_text("def add(a, b):\n    return a + b\n")

        agent = make_agent(tmp_path, [
            '<tool name="read_file" path="main.py" start="1" end="5"></tool>',
            "<final>Read the file.</final>",
        ])
        agent.ask("read main.py")
        history = agent.session["history"]
        tool_result = next(e for e in history if e["role"] == "tool")
        assert "def add" in tool_result["content"]

    def test_write_file(self, tmp_path):
        agent = make_agent(tmp_path, [
            textwrap.dedent("""\
                <tool name="write_file" path="output.py">
                <content>
                def hello():
                    return "hello"
                </content>
                </tool>
            """),
            "<final>Done writing.</final>",
        ])
        agent.ask("write a hello function")
        assert (tmp_path / "output.py").exists()
        assert "def hello" in (tmp_path / "output.py").read_text()

    def test_run_shell(self, tmp_path):
        agent = make_agent(tmp_path, [
            '<tool name="run_shell" command="echo hello" timeout="5"></tool>',
            "<final>Command ran.</final>",
        ])
        agent.ask("run echo hello")
        history = agent.session["history"]
        tool_result = next(e for e in history if e["role"] == "tool")
        assert "hello" in tool_result["content"]

    def test_path_sandbox_escape(self, tmp_path):
        agent = make_agent(tmp_path, [])
        with pytest.raises(ValueError, match="escapes workspace"):
            agent._safe_path("../../etc/passwd")

    def test_unknown_tool_returns_error(self, tmp_path):
        agent = make_agent(tmp_path, [
            '<tool name="fake_tool" arg="x"></tool>',
            "<final>Done.</final>",
        ])
        agent.ask("do something")
        history = agent.session["history"]
        tool_result = next((e for e in history if e["role"] == "tool"), None)
        assert tool_result is not None
        assert "unknown tool" in tool_result["content"]

    def test_write_creates_parent_dirs(self, tmp_path):
        agent = make_agent(tmp_path, [
            textwrap.dedent("""\
                <tool name="write_file" path="src/utils/helpers.py">
                <content>x = 1</content>
                </tool>
            """),
            "<final>Done.</final>",
        ])
        agent.ask("create nested file")
        assert (tmp_path / "src" / "utils" / "helpers.py").exists()

    def test_full_toolset_has_search(self, tmp_path):
        agent = make_agent(tmp_path, [], tool_set="full")
        assert "search" in agent.tools
        assert "patch_file" in agent.tools

    def test_core_toolset_no_search(self, tmp_path):
        agent = make_agent(tmp_path, [], tool_set="core")
        assert "search" not in agent.tools
        assert "patch_file" not in agent.tools


# ─────────────────────────────────────────────
# Agent loop tests
# ─────────────────────────────────────────────

class TestAgentLoop:

    def test_final_answer_direct(self, tmp_path):
        agent = make_agent(tmp_path, ["<final>42 is the answer.</final>"])
        result = agent.ask("what is the answer?")
        assert "42" in result

    def test_tool_then_final(self, tmp_path):
        (tmp_path / "main.py").write_text("x = 1\n")
        agent = make_agent(tmp_path, [
            '<tool name="read_file" path="main.py"></tool>',
            "<final>The file contains x = 1.</final>",
        ])
        result = agent.ask("what is in main.py?")
        assert "x = 1" in result

    def test_memory_tracks_task(self, tmp_path):
        agent = make_agent(tmp_path, ["<final>Done.</final>"])
        agent.ask("build a sorting function")
        assert "sorting" in agent.session["memory"]["task"]

    def test_memory_tracks_files(self, tmp_path):
        (tmp_path / "utils.py").write_text("pass\n")
        agent = make_agent(tmp_path, [
            '<tool name="read_file" path="utils.py"></tool>',
            "<final>Read it.</final>",
        ])
        agent.ask("read utils.py")
        assert "utils.py" in agent.session["memory"]["files"]

    def test_session_persists(self, tmp_path):
        agent = make_agent(tmp_path, ["<final>Done.</final>"])
        agent.ask("hello")
        sid = agent.session["id"]
        store = SessionStore(tmp_path / ".nano-coding-agent" / "sessions")
        loaded = store.load(sid)
        assert loaded["id"] == sid
        assert len(loaded["history"]) > 0

    def test_reset_clears_history(self, tmp_path):
        agent = make_agent(tmp_path, ["<final>Done.</final>"])
        agent.ask("hello")
        agent.reset()
        assert agent.session["history"] == []
        assert agent.session["memory"]["task"] == ""

    def test_plain_text_response_accepted(self, tmp_path):
        agent = make_agent(tmp_path, [
            "I've finished. The answer is in output.py."
        ])
        result = agent.ask("do something")
        assert "output.py" in result

    def test_repeated_tool_call_blocked(self, tmp_path):
        """Agent should detect repeated identical calls and not loop forever."""
        (tmp_path / "x.py").write_text("pass\n")
        # Simulate model that keeps repeating the same read_file call
        agent = make_agent(tmp_path, [
            '<tool name="read_file" path="x.py"></tool>',
            '<tool name="read_file" path="x.py"></tool>',
            '<tool name="read_file" path="x.py"></tool>',
            "<final>Done.</final>",
        ])
        agent.ask("read x.py forever")
        # Should complete without hanging; final recorded
        history = agent.session["history"]
        assert any(e["role"] == "assistant" for e in history)


# ─────────────────────────────────────────────
# Context trimming tests
# ─────────────────────────────────────────────

class TestContextTrimming:

    def test_history_text_empty(self, tmp_path):
        agent = make_agent(tmp_path, [])
        assert "(empty)" in agent._history_text()

    def test_clip_applied_to_long_tool_output(self, tmp_path):
        agent = make_agent(tmp_path, [])
        # Inject a long tool result
        agent.session["history"].append({
            "role": "tool", "name": "run_shell",
            "args": {"command": "cat big.txt"},
            "content": "x" * 5000,
            "ts": "2026-01-01T00:00:00Z",
        })
        text = agent._history_text()
        # The content should be clipped somewhere
        assert len(text) < 5000 + 500


# ─────────────────────────────────────────────
# StreamParser tests
# ─────────────────────────────────────────────

class TestStreamParser:

    def test_detects_final_mid_stream(self):
        finals = []
        tokens = []

        sp = StreamParser(
            on_tool=lambda n, a: "ok",
            on_final=finals.append,
            on_token=tokens.append,
        )
        for ch in "<final>Task complete.</final>":
            sp.feed(ch)

        assert finals == ["Task complete."]
        assert sp.done

    def test_detects_tool_mid_stream(self):
        tools = []

        def on_tool(name, args):
            tools.append((name, args))
            return "result"

        sp = StreamParser(
            on_tool=on_tool,
            on_final=lambda t: None,
            on_token=lambda t: None,
        )
        raw = '<tool name="list_files" path="."></tool>'
        for ch in raw:
            sp.feed(ch)

        assert len(tools) == 1
        assert tools[0][0] == "list_files"
        assert sp.done

    def test_flush_handles_plain_text(self):
        finals = []
        sp = StreamParser(
            on_tool=lambda n, a: "ok",
            on_final=finals.append,
            on_token=lambda t: None,
        )
        for ch in "The answer is 42.":
            sp.feed(ch)
        sp.flush()
        assert finals == ["The answer is 42."]
