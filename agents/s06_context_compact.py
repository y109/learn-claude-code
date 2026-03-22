#!/usr/bin/env python3
# Harness: compression -- clean memory for infinite sessions.
"""
s06_context_compact.py - Compact

Three-layer compression pipeline so the agent can work forever:

    Every turn:
    +------------------+
    | Tool call result |
    +------------------+
            |
            v
    [Layer 1: micro_compact]        (silent, every turn)
      Replace tool_result content older than last 3
      with "[Previous: used {tool_name}]"
            |
            v
    [Check: tokens > 50000?]
       |               |
       no              yes
       |               |
       v               v
    continue    [Layer 2: auto_compact]
                  Save full transcript to .transcripts/
                  Ask LLM to summarize conversation.
                  Replace all messages with [summary].
                        |
                        v
                [Layer 3: compact tool]
                  Model calls compact -> immediate summarization.
                  Same as auto, triggered manually.

Key insight: "The agent can forget strategically and keep working forever."
"""

import json
import os
import subprocess
import time
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."

THRESHOLD = 50000
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
KEEP_RECENT = 3


DIM, CYAN, YELLOW, GREEN, RED, RESET = "\033[90m", "\033[36m", "\033[33m", "\033[32m", "\033[31m", "\033[0m"


def estimate_tokens(messages: list) -> int:
    """Rough token count: ~4 chars per token."""
    return len(str(messages)) // 4


# -- Layer 1: micro_compact - replace old tool results with placeholders --
def micro_compact(messages: list) -> list:
    # Collect (msg_index, part_index, tool_result_dict) for all tool_result entries
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part_idx, part in enumerate(msg["content"]):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append((msg_idx, part_idx, part))
    if len(tool_results) <= KEEP_RECENT:
        print(f"{DIM}[micro_compact] {len(tool_results)} tool_results ≤ {KEEP_RECENT}, skip{RESET}")
        return messages
    # Find tool_name for each result by matching tool_use_id in prior assistant messages
    tool_name_map = {}
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "type") and block.type == "tool_use":
                        tool_name_map[block.id] = block.name
    # Clear old results (keep last KEEP_RECENT)
    to_clear = tool_results[:-KEEP_RECENT]
    cleared = 0
    for _, _, result in to_clear:
        if isinstance(result.get("content"), str) and len(result["content"]) > 100:
            tool_id = result.get("tool_use_id", "")
            tool_name = tool_name_map.get(tool_id, "unknown")
            old_len = len(result["content"])
            result["content"] = f"[Previous: used {tool_name}]"
            cleared += 1
            print(f"{YELLOW}[micro_compact] truncated {tool_name} result: {old_len} chars → placeholder{RESET}")
    if cleared:
        print(f"{YELLOW}[micro_compact] cleared {cleared}/{len(to_clear)} old results, kept recent {KEEP_RECENT}{RESET}")
    return messages


# -- Layer 2: auto_compact - save transcript, summarize, replace messages --
def auto_compact(messages: list) -> list:
    tokens_before = estimate_tokens(messages)
    msg_count = len(messages)
    print(f"{RED}{'='*60}")
    print(f"[auto_compact] TRIGGERED — {tokens_before} tokens, {msg_count} messages")
    print(f"{'='*60}{RESET}")
    # Save full transcript to disk
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    print(f"{DIM}[auto_compact] transcript saved: {transcript_path}{RESET}")
    # Ask LLM to summarize
    conversation_text = json.dumps(messages, default=str)[:80000]
    print(f"{DIM}[auto_compact] asking LLM to summarize ({len(conversation_text)} chars)...{RESET}")
    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content":
            "Summarize this conversation for continuity. Include: "
            "1) What was accomplished, 2) Current state, 3) Key decisions made. "
            "Be concise but preserve critical details.\n\n" + conversation_text}],
        max_tokens=2000,
    )
    summary = response.content[0].text
    print(f"{GREEN}[auto_compact] summary ({len(summary)} chars):{RESET}")
    print(f"{DIM}{summary[:500]}{'...' if len(summary) > 500 else ''}{RESET}")
    # Replace all messages with compressed summary
    new_messages = [
        {"role": "user", "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"},
        {"role": "assistant", "content": "Understood. I have the context from the summary. Continuing."},
    ]
    tokens_after = estimate_tokens(new_messages)
    print(f"{GREEN}[auto_compact] {tokens_before} → {tokens_after} tokens ({msg_count} msgs → 2 msgs, {(1-tokens_after/tokens_before)*100:.0f}% reduced){RESET}")
    return new_messages


# -- Tool implementations --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "compact":    lambda **kw: "Manual compression requested.",
}

TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "compact", "description": "Trigger manual conversation compression.",
     "input_schema": {"type": "object", "properties": {"focus": {"type": "string", "description": "What to preserve in the summary"}}}},
]


def agent_loop(messages: list):
    round_num = 0
    while True:
        round_num += 1
        # Layer 1: micro_compact before each LLM call
        tokens_now = estimate_tokens(messages)
        print(f"\n{CYAN}{'='*60}")
        print(f"[Round {round_num}] messages={len(messages)}, ~{tokens_now} tokens, threshold={THRESHOLD}")
        print(f"{'='*60}{RESET}")
        micro_compact(messages)
        # Layer 2: auto_compact if token estimate exceeds threshold
        tokens_after_micro = estimate_tokens(messages)
        if tokens_after_micro > THRESHOLD:
            print(f"{RED}[auto_compact] tokens {tokens_after_micro} > {THRESHOLD}, triggering...{RESET}")
            messages[:] = auto_compact(messages)
        else:
            print(f"{DIM}[token check] {tokens_after_micro} ≤ {THRESHOLD}, no auto_compact needed{RESET}")
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        # Log response
        print(f"{YELLOW}[response] stop_reason={response.stop_reason}{RESET}")
        for block in response.content:
            if hasattr(block, "text"):
                print(f"{DIM}  [text] {block.text[:300]}{RESET}")
            elif hasattr(block, "name"):
                print(f"{DIM}  [tool_use] {block.name}({json.dumps(block.input, ensure_ascii=False)[:200]}){RESET}")
        # Bedrock fix: check actual tool_use blocks, not just stop_reason
        tool_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_blocks:
            return
        results = []
        manual_compact = False
        for block in tool_blocks:
            if block.name == "compact":
                manual_compact = True
                output = "Compressing..."
            else:
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
            print(f"{GREEN}> {block.name}: {str(output)[:200]}{RESET}")
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
        messages.append({"role": "user", "content": results})
        # Layer 3: manual compact triggered by the compact tool
        if manual_compact:
            print(f"{RED}[Layer 3: manual compact] model requested compression{RESET}")
            messages[:] = auto_compact(messages)
        if response.stop_reason != "tool_use":
            return


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
