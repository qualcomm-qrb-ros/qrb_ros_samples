# =============================================================================
# VIBE CODING WARNING
# =============================================================================
# You are running an “LLM-driven automation script”: the model can generate tool calls (Read/Write/Edit/Glob/Grep/Bash)
# and will cause real side effects on your local filesystem (creating/updating markdown files). Do NOT “run by vibes”.
#
# Read before running (recommended in order):
# 1) Run in an isolated directory: use a fresh folder or a container to avoid accidental changes to real projects.
# 2) Know what can be written: this script may create/update task_plan.md, findings.md, progress.md, SKILL.md.
# 3) Validate tool args: for Read/Grep, path MUST be a file path; directories trigger errors (logged to progress.md).
# 4) External commands: Grep may invoke system grep (via subprocess). Ensure grep exists; otherwise it falls back to Python.
# 5) Watch outputs and logs: each round prints prompts and logs key events to progress.md (errors, file update events).
# 6) If you see loops or meaningless actions: inspect progress.md and recent_tool_outputs, stop and tighten prompt constraints.
#
# You own the outcome:
# - LLM output may be wrong, repetitive, or unsafe for your intent; review planning files regularly.
# - Automated writes can be irreversible; use version control (git) to enable easy rollback.
# =============================================================================


# This project is inspired and refer from : https://github.com/OthmanAdi/planning-with-files/
# Thanks to OthmanAdi and his efforts on replicating manus-like context engineering


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

user_goal = "generate a product definition document for children story teller pad with aigc UI."

import os
import re
import json
import glob
from datetime import datetime
import textwrap

# =============================================================================
# 1) 基本配置 / Basic configuration
# =============================================================================

# 中文：规划文件必须在项目根目录（符合 SKILL.md 的要求）。
# English: Planning files must live in the project root (per SKILL.md requirement).
PROJECT_DIR = os.getcwd()

# 中文：三大规划文件路径。
# English: Paths for the three planning files.
TASK_PLAN_FILE = os.path.join(PROJECT_DIR, "task_plan.md")
FINDINGS_FILE = os.path.join(PROJECT_DIR, "findings.md")
PROGRESS_FILE = os.path.join(PROJECT_DIR, "progress.md")

# 中文：本地 SKILL.md 用于 LLM 通过 Read/Grep 学习规则（替代 WebSearch/WebFetch）。
# English: Local SKILL.md for LLM to learn rules via Read/Grep (instead of WebSearch/WebFetch).
SKILL_FILE = os.path.join(PROJECT_DIR, "SKILL.md")

# 中文：2-Action Rule 阈值：每 2 次“查看类动作”就必须落盘。
# English: 2-Action Rule threshold: flush to disk after every 2 "view-like actions".
VIEW_ACTION_LIMIT = 2

# 中文：可选节点列表（Director 会从中选择本轮需要的节点）。
# English: Candidate nodes (Director selects the nodes needed for the current round).
AVAILABLE_NODES = ["planner", "researcher", "executor", "reviewer"]


# =============================================================================
# 2) 基础 I/O 工具 / Basic I/O helpers
# =============================================================================

def now_ts():
    # 中文：返回时间戳字符串。
    # English: Return a timestamp string.
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def now_date():
    # 中文：返回日期字符串。
    # English: Return a date string.
    return datetime.now().strftime("%Y-%m-%d")


def read_text(path):
    # 中文：读取文本文件，不存在则返回空字符串。
    # English: Read a text file; return empty string if it does not exist.
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path, content):
    # 中文：写入文本文件（覆盖）。
    # English: Write a text file (overwrite).
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def append_text(path, content):
    # 中文：追加写入文本文件。
    # English: Append text to a file.
    old = read_text(path)
    write_text(path, old + content)


def head_n_lines(text, n=30):
    # 中文：取前 n 行，模拟 hook 的 head -30。
    # English: Take first n lines, simulating head -30 hook.
    return "\n".join(text.splitlines()[:n])


def tail_n_lines(text, n=40):
    # 中文：取后 n 行，避免 prompt 过长。
    # English: Take last n lines to keep prompts short.
    return "\n".join(text.splitlines()[-n:])


# =============================================================================
# 3) 生成本地 SKILL.md / Create local SKILL.md
# =============================================================================

def ensure_skill_file():
    if not os.path.exists(SKILL_FILE):
        print("cannot find './SKILL.md' , please double confirm")
        exit()


# =============================================================================
# 4) 由 user_goal 初始化 planning 文件 / Initialize planning files from user_goal
# =============================================================================

def derive_title_from_goal(user_goal, max_len=60):
    # 中文：从 user_goal 派生一个短标题，用于 task_plan.md 标题行。
    # English: Derive a short title from user_goal for task_plan.md header.
    title = ""
    if not user_goal:
        print("user_goal is NOT SET")
        exit()
    title = re.sub(r"\s+", " ", user_goal.strip())
    if len(title) > max_len:
        title = title[:max_len].rstrip() + "..."
    return title

# refer from : https://github.com/OthmanAdi/planning-with-files/blob/master/scripts/init-session.sh
def create_planning_mds(user_goal, title, date_str):
    
    if not os.path.exists(TASK_PLAN_FILE):
        # 中文：首次创建 task_plan.md 时直接写入 goal 与阶段结构。
        # English: On first creation, write the goal and phase structure directly.
        content = textwrap.dedent(f"""\
        # Task Plan: {title}

        ## Goal
        {user_goal}

        ## Current Phase
        Phase 1

        ## Phases

        ### Phase 1: Requirements & Discovery
        - [ ] Understand user intent and constraints
        - [ ] Read local references (SKILL.md) and extract rules
        - [ ] Write key findings to findings.md
        - **Status:** in_progress

        ### Phase 2: Planning & Structure
        - [ ] Define approach, nodes, and prompt assembly plan
        - [ ] Confirm tool protocol and file update rules
        - **Status:** pending

        ### Phase 3: Implementation
        - [ ] Execute the plan using allowed tools
        - [ ] Follow the 2-action rule strictly
        - **Status:** pending

        ### Phase 4: Testing & Verification
        - [ ] Verify requirements and behavior
        - [ ] Record test results in progress.md
        - **Status:** pending

        ### Phase 5: Delivery
        - [ ] Review outputs and planning files
        - [ ] Deliver final artifact
        - **Status:** pending

        ## Decisions Made
        | Decision | Rationale |
        |----------|-----------|
        | Use file-based planning (task_plan/findings/progress) | Persistent working memory on disk |

        ## Errors Encountered
        | Error | Resolution |
        |-------|------------|
        """)
        write_text(TASK_PLAN_FILE, content)

    if not os.path.exists(FINDINGS_FILE):
        # 中文：首次创建 findings.md 时写入 goal 作为 Requirement。
        # English: On first creation, write the goal as a requirement entry.
        content = textwrap.dedent(f"""\
        # Findings & Decisions

        ## Requirements
        - Goal: {user_goal}

        ## Research Findings
        -

        ## Technical Decisions
        | Decision | Rationale |
        |----------|-----------|

        ## Issues Encountered
        | Issue | Resolution |
        |-------|------------|

        ## Resources
        - SKILL.md (local reference)
        """)
        write_text(FINDINGS_FILE, content)

    if not os.path.exists(PROGRESS_FILE):
        # 中文：首次创建 progress.md 时写入会话与当前状态。
        # English: On first creation, write session and current status.
        content = textwrap.dedent(f"""\
        # Progress Log

        ## Session: {date_str}

        ### Current Status
        - **Goal:** {title}
        - **Phase:** 1 - Requirements & Discovery
        - **Started:** {date_str}

        ### Actions Taken
        - Initialized planning files from user_goal

        ### Test Results
        | Test | Expected | Actual | Status |
        |------|----------|--------|--------|

        ### Errors
        | Error | Resolution |
        |-------|------------|
        """)
        write_text(PROGRESS_FILE, content)


def init_planning_files_from_goal(user_goal):
    # 中文：Create Plan First：按 user_goal 创建 task_plan/findings/progress（不使用占位符模板）。
    # English: Create Plan First: create task_plan/findings/progress from user_goal (no placeholder template).

    title = derive_title_from_goal(user_goal)
    date_str = now_date()
    create_planning_mds(user_goal, title, date_str)

# =============================================================================
# 5) Round ID 自动递增 / Auto-increment Round ID
# =============================================================================

def get_next_round_id():
    # 中文：从 progress.md 解析已执行的 Round 号，返回 max+1（断点恢复友好）。
    # English: Parse executed round numbers from progress.md and return max+1 (resume-safe).
    txt = read_text(PROGRESS_FILE)
    rounds = re.findall(r"^## Round\s+(\d+)\b", txt, flags=re.M)
    if not rounds:
        return 1
    return max(int(x) for x in rounds) + 1


def log_round_start(round_id, objective, nodes):
    # 中文：在 progress.md 记录本轮开始信息。
    # English: Log round start info into progress.md.
    append_text(
        PROGRESS_FILE,
        textwrap.dedent(f"""\

        ## Round {round_id}
        - [{now_ts()}] Round started
        - Objective: {objective}
        - Nodes: {", ".join(nodes) if nodes else "(none)"}
        """)
    )


# =============================================================================
# 6) 工具系统 + hooks / Tool system + hooks
# =============================================================================

# 中文：allowed-tools 仅保留 Read/Write/Edit/Bash/Glob/Grep。
# English: allowed-tools includes only Read/Write/Edit/Bash/Glob/Grep.
ALLOWED_TOOLS = {"Read", "Write", "Edit", "Bash", "Glob", "Grep"}

# 中文：PreToolUse hook 覆盖所有允许工具。
# English: PreToolUse hook applies to all allowed tools.
PRE_HOOK_TOOLS = {"Read", "Write", "Edit", "Bash", "Glob", "Grep"}

# 中文：PostToolUse hook 仅对 Write/Edit 生效。
# English: PostToolUse hook applies only to Write/Edit.
POST_HOOK_TOOLS = {"Write", "Edit"}


def pre_tool_use_hook(tool_name):
    # 中文：模拟 PreToolUse：每次工具调用前打印 task_plan.md 前 30 行。
    # English: Simulate PreToolUse: print first 30 lines of task_plan.md before tool calls.
    if tool_name in PRE_HOOK_TOOLS:
        plan_head = head_n_lines(read_text(TASK_PLAN_FILE), 30)
        print("=" * 60)
        print("\n[PreToolUse HOOK] task_plan.md head -30\n" + "-" * 60)
        print(plan_head if plan_head.strip() else "(task_plan.md empty)")
        print("=" * 60)
    else :
        print("LLM used UN-DEFINED tools in PRE HOOK.")
        exit()

def post_tool_use_hook(tool_name):
    # 中文：模拟 PostToolUse：Write/Edit 后提示更新 phase 状态。
    # English: Simulate PostToolUse: remind updating phase status after Write/Edit.
    if tool_name in POST_HOOK_TOOLS:
        print("\n[planning-with-files] File updated. If this completes a phase, update task_plan.md status.")


def tool_call(tool_name, **kwargs):
    # 中文：统一工具调用入口：校验允许工具、执行 hooks、执行工具、再执行 hooks。
    # English: Unified tool call entry: validate tool, run hooks, execute tool, run hooks.

    if tool_name not in ALLOWED_TOOLS:
        raise ValueError(f"Tool not allowed: {tool_name}")

    pre_tool_use_hook(tool_name)

    result = None

    if tool_name == "Read":
        # 中文：Read 读取 path。
        # English: Read reads a file at path.
        path = kwargs.get("path", "")
        result = read_text(path)

    elif tool_name == "Write":
        # 中文：Write 覆盖写入 path。
        # English: Write overwrites file at path.
        path = kwargs.get("path", "")
        content = kwargs.get("content", "")
        write_text(path, content)
        result = f"Wrote {len(content)} chars to {path}"

    elif tool_name == "Edit":
        # 中文：Edit 做一次字符串替换（初学者友好实现）。
        # English: Edit performs a single string replacement (beginner-friendly).
        path = kwargs.get("path", "")
        old = kwargs.get("old", "")
        new = kwargs.get("new", "")
        text = read_text(path)
        if old not in text:
            raise ValueError("Edit failed: old text not found")
        text = text.replace(old, new, 1)
        write_text(path, text)
        result = f"Edited {path}: replaced one occurrence."

    elif tool_name == "Glob":
        # 中文：Glob 返回匹配 pattern 的文件列表（不递归）。
        # English: Glob returns files matching pattern (non-recursive).
        pattern = kwargs.get("pattern", "*")
        result = glob.glob(os.path.join(PROJECT_DIR, pattern))

    elif tool_name == "Grep":
        # 中文：Grep 优先调用系统 grep（更简单直接），失败则回退到 Python 子串匹配。
        # English: Grep prefers system grep (simpler/direct), and falls back to Python substring match on failure.
        import subprocess

        path = kwargs.get("path", "")
        pattern = kwargs.get("pattern", "*")

        # 中文：调用 grep：-n 输出行号；-- 防止 keyword 被当成参数；text=True 返回字符串。
        # English: Call grep: -n prints line numbers; -- prevents keyword from being parsed as an option; text=True returns strings.
        cp = subprocess.run(
            ["grep", "-r", pattern, path],
            capture_output=True,
            text=True
        )

        if cp.returncode == 0:
            # 中文：匹配到内容：按行拆分返回。
            # English: Matches found: split stdout into lines.
            result = cp.stdout.splitlines()

        elif cp.returncode == 1:
            # 中文：没有匹配：这是 grep 的正常返回码。
            # English: No matches: this is grep's normal return code.
            result = []

        else:
            # 中文：其他返回码代表真正错误：抛出异常并带 stderr。
            # English: Other return codes mean real errors: raise with stderr.
            raise RuntimeError(f"grep failed rc={cp.returncode}: {cp.stderr.strip()}")

  
        # 中文：防止结果过大：最多返回前 200 条命中。
        # English: Prevent huge outputs: return at most the first 200 matches.
        result = result[:200]

    elif tool_name == "Bash":
        # 中文：Bash 出于安全仅模拟，不执行真实命令。
        # English: Bash is simulated for safety; it does not execute real commands.
        cmd = kwargs.get("cmd", "")
        # ONLY FOR DEMO , NEED TO USE SANDBOX in future
        import subprocess
        r = subprocess.run(
            ["bash","-c", cmd],
            cwd="./",
            capture_output=True,   # 等价于 stdout=PIPE, stderr=PIPE
            text=True,             # 输出为 str（否则是 bytes）
        )
        all_output = (r.stdout or "") + (r.stderr or "")
        print(all_output)

    post_tool_use_hook(tool_name)
    return result


# =============================================================================
# 7) 2-Action Rule（本地查看类工具）/ 2-Action Rule (local view-like tools)
# =============================================================================

# 中文：将 Read/Glob/Grep 视为“查看类动作”；但读 planning 文件不计数。
# English: Treat Read/Glob/Grep as view-like actions; reading planning files does not count.
VIEW_TOOLS = {"Read", "Glob", "Grep"}

# 中文：用于 2-action rule 的计数与缓冲区。
# English: Counters and buffers for 2-action rule.
view_action_count = 0
buffer_findings = []
buffer_progress = []


def is_view_action(tool_name, args):
    # 中文：Read planning 文件不算 view；读 SKILL.md 或其它文件算 view。
    # English: Reading planning files is not a view action; reading SKILL.md/others is a view action.
    if tool_name not in VIEW_TOOLS:
        return False

    if tool_name == "Read":
        path = (args or {}).get("path", "")
        planning_paths = [TASK_PLAN_FILE, FINDINGS_FILE, PROGRESS_FILE]
        if os.path.abspath(path) in [os.path.abspath(p) for p in planning_paths]:
            return False

    return True


def add_finding_buffer(line):
    # 中文：将发现先写入缓冲，等待触发 flush。
    # English: Add a finding to buffer, waiting for flush.
    global buffer_findings
    buffer_findings.append(f"- [{now_ts()}] {line}\n")


def add_progress_buffer(line):
    # 中文：将日志先写入缓冲，等待触发 flush。
    # English: Add progress log to buffer, waiting for flush.
    global buffer_progress
    buffer_progress.append(f"- [{now_ts()}] {line}\n")


def flush_buffers_to_disk():
    # 中文：将缓冲写入 findings/progress 并清空计数。
    # English: Flush buffers to findings/progress and reset counters.
    global view_action_count, buffer_findings, buffer_progress

    if buffer_findings:
        append_text(FINDINGS_FILE, "\n## Auto Flush Findings\n" + "".join(buffer_findings))
    if buffer_progress:
        append_text(PROGRESS_FILE, "\n## Auto Flush Progress\n" + "".join(buffer_progress))

    view_action_count = 0
    buffer_findings = []
    buffer_progress = []


def record_view_action(tool_name, args):
    # 中文：记录一次 view-like 工具调用，达到阈值即 flush。
    # English: Record one view-like tool call; flush when reaching threshold.
    global view_action_count
    view_action_count += 1
    add_progress_buffer(f"VIEW: {tool_name} args={args}")

    if view_action_count >= VIEW_ACTION_LIMIT:
        add_progress_buffer("2-Action Rule triggered => flushing buffered findings/progress to disk.")
        flush_buffers_to_disk()


# =============================================================================
# 8) 磁盘上下文加载 / Load disk context
# =============================================================================

def load_disk_context_for_prompt():
    # 中文：为 prompt 提供磁盘上下文（plan 取 head，findings/progress 取 tail）。
    # English: Provide disk context for prompts (plan head; findings/progress tail).
    plan = read_text(TASK_PLAN_FILE)
    findings = read_text(FINDINGS_FILE)
    progress = read_text(PROGRESS_FILE)
    return {
        "task_plan_head": head_n_lines(plan, 30),
        "findings_tail": tail_n_lines(findings, 40),
        "progress_tail": tail_n_lines(progress, 40),
    }


# =============================================================================
# 9) Prompt 规范（全英文）/ Prompt specs (all English)
# =============================================================================

# 中文：系统规则（英文），用于强制 planning-with-files 纪律。
# English: System rules (English) enforcing planning-with-files discipline.
SYSTEM_RULES_EN = textwrap.dedent("""\
You must follow "planning-with-files" discipline:
- Treat filesystem as persistent memory: write important info to task_plan.md/findings.md/progress.md.
- Create plan first (task_plan.md) before complex tasks.
- 2-Action Rule: after every 2 view/search operations, save key findings to disk immediately.
- Read Before Decide: before major decisions, read task_plan.md to keep goals in attention.
- Update After Act: after completing a phase, mark status complete and log errors/changes.
- Log ALL errors to task_plan.md. Never repeat exact same failing action. Use 3-strike protocol.
""").strip()

# 中文：节点角色说明（英文）。
# English: Node role descriptions (English).
NODE_ROLES_EN = {
    "planner": "Plan phases, decide next actions, and keep task_plan.md consistent and minimal.",
    "researcher": "Inspect local references (e.g., SKILL.md) and extract key rules into findings.md.",
    "executor": "Carry out concrete steps using allowed tools and log progress to progress.md.",
    "reviewer": "Review file states, detect gaps, and propose fixes and phase completion updates.",
}

# 中文：工具协议（英文），TOOL_CALL 必须仅输出该块，否则解析失败。
# English: Tool protocol (English); TOOL_CALL must be the only output block when used.
TOOL_PROTOCOL_EN = textwrap.dedent("""\
TOOL USAGE PROTOCOL (must follow exactly):
- If you need a tool, output exactly ONE tool call and NOTHING ELSE.
- Use this exact format (YAML shell + JSON args):

TOOL_CALL:
  name: <one of: Read, Write, Edit, Bash, Glob, Grep>
  args: {"key":"value", ...}

- No explanations, no extra text, no code fences.

- If you do NOT need a tool, output:

FINAL:
- bullet points...

WRITE_FINDINGS:
- lines to append into findings.md (optional)

WRITE_PROGRESS:
- lines to append into progress.md (optional)
""").strip()

# 中文：Director 协议（英文），用于生成 round_objective 与节点列表。
# English: Director protocol (English) to generate round_objective and node list.
DIRECTOR_PROTOCOL_EN = textwrap.dedent("""\
DIRECTOR OUTPUT PROTOCOL (must follow exactly):
- If the overall task is NOT done, output ONLY a single line:

ROUND_PLAN_JSON: {"objective":"...", "nodes":["planner","researcher",...], "stop": false}

- If the overall task IS done, output ONLY a single line:

ROUND_PLAN_JSON: {"objective":"...", "nodes":[], "stop": true}

- Do not output anything else besides that single line.
""").strip()


def build_director_prompt(user_goal, disk_ctx):
    # 中文：构造 Director prompt（英文），让 LLM 生成下一轮 objective 与 nodes。
    # English: Build Director prompt (English) so LLM generates next round objective and nodes.
    prompt = textwrap.dedent(f"""\
    SYSTEM:
    {SYSTEM_RULES_EN}

    ROLE:
    You are the Director of a multi-round task execution. You decide the next round's objective and which nodes should run.

    USER GOAL:
    {user_goal}

    CONSTRAINTS:
    - Allowed tools: Read, Write, Edit, Bash, Glob, Grep (no web tools).
    - Use planning files as persistent memory.
    - Keep objectives concrete and incremental.
    - Avoid repeating identical tool calls; prefer Grep for targeted extraction instead of repeated Read.

    DISK CONTEXT: task_plan.md (head)
    {disk_ctx["task_plan_head"]}

    DISK CONTEXT: findings.md (tail)
    {disk_ctx["findings_tail"]}

    DISK CONTEXT: progress.md (tail)
    {disk_ctx["progress_tail"]}

    AVAILABLE NODES:
    {", ".join(AVAILABLE_NODES)}

    {DIRECTOR_PROTOCOL_EN}
    """).strip()
    return prompt


def build_node_prompt(round_id, node, user_goal, round_objective, recent_tool_outputs, disk_ctx):
    # 中文：构造节点 prompt（英文），并在运行时打印，展示每轮如何组装 prompt。
    # English: Build node prompt (English), printed at runtime to show prompt assembly each round.
    role = NODE_ROLES_EN.get(node, "Be helpful.")

    # 中文：把 RECENT TOOL OUTPUTS 包在代码块中，且已做“单行摘要+去重计数”，避免结构被打爆。
    # English: Wrap RECENT TOOL OUTPUTS in a code block with single-line summaries and de-dup counts.
    if recent_tool_outputs:
        tool_summary_lines = "\n".join([f"{k}: {v}" for k, v in recent_tool_outputs.items()])
        tool_summary = "```text\n" + tool_summary_lines + "\n```"
    else:
        tool_summary = "```text\n(none)\n```"

    prompt = textwrap.dedent(f"""\
    SYSTEM:
    {SYSTEM_RULES_EN}

    NODE: {node}
    ROLE:
    {role}

    USER GOAL:
    {user_goal}

    ROUND ID:
    {round_id}

    ROUND OBJECTIVE:
    {round_objective}

    RECENT TOOL OUTPUTS:
    {tool_summary}

    DISK CONTEXT: task_plan.md (head)
    {disk_ctx["task_plan_head"]}

    DISK CONTEXT: findings.md (tail)
    {disk_ctx["findings_tail"]}

    DISK CONTEXT: progress.md (tail)
    {disk_ctx["progress_tail"]}

    {TOOL_PROTOCOL_EN}
    """).strip()

    banner = f"\n\n{'=' * 90}\nPROMPT (Round {round_id} | Node: {node})\n{'=' * 90}\n"
    return banner + prompt + "\n"


# =============================================================================
# 10) 输出解析与稳健提取 / Parsing with robustness
# =============================================================================

def extract_first_json_object(text, start_pos):
    # 中文：从 start_pos 起扫描括号，提取第一个完整 JSON 对象字符串。
    # English: Scan braces from start_pos to extract the first complete JSON object string.
    i = text.find("{", start_pos)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j + 1]
    return None


def parse_round_plan_json(text):
    # 中文：解析 Director 输出，允许存在多余文本，提取 ROUND_PLAN_JSON 后的 JSON。
    # English: Parse Director output tolerantly by extracting JSON following ROUND_PLAN_JSON.
    idx = text.find("ROUND_PLAN_JSON:")
    if idx < 0:
        raise ValueError("Director output parse error: missing ROUND_PLAN_JSON")

    json_str = extract_first_json_object(text, idx)
    if not json_str:
        raise ValueError("Director output parse error: cannot extract JSON object")

    obj = json.loads(json_str)
    if "objective" not in obj or "nodes" not in obj or "stop" not in obj:
        raise ValueError("Director output parse error: missing fields in JSON")
    return obj


def parse_tool_call(text):
    # 中文：解析 TOOL_CALL，允许存在多余文本，优先从 TOOL_CALL: 之后解析。
    # English: Parse TOOL_CALL tolerantly by focusing on the segment after TOOL_CALL:.
    idx = text.find("TOOL_CALL:")
    if idx < 0:
        return None

    segment = text[idx:]
    m_name = re.search(r"name:\s*([A-Za-z]+)", segment)
    if not m_name:
        raise ValueError("TOOL_CALL parse error: missing name")
    tool_name = m_name.group(1).strip()

    m_args = re.search(r"args:\s*(\{.*\})", segment, flags=re.S)
    args = json.loads(m_args.group(1).strip()) if m_args else {}
    return tool_name, args


def parse_final_writes(text):
    # 中文：解析 FINAL/WRITE_FINDINGS/WRITE_PROGRESS 块，提取写入内容（行前缀为 '-'）。
    # English: Parse FINAL/WRITE_FINDINGS/WRITE_PROGRESS blocks and extract lines (prefixed with '-').

    def extract_block(block_name):
        pattern = rf"{block_name}:\s*\n(.*?)(\n[A-Z_]+:\s*\n|\Z)"
        m = re.search(pattern, text, flags=re.S)
        if not m:
            return []
        block = m.group(1).strip()
        if not block:
            return []
        lines = []
        for ln in block.splitlines():
            ln = ln.strip()
            if ln.startswith("-"):
                lines.append(ln[1:].strip())
        return lines

    findings_lines = extract_block("WRITE_FINDINGS")
    progress_lines = extract_block("WRITE_PROGRESS")
    return findings_lines, progress_lines


def summarize_tool_result(result, limit=200):
    # 中文：把工具结果压缩为单行摘要，去掉换行与多余空白，避免 prompt 被撑爆。
    # English: Convert tool results into a single-line summary by removing newlines and extra whitespace.

    if result is None:
        return "(no result)"

    if isinstance(result, list):
        # 中文：列表只展示长度与前三项预览（单行）。
        # English: For lists, show length and a short preview (single line).
        preview = "; ".join([str(x).replace("\n", " ")[:60] for x in result[:3]])
        s = f"list(len={len(result)}): {preview}"
    else:
        # 中文：字符串/其他对象转字符串，并压缩空白为单空格。
        # English: Convert to string and compress whitespace into single spaces.
        s = str(result)

    # 中文：把所有换行/制表符等空白压成单空格。
    # English: Collapse all whitespace (including newlines/tabs) into a single space.
    s = re.sub(r"\s+", " ", s).strip()

    # 中文：截断到 limit，避免 prompt 变长。
    # English: Truncate to limit to avoid prompt growth.
    return s[:limit] + ("..." if len(s) > limit else "")


# =============================================================================
# 11) 3-Strike Error Protocol / 3-Strike Error Protocol
# =============================================================================

def append_error_to_task_plan(error, resolution):
    # 中文：向 task_plan.md 的 Errors Encountered 表追加一行。
    # English: Append one row to Errors Encountered table in task_plan.md.
    append_text(TASK_PLAN_FILE, f"| {error} | {resolution} |\n")


def do_with_3_strike(action_name, action_func):
    # 中文：执行动作最多 3 次，失败则按 3-strike 协议写盘并升级提示。
    # English: Try an action up to 3 times; log failures per 3-strike protocol and escalate after 3 failures.
    for attempt in [1, 2, 3]:
        try:
            return action_func(attempt)
        except Exception as e:
            err_name = type(e).__name__
            msg = str(e)

            if attempt == 1:
                resolution = "Attempt1: Diagnose & Fix (adjust inputs/guards)"
            elif attempt == 2:
                resolution = "Attempt2: Alternative approach (change method/tool; avoid repeating same failure)"
            else:
                resolution = "Attempt3: Broader rethink (reduce scope/revisit assumptions)"

            append_error_to_task_plan(f"{action_name}:{err_name}", resolution)
            append_text(PROGRESS_FILE, textwrap.dedent(f"""\

            ## Error Log
            - [{now_ts()}] ACTION FAILED: {action_name}
              Attempt: {attempt}
              Error: {err_name}
              Message: {msg}
              Planned Resolution: {resolution}
            """))

            if attempt == 3:
                print("\n[ESCALATE TO USER] After 3 failures, need guidance.")
                print(f"Action: {action_name}")
                print(f"Last error: {err_name}: {msg}")
                return None

# =============================================================================
# 12) Stop hook（简化 check-complete）/ Stop hook (simplified check-complete)
# =============================================================================

def stop_hook_check_complete():
    # 中文：若 task_plan.md 中仍有 pending/in_progress，则认为未完成。
    # English: If task_plan.md still has pending/in_progress, it is not complete.
    plan = read_text(TASK_PLAN_FILE)
    remaining = re.findall(r"\*\*Status:\*\*\s*(pending|in_progress)", plan)
    if remaining:
        print("\n[Stop hook] Not complete yet. Remaining statuses:", remaining)
        print("Tip: update remaining phases to complete in task_plan.md")
        return False
    print("\n[Stop hook] All phases look complete. ✅")
    return True


# =============================================================================
# 14) LLM 工具回路（节点级）/ LLM tool loop (node-level)
# =============================================================================

def llm_node_tool_loop(round_id, node, user_goal, round_objective, max_steps=10):
    # 中文：节点循环：LLM 产出 TOOL_CALL 或 FINAL；TOOL_CALL 会被执行并回填上下文。
    # English: Node loop: LLM emits TOOL_CALL or FINAL; TOOL_CALL is executed and fed back as context.

    # 中文：保存“去重后的工具输出”用于 prompt。
    # English: Store de-duplicated tool outputs for prompts.
    recent_tool_outputs = {}

    # 中文：相同 tool+args 的摘要缓存与计数（用于 xN）。
    # English: Cache and count identical tool+args summaries (for xN).
    call_cache = {}
    call_counts = {}

    # 中文：重复 Read 同一文件的护栏计数（单独维护，避免与去重计数混淆）。
    # English: Separate guardrail counter for repeated Read on the same file.
    read_repeat_counts = {}

    for step in range(1, max_steps + 1):
        disk_ctx = load_disk_context_for_prompt()
        prompt = build_node_prompt(round_id, node, user_goal, round_objective, recent_tool_outputs, disk_ctx)

        # print(prompt)
        print("\n[LLM OUTPUT STREAM]\n" + "-" * 60)
        llm_text = call_llm(prompt)
        print("\n" + "-" * 60 + "\n[END LLM OUTPUT]\n")

        tool_req = parse_tool_call(llm_text)
        if tool_req is not None:
            tool_name, args = tool_req

            # 中文：护栏：同一节点内重复 Read 同一路径超过 2 次则阻止，并引导用 Grep。
            # English: Guardrail: block repeated Read of the same path > 2 times and suggest Grep.
            if tool_name == "Read":
                p = os.path.abspath((args or {}).get("path", ""))
                read_repeat_counts[p] = read_repeat_counts.get(p, 0) + 1
                if read_repeat_counts[p] > 2:
                    append_text(PROGRESS_FILE, f"\n- [{now_ts()}] Guardrail: repeated Read({p}) > 2 times; suggest using Grep for targeted extraction.\n")
                    recent_tool_outputs["Guardrail"] = f"Repeated Read({p}) too many times. Use Grep(path={p}, keyword=...) instead."
                    continue

            # 中文：执行工具，并拿到原始结果。
            # English: Execute the tool and get the raw result.
            result = tool_call(tool_name, **args)

            # 中文：以 (tool_name + args_json) 作为同一调用签名，用于去重计数。
            # English: Use (tool_name + args_json) as signature for de-dup counting.
            args_json = json.dumps(args, sort_keys=True, ensure_ascii=False)
            sig = (tool_name, args_json)

            call_counts[sig] = call_counts.get(sig, 0) + 1
            if sig not in call_cache:
                call_cache[sig] = summarize_tool_result(result)

            # 中文：重建 recent_tool_outputs 为去重后的列表，每条显示 xN。
            # English: Rebuild recent_tool_outputs as a de-duplicated list with xN counts.
            recent_tool_outputs.clear()
            idx = 1
            for (tname, ajson), cnt in call_counts.items():
                summary = call_cache[(tname, ajson)]
                recent_tool_outputs[f"{tname}#{idx}"] = f"{summary} | args={ajson} | x{cnt}"
                idx += 1

            # 中文：2-action rule：对 view-like 工具计数并可能自动 flush。
            # English: 2-action rule: count view-like tools and auto-flush when needed.
            if is_view_action(tool_name, args):
                record_view_action(tool_name, args)
                add_finding_buffer(f"Used {tool_name} with args={args}; stored summary as {tname}#{idx-1}.")

            # 中文：继续下一步，让 LLM 看到更新后的工具输出。
            # English: Continue to let the LLM see updated tool outputs.
            continue

        # 中文：FINAL 输出：解析并写入 findings/progress，然后结束该节点回路。
        # English: FINAL output: parse and write to findings/progress, then end this node loop.
        findings_lines, progress_lines = parse_final_writes(llm_text)

        if findings_lines:
            append_text(
                FINDINGS_FILE,
                "\n## LLM Suggested Findings\n" +
                "\n".join([f"- [{now_ts()}] {x}" for x in findings_lines]) + "\n"
            )

        if progress_lines:
            append_text(
                PROGRESS_FILE,
                "\n## LLM Suggested Progress\n" +
                "\n".join([f"- [{now_ts()}] {x}" for x in progress_lines]) + "\n"
            )

        return llm_text

    # 中文：防止死循环：超过 max_steps 则记录并退出。
    # English: Prevent infinite loops: log and stop after max_steps.
    append_text(PROGRESS_FILE, f"\n- [{now_ts()}] Node loop hit max_steps={max_steps}, stopped.\n")
    return None


# =============================================================================
# 15) Director：让 LLM 生成每轮 objective + nodes / Director: LLM generates round objective + nodes
# =============================================================================

def llm_director_next_round(user_goal):
    # 中文：调用 Director prompt，让 LLM 生成下一轮计划（objective + nodes + stop）。
    # English: Call Director prompt to let LLM generate next round plan (objective + nodes + stop).
    disk_ctx = load_disk_context_for_prompt()
    director_prompt = build_director_prompt(user_goal, disk_ctx)

    print("\n\n" + "=" * 90)
    print("DIRECTOR PROMPT")
    print("=" * 90 + "\n")
    print(director_prompt)
    print("\n[DIRECTOR LLM OUTPUT STREAM]\n" + "-" * 60)

    out = call_llm(director_prompt)

    print("\n" + "-" * 60 + "\n[END DIRECTOR OUTPUT]\n")
    plan = parse_round_plan_json(out)

    # 中文：清洗 nodes：只保留可用节点，去重并保持顺序。
    # English: Sanitize nodes: keep allowed nodes only, deduplicate while preserving order.
    nodes = []
    for n in plan.get("nodes", []):
        if n in AVAILABLE_NODES and n not in nodes:
            nodes.append(n)

    objective = (plan.get("objective", "") or "").strip() or "No objective provided."
    stop = bool(plan.get("stop", False))

    return {"objective": objective, "nodes": nodes, "stop": stop}


# =============================================================================
# 16) 主执行：Round 循环 / Main execution: round loop
# =============================================================================

def run_long_task():
    # 中文：按你要求，不从 CLI 读，直接用 user_goal 变量。
    # English: As requested, do not read from CLI; use user_goal variable directly.

    # 中文：准备本地 SKILL.md 并初始化三大 planning 文件。
    # English: Prepare local SKILL.md and initialize the three planning files.
    ensure_skill_file()
    init_planning_files_from_goal(user_goal)

    # 中文：写入会话开始日志。
    # English: Log session start.
    append_text(PROGRESS_FILE, f"\n## Session Start\n- [{now_ts()}] Session started.\n")

    # 中文：主循环：每轮由 Director 决定 objective 与 nodes，直到 stop=true。
    # English: Main loop: each round is planned by Director (objective + nodes) until stop=true.
    while True:
        round_id = get_next_round_id()

        # 中文：Director 生成下一轮计划（objective + nodes + stop）。
        # English: Director generates the next round plan (objective + nodes + stop).
        plan = llm_director_next_round(user_goal)
        objective = plan["objective"]
        nodes = plan["nodes"]
        stop = plan["stop"]

        # 中文：如果 Director 认为任务已完成，则跳出循环并执行 stop hook。
        # English: If Director says task is done, break and run stop hook.
        if stop:
            append_text(PROGRESS_FILE, f"\n- [{now_ts()}] Director decided to stop. Objective: {objective}\n")
            break

        # 中文：记录本轮开始信息（Round ID 与 objective 由实际执行决定）。
        # English: Log round start (round ID and objective are decided by actual execution).
        log_round_start(round_id, objective, nodes)

        # 中文：按 Director 指定的节点顺序运行；每个节点内部可多步 TOOL_CALL。
        # English: Run nodes in Director-specified order; each node may perform multiple TOOL_CALL steps.
        for node in nodes:
            llm_node_tool_loop(round_id, node, user_goal, objective, max_steps=10)

        # 中文：轮结束后强制 flush（防止缓冲残留）。
        # English: Force flush after each round to avoid leftover buffers.
        if buffer_findings or buffer_progress:
            add_progress_buffer("End of round => flushing remaining buffers.")
            flush_buffers_to_disk()

    # 中文：结束时执行 stop hook 检查 phase 是否全部完成。
    # English: Run stop hook at the end to verify all phases are complete.
    stop_hook_check_complete()

    # 中文：写入 session 结束标记。
    # English: Write session end marker.
    append_text(PROGRESS_FILE, f"\n- [{now_ts()}] Session ended.\n")


# =============================================================================
# main / main
# =============================================================================

if __name__ == "__main__":
    run_long_task()
    print("\nDone. Files in project root:")
    print(" - task_plan.md")
    print(" - findings.md")
    print(" - progress.md")
    print(" - SKILL.md")
