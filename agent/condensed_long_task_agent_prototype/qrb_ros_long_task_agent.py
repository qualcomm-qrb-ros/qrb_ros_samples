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


#!/usr/bin/env python3  # 脚本入口声明 / Script entry declaration
# -*- coding: utf-8 -*-  # 指定UTF-8编码 / Specify UTF-8 encoding


import os
import sys
import json
import re
import html
import hashlib
import subprocess
from datetime import datetime
from urllib import request
from urllib.error import URLError

STATE_FILE = ".pwf_state.json"  # 运行时状态文件 / Runtime state
TASK_PLAN_FILE = "task_plan.md"  # 计划文件（内容为JSON/YAML子集）/ Plan file (JSON, YAML-subset)
FINDINGS_FILE = "findings.md"  # 发现文件 / Findings
PROGRESS_FILE = "progress.md"  # 进度文件 / Progress

MAX_TOOL_OUTPUT_CHARS = 1200
MAX_APPEND_LINES = 30
DEFAULT_TIMEOUT_SEC = 60
MAX_ROUNDS = 200

# =============================
# 1) MiniContainer：工具封装（含脚本工具 + timeout bytes 解码）
#    MiniContainer: tool wrapper (script tools + timeout bytes decode)
# =============================

class MiniContainer:
    def __init__(self, workdir):
        self.workdir = os.path.abspath(workdir)
        self.env = dict(os.environ)

    def _run_script(self, argv, timeout_sec=DEFAULT_TIMEOUT_SEC):
        try:
            r = subprocess.run(
                argv,
                cwd=self.workdir,
                env=self.env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            return {"exit_code": r.returncode, "stdout": r.stdout or "", "stderr": r.stderr or ""}
        except subprocess.TimeoutExpired:
            return {"exit_code": 124, "stdout": "", "stderr": "[TIMEOUT]"}

    def exec_shell(self, command, timeout_sec=DEFAULT_TIMEOUT_SEC):
        try:
            r = subprocess.run(
                command,
                shell=True,
                cwd=self.workdir,
                env=self.env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            return {"exit_code": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
        except subprocess.TimeoutExpired as e:
            out = e.stdout
            err = e.stderr
            if isinstance(out, bytes):
                out = out.decode("utf-8", errors="replace")
            if isinstance(err, bytes):
                err = err.decode("utf-8", errors="replace")
            return {"exit_code": 124, "stdout": out or "", "stderr": (err or "") + "\n[TIMEOUT]"}

    def exec_python(self, code, timeout_sec=DEFAULT_TIMEOUT_SEC):
        try:
            r = subprocess.run(
                [sys.executable, "-"],
                input=code,
                cwd=self.workdir,
                env=self.env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            return {"exit_code": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
        except subprocess.TimeoutExpired as e:
            out = e.stdout
            err = e.stderr
            if isinstance(out, bytes):
                out = out.decode("utf-8", errors="replace")
            if isinstance(err, bytes):
                err = err.decode("utf-8", errors="replace")
            return {"exit_code": 124, "stdout": out or "", "stderr": (err or "") + "\n[TIMEOUT]"}

    def exec_search_web(self, query, topk=5, timeout_sec=DEFAULT_TIMEOUT_SEC):
        argv = [sys.executable, "search_web.py", "--query", str(query), "--topk", str(int(topk))]
        r = self._run_script(argv, timeout_sec=timeout_sec)
        out = (r.get("stdout") or "").strip()
        r["stdout"] = out if out else "no web search result"
        return r

    def exec_search_database(self, query, topk=5, timeout_sec=DEFAULT_TIMEOUT_SEC):
        argv = [sys.executable, "check_database.py", "--query", str(query), "--topk", str(int(topk))]
        r = self._run_script(argv, timeout_sec=timeout_sec)
        out = (r.get("stdout") or "").strip()
        r["stdout"] = out if out else "no database search result"
        return r


def get_container():
    return MiniContainer(os.getcwd())

# =============================
# 2) 文件工具 / File utilities
# =============================


def read_text(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def append_lines(path, lines):
    lines = (lines or [])[:MAX_APPEND_LINES]
    if not lines:
        return
    with open(path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line).rstrip("\n") + "\n")

# =============================
# 3) Task key / 任务指纹
# =============================


def normalize_task_text(s):
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def compute_task_key(user_message):
    norm = normalize_task_text(user_message)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()

# =============================
# 4) Runtime state / 运行时状态
# =============================


def default_state():
    return {
        "last_tool_result": None,
        "actions_since_last_findings": 0,
        "root_user_message": "",
        "plan_created": False,
        "task_key": "",
    }


def load_state():
    raw = read_text(STATE_FILE)
    if not raw.strip():
        return default_state()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return default_state()
    for k, v in default_state().items():
        if k not in obj:
            obj[k] = v
    return obj


def save_state(state):
    write_text(STATE_FILE, json.dumps(state, ensure_ascii=False, indent=2))


def reset_all_files_for_new_task(user_message):
    # 清空文件 / Truncate files
    write_text(TASK_PLAN_FILE, "")
    write_text(FINDINGS_FILE, "")
    write_text(PROGRESS_FILE, "")
    # 重置状态 / Reset state
    st = default_state()
    st["task_key"] = compute_task_key(user_message)
    st["root_user_message"] = user_message
    st["plan_created"] = False
    save_state(st)

# =============================
# 5) findings/progress templates (unchanged)
# =============================


def default_findings_template():
    return (
        "# Findings & Decisions  # 发现与决策 / Findings & decisions\n\n"
        "## Requirements  # 需求 / Requirements\n-  \n\n"
        "## Research Findings  # 研究发现 / Research findings\n-  \n\n"
        "## Technical Decisions  # 技术决策 / Technical decisions\n| Decision | Rationale |\n|---|---|\n\n"
        "## Issues Encountered  # 遇到的问题 / Issues\n| Issue | Resolution |\n|---|---|\n\n"
        "## Resources  # 资源 / Resources\n-  \n"
    )


def default_progress_template():
    today = datetime.now().strftime("%Y-%m-%d")
    return (
        "# Progress Log  # 进度日志 / Progress log\n\n"
        f"## Session: {today}  # 会话 / Session\n\n"
        "### Actions Taken  # 已做动作 / Actions taken\n-  \n\n"
        "### Test Results  # 测试结果 / Test results\n| Test | Expected | Actual | Status |\n|---|---|---|---|\n\n"
        "### Errors  # 错误 / Errors\n| Error | Resolution |\n|---|---|\n\n"
    )


def ensure_findings_and_progress_exist():
    if not os.path.exists(FINDINGS_FILE):
        write_text(FINDINGS_FILE, default_findings_template())
    if not os.path.exists(PROGRESS_FILE):
        write_text(PROGRESS_FILE, default_progress_template())

# =============================
# 6) task_plan dict I/O (NEW) / task_plan 内存dict + dump
# =============================


def default_task_plan_dict(task_key):
    return {
        "meta": {"created": datetime.now().strftime("%Y-%m-%d"), "task_key": task_key},
        "goal": "",
        "current_phase": "Phase 1",
        "phases": [],
        "errors": [],
    }


def load_task_plan_dict():
    raw = read_text(TASK_PLAN_FILE).strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def save_task_plan_dict(plan_dict):
    write_text(TASK_PLAN_FILE, json.dumps(plan_dict, ensure_ascii=False, indent=2) + "\n")


def ensure_task_plan_dict(task_key):
    plan = load_task_plan_dict()
    if plan is None:
        plan = default_task_plan_dict(task_key)
        save_task_plan_dict(plan)
    return plan


def apply_task_plan_updates(plan_dict, updates):
    # updates: {"checklist_set":[{phase_id,item_text,done}], "phase_status":[{phase_id,status}]}
    errs = []
    if not isinstance(updates, dict):
        return errs

    phases = plan_dict.get("phases")
    if not isinstance(phases, list):
        plan_dict["phases"] = []
        phases = plan_dict["phases"]

    # phase_status
    for u in (updates.get("phase_status") or []):
        pid = (u.get("phase_id") or "").strip()
        st = (u.get("status") or "").strip()
        if not pid or st not in ("todo", "in_progress", "complete"):
            continue
        hit = None
        for p in phases:
            if str(p.get("id", "")).strip() == pid:
                hit = p
                break
        if not hit:
            errs.append(f"phase_id not found: {pid}")
            continue
        hit["status"] = st

    # checklist_set
    for u in (updates.get("checklist_set") or []):
        pid = (u.get("phase_id") or "").strip()
        text = (u.get("item_text") or "").strip()
        done = bool(u.get("done", False))
        if not pid or not text:
            continue
        hit = None
        for p in phases:
            if str(p.get("id", "")).strip() == pid:
                hit = p
                break
        if not hit:
            errs.append(f"phase_id not found: {pid}")
            continue
        checklist = hit.get("checklist")
        if not isinstance(checklist, list):
            hit["checklist"] = []
            checklist = hit["checklist"]
        item = None
        for it in checklist:
            if str(it.get("text", "")).strip() == text:
                item = it
                break
        if not item:
            errs.append(f"checklist item not found in {pid}: {text}")
            continue
        item["done"] = done

    return errs

# =============================
# 7) Snapshot extraction (task_plan from dict) / 快照提炼
# =============================


def tail_lines(text, n):
    lines = text.splitlines()
    return lines[-n:] if len(lines) > n else lines


def extract_state_snapshot(task_plan_dict, findings_text, progress_text):
    goal = [str(task_plan_dict.get("goal", "")).strip()] if str(task_plan_dict.get("goal", "")).strip() else []
    current_phase = [str(task_plan_dict.get("current_phase", "")).strip()] if str(task_plan_dict.get("current_phase", "")).strip() else []
    phase_statuses = []
    for p in (task_plan_dict.get("phases") or []):
        pid = str(p.get("id", "")).strip()
        st = str(p.get("status", "")).strip()
        if pid or st:
            phase_statuses.append({"phase": pid, "status": st})
    return {
        "goal": goal,
        "current_phase": current_phase,
        "phase_statuses": phase_statuses,
        "findings_tail": tail_lines(findings_text, 25),
        "progress_tail": tail_lines(progress_text, 25),
    }

# =============================
# 8) JSON safe parse
# =============================


def parse_json_safely(text):
    s = (text or "").strip()
    if not s:
        raise json.JSONDecodeError("Empty input", s, 0)
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
        s = s.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise json.JSONDecodeError("No JSON object found", s, 0)
    return json.loads(html.unescape(s[start:end + 1]))

# =============================
# 9) LLM calling
# =============================


#def call_llm(prompt):


# =============================
# 10) task_plan create prompt (dict)
# =============================


def build_plan_prompt(user_task):
    today = datetime.now().strftime("%Y-%m-%d")
    schema = {"task_plan": "object (dict to dump into task_plan.md)", "notes": "string (optional)"}
    lines = [
        "You are generating task_plan.md as a JSON object (valid YAML 1.2 subset).",
        "Return JSON only.",
        "task_plan schema:",
        "- meta: {created: YYYY-MM-DD, task_key: string}",
        "- goal: string",
        "- current_phase: 'Phase N'",
        "- phases: list of {id:'Phase N', title:string, status:todo|in_progress|complete, checklist:[{text:string, done:boolean}]}",
        "- errors: list of {error:string, resolution:string}",
        f"Use created date: {today}.",
    ]
    prompt = "[INSTRUCTIONS]\n" + "\n".join(lines) + "\n\n"
    prompt += "[USER_TASK]\n" + user_task.strip() + "\n\n"
    prompt += "[OUTPUT_SCHEMA]\n" + json.dumps(schema, ensure_ascii=False, indent=2) + "\n\n"
    prompt += "Return JSON only."
    return prompt


def generate_task_plan_from_user(user_task, task_key):
    llm_text = call_llm(build_plan_prompt(user_task)).strip()
    parsed = parse_json_safely(llm_text)
    plan = parsed.get("task_plan")
    if not isinstance(plan, dict):
        raise RuntimeError("LLM did not return task_plan object")
    plan.setdefault("meta", {})
    plan["meta"].setdefault("created", datetime.now().strftime("%Y-%m-%d"))
    plan["meta"]["task_key"] = task_key
    save_task_plan_dict(plan)
    return plan

# =============================
# 11) Round prompt (adds task_plan_updates)
# =============================


def build_round_prompt(user_message, state_snapshot, last_tool_result, runtime_state, round_id):
    policy_lines = [
        "Planning-with-Files rules:",
        "1) Create Plan First: task_plan.md must exist and guide work.",
        "2) The 2-Action Rule: after every 2 view/browser/search operations, save key findings to files.",
        "3) Read Before Decide: before major decisions, read the plan (in snapshot).",
        "4) Update After Act: update progress.md and task_plan.md state.",
        "5) Log ALL Errors: any error must be recorded.",
        "Output requirements:",
        "- Output JSON only.",
        "- writeback MUST be present every turn.",
        "- At most ONE tool_call per turn.",
        f"Round {round_id}/{MAX_ROUNDS}."
    ]

    schema = {
        "assistant_answer": "string",
        "done": "boolean",
        "done_reason": "string (optional)",
        "tool_call": {
            "tool": "shell|python|search_web|search_database",
            "command": "string (if tool=shell)",
            "code": "string (if tool=python)",
            "query": "string (if tool=search_web or search_database)",
            "topk": "integer (if tool=search_web or search_database)",
            "timeout_sec": "integer (optional)",
        },
        "writeback": {
            "findings_append": ["lines"],
            "progress_append": ["lines"],
            "task_plan_updates": {
                "phase_status": [{"phase_id": "Phase 1", "status": "in_progress"}],
                "checklist_set": [{"phase_id": "Phase 1", "item_text": "...", "done": True}],
            },
        },
    }

    prompt = "[POLICY]\n" + "\n".join(policy_lines) + "\n\n"
    prompt += "[STATE_SNAPSHOT]\n" + json.dumps(state_snapshot, ensure_ascii=False, indent=2) + "\n\n"
    prompt += "[LAST_TOOL_RESULT]\n" + json.dumps(last_tool_result, ensure_ascii=False, indent=2) + "\n\n"
    prompt += "[RUNTIME_HINTS]\n" + json.dumps(runtime_state, ensure_ascii=False, indent=2) + "\n\n"
    prompt += "[USER]\n" + user_message.strip() + "\n\n"
    prompt += "[OUTPUT_SCHEMA]\n" + json.dumps(schema, ensure_ascii=False, indent=2) + "\n\n"
    prompt += "Return JSON only."
    return prompt

# =============================
# 12) Tool output summary + writeback (progress/findings unchanged)
# =============================


def summarize_tool_output(tool_result):
    if tool_result is None:
        return None
    stdout = (tool_result.get("stdout") or "").strip()
    stderr = (tool_result.get("stderr") or "").strip()
    summary = ""
    if stdout:
        summary += stdout[:MAX_TOOL_OUTPUT_CHARS]
    if stderr:
        if summary:
            summary += "\n--- STDERR ---\n"
        summary += stderr[:MAX_TOOL_OUTPUT_CHARS]
    return summary


def apply_writeback(writeback, state):
    # KEEP findings/progress behavior UNCHANGED
    findings_lines = writeback.get("findings_append") or []
    if findings_lines:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        append_lines(FINDINGS_FILE, [f"\n### Auto-Note {stamp}"] + findings_lines)
        state["actions_since_last_findings"] = 0

    progress_lines = writeback.get("progress_append") or []
    if progress_lines:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        append_lines(PROGRESS_FILE, [f"\n### Auto-Log {stamp}"] + progress_lines)

# =============================
# 13) Main loop
# =============================


def run_task(user_message):
    ensure_findings_and_progress_exist()

    runtime_state = load_state()
    runtime_state["root_user_message"] = user_message
    task_key = runtime_state.get("task_key") or compute_task_key(user_message)
    runtime_state["task_key"] = task_key
    save_state(runtime_state)

    # 1) Ensure task_plan dict exists in memory
    task_plan_dict = ensure_task_plan_dict(task_key)

    # 2) If first-time plan not created, ask LLM to create dict
    if (not bool(runtime_state.get("plan_created", False))) or (load_task_plan_dict() is None):
        task_plan_dict = generate_task_plan_from_user(user_message, task_key)
        runtime_state["plan_created"] = True
        save_state(runtime_state)

    container = get_container()
    final_answer = ""

    for round_id in range(1, MAX_ROUNDS + 1):
        runtime_state = load_state()

        findings_text = read_text(FINDINGS_FILE)
        progress_text = read_text(PROGRESS_FILE)

        snapshot = extract_state_snapshot(task_plan_dict, findings_text, progress_text)
        last_tool = runtime_state.get("last_tool_result")

        current_user = user_message if round_id == 1 else "continue"
        parsed = parse_json_safely(call_llm(build_round_prompt(current_user, snapshot, last_tool, runtime_state, round_id)).strip())

        if parsed.get("assistant_answer") is not None:
            final_answer = str(parsed.get("assistant_answer", final_answer))

        done = bool(parsed.get("done", False))
        tool_call = parsed.get("tool_call") or {}
        writeback = parsed.get("writeback") or {}

        writeback.setdefault("progress_append", [])
        writeback.setdefault("findings_append", [])
        writeback.setdefault("task_plan_updates", {})

        # Apply task_plan updates directly to dict (NEW)
        tp_errs = apply_task_plan_updates(task_plan_dict, writeback.get("task_plan_updates") or {})
        if tp_errs:
            for msg in tp_errs[:10]:
                writeback["progress_append"].append(f"- [ERROR] task_plan_update: {msg}")

        tool_result = None

        if tool_call.get("tool") == "search_web":
            query = tool_call.get("query", "")
            topk = int(tool_call.get("topk", 5) or 5)
            timeout_sec = int(tool_call.get("timeout_sec", DEFAULT_TIMEOUT_SEC) or DEFAULT_TIMEOUT_SEC)
            tool_result = container.exec_search_web(query, topk=topk, timeout_sec=timeout_sec)
            ppp(tool_result)
            runtime_state["actions_since_last_findings"] = int(runtime_state.get("actions_since_last_findings", 0)) + 1
            runtime_state["last_tool_result"] = {"tool": "search_web", "exit_code": tool_result.get("exit_code"), "summary": summarize_tool_output(tool_result)}
            if not writeback["progress_append"]:
                writeback["progress_append"] = [f"- Ran search_web: query='{query}' topk={topk} (exit={tool_result.get('exit_code')})"]

        if tool_call.get("tool") == "search_database":
            query = tool_call.get("query", "")
            topk = int(tool_call.get("topk", 5) or 5)
            timeout_sec = int(tool_call.get("timeout_sec", DEFAULT_TIMEOUT_SEC) or DEFAULT_TIMEOUT_SEC)
            tool_result = container.exec_search_database(query, topk=topk, timeout_sec=timeout_sec)
            runtime_state["actions_since_last_findings"] = int(runtime_state.get("actions_since_last_findings", 0)) + 1
            runtime_state["last_tool_result"] = {"tool": "search_database", "exit_code": tool_result.get("exit_code"), "summary": summarize_tool_output(tool_result)}
            if not writeback["progress_append"]:
                writeback["progress_append"] = [f"- Ran search_database: query='{query}' topk={topk} (exit={tool_result.get('exit_code')})"]

        if tool_call.get("tool") == "shell" and tool_call.get("command"):
            cmd = tool_call["command"]
            timeout_sec = int(tool_call.get("timeout_sec", DEFAULT_TIMEOUT_SEC) or DEFAULT_TIMEOUT_SEC)
            tool_result = container.exec_shell(cmd, timeout_sec=timeout_sec)
            runtime_state["actions_since_last_findings"] = int(runtime_state.get("actions_since_last_findings", 0)) + 1
            runtime_state["last_tool_result"] = {"tool": "shell", "command": cmd, "exit_code": tool_result.get("exit_code"), "summary": summarize_tool_output(tool_result)}
            if not writeback["progress_append"]:
                writeback["progress_append"] = [f"- Ran shell: {cmd} (exit={tool_result.get('exit_code')})"]

        if tool_call.get("tool") == "python" and tool_call.get("code"):
            code_str = tool_call["code"]
            timeout_sec = int(tool_call.get("timeout_sec", DEFAULT_TIMEOUT_SEC) or DEFAULT_TIMEOUT_SEC)
            tool_result = container.exec_python(code_str, timeout_sec=timeout_sec)
            runtime_state["actions_since_last_findings"] = int(runtime_state.get("actions_since_last_findings", 0)) + 1
            runtime_state["last_tool_result"] = {"tool": "python", "exit_code": tool_result.get("exit_code"), "summary": summarize_tool_output(tool_result)}
            if not writeback["progress_append"]:
                writeback["progress_append"] = [f"- Ran python code (exit={tool_result.get('exit_code')})"]

        # 2-action fallback (unchanged)
        if int(runtime_state.get("actions_since_last_findings", 0)) >= 2 and not writeback.get("findings_append"):
            ltr = runtime_state.get("last_tool_result") or {}
            summary = (ltr.get("summary") or "").strip()
            if summary:
                writeback["findings_append"] = [f"- Auto-saved (2-action rule) summary: {summary[:200]}"]

        apply_writeback(writeback, runtime_state)
        save_state(runtime_state)

        # End of round: dump whole dict to task_plan.md (NEW)
        save_task_plan_dict(task_plan_dict)

        print(f"[Round {round_id}/{MAX_ROUNDS}] done={done} tool={tool_call.get('tool','')}")
        if done:
            print(final_answer)
            return final_answer

    print(f"MAX_ROUNDS {MAX_ROUNDS} encountered, please check if task is too complicated.")
    return final_answer


def main():
    #user_message = input("User> ")
    current_key = compute_task_key(user_message)
    st = load_state()
    if st.get("task_key") != current_key:
        reset_all_files_for_new_task(user_message)
    run_task(user_message)


if __name__ == "__main__":
    main()
