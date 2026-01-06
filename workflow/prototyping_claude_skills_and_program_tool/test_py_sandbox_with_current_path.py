
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# VIBE CODING WARNING:
# Minimal Python sandbox that runs code under ./_sandbox and offers two file utilities.
# POSIX resource limits are applied when available; Windows runs without them.
# This is NOT a complete security boundary. Use in controlled/educational contexts.

import os
import sys
import time
import subprocess
import yaml
from typing import Dict, Any, List, Optional

# Try POSIX resource limits; not available on Windows
try:
    import resource  # type: ignore
except Exception:
    resource = None


# -- core: ensure sandbox dir --------------------------------------------------

SANDBOX_DIRNAME = "_sandbox"

def ensure_sandbox_dir() -> str:
    """
    Ensure ./_sandbox exists and return its absolute path.
    """
    path = os.path.abspath(os.path.join(os.getcwd(), SANDBOX_DIRNAME))
    os.makedirs(path, exist_ok=True)
    return path


# -- optional: POSIX resource limits ------------------------------------------

def _limit_resources(memory_mb: int, cpu_seconds: int, nofile: int = 64, fsize_mb: int = 16, nproc: int = 64):
    """
    Apply soft caps for the child process (POSIX only).
    No-op on Windows.
    """
    if resource is None:
        return

    # Cap virtual memory (approx RAM ceiling)
    resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024, memory_mb * 1024 * 1024))
    # Cap CPU time (seconds)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    # Cap maximum file size (limits disk writes)
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_mb * 1024 * 1024, fsize_mb * 1024 * 1024))
    # Cap number of open file descriptors
    resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))
    # Cap number of child processes (mitigates fork bombs)
    resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))

    # Put child in its own process group (POSIX)
    try:
        os.setpgrp()
    except Exception:
        pass


# -- helpers: formatting -------------------------------------------------------

def _format_seconds(value: float, decimals: int = 2) -> str:
    """
    Truncate to fixed decimals (e.g., 0.0479 -> "0.04").
    """
    if value < 0 or not (value == value):  # NaN check
        value = 0.0
    factor = 10 ** decimals
    truncated = int(value * factor) / factor
    return f"{truncated:.{decimals}f}"


# -- main: run code in ./_sandbox ---------------------------------------------

def run_user_code(
    code: str,
    input_data: str = "",
    timeout_sec: int = 2,
    memory_mb: int = 128,
    cpu_seconds: int = 2,
) -> Dict[str, Any]:
    """
    Execute Python source in a subprocess under ./_sandbox.
    Captures stdout/stderr; enforces timeout. POSIX gets resource caps.

    Returns:
        dict: {stdout, stderr, returncode, duration, timed_out}
    """
    if not isinstance(code, str) or not code.strip():
        return {"stdout": "", "stderr": "empty code", "returncode": -1, "duration": "0.00", "timed_out": False}

    sandbox = ensure_sandbox_dir()
    user_code_path = os.path.join(sandbox, "user_code.py")

    # Write/overwrite the code file in ./_sandbox
    with open(user_code_path, "w", encoding="utf-8") as f:
        f.write(code)

    # Use isolated/unbuffered flags; cross-platform
    cmd = [sys.executable, "-I", "-S", "-u", user_code_path]

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            input=input_data,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=sandbox,
            # preexec_fn is POSIX-only; set None on Windows to avoid errors
            preexec_fn=(lambda: _limit_resources(memory_mb, cpu_seconds)) if resource is not None else None,
            start_new_session=True,  # new process group/session
        )
        duration = _format_seconds(time.time() - start)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "duration": duration,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as e:
        duration = _format_seconds(time.time() - start)
        return {
            "stdout": e.stdout or "",
            "stderr": (e.stderr or "") + "\n[timeout] exceeded",
            "returncode": -1,
            "duration": duration,
            "timed_out": True,
        }

# -- tiny demo ----------------------------------------------------------------

if __name__ == "__main__":  
    question = "检查当前文件目录并且只保留跟david tao相关的文件"
    tool_res = "no tool result"
    prompt = f"""

### CONTEXT
You are an assistant who can use code execution tool, target language is python.
Question: {question}
Tool Use Results: {tool_res}

### ACTION SPACE
[1] code_exec
  Description: this tool will run python code clock in sandbox and only return execution results. you need to define your required return strings for "Tool Use Results" part.
  Parameters:
    - code_block (str): the complete code block that can run in a single python file

[2] answer
  Description: Answer the question based on code execution tool
  Parameters:
    - answer (str): Final answer to the question

## NEXT ACTION
Decide the next action based on the context and available actions.
Return your response in this format:

```yaml
thinking: |
    <your step-by-step reasoning process>
action: code_exec OR answer
reason: <why you chose this action>
answer: <if action is answer>
code_block: <your_code>
```
IMPORTANT: Make sure to:
1. Use proper indentation (4 spaces) for all multi-line fields
2. Use the | character for multi-line text fields
3. Keep single-line fields without the | character

### END

"""
    response = call_llm(system="",user=prompt)
    
    # Parse the response to get the decision
    yaml_str = response.split("```yaml")[1].split("```")[0].strip()
    act_dict = yaml.safe_load(yaml_str)
    if act_dict["code_block"] == "":
        print("no code block ! exit first !!! ")
        exit()
    print(act_dict["code_block"])
    
    demo_code = act_dict["code_block"]

    tool_res = run_user_code(demo_code, input_data="", timeout_sec=1, memory_mb=64, cpu_seconds=1)
    
    print(tool_res)

    prompt = f"""

### CONTEXT
You are an assistant who can use code execution tool, target language is python.
Question: {question}
Tool Use Results: {tool_res}

### ACTION SPACE
[1] code_exec
  Description: this tool will run python code clock in sandbox and only return execution results. you need to define your required return strings for "Tool Use Results" part.
  Parameters:
    - code_block (str): the complete code block that can run in a single python file

[2] answer
  Description: Answer the question based on code execution tool
  Parameters:
    - answer (str): Final answer to the question

## NEXT ACTION
Decide the next action based on the context and available actions.
Return your response in this format:

```yaml
thinking: |
    <your step-by-step reasoning process>
action: code_exec OR answer
reason: <why you chose this action>
answer: <if action is answer>
code_block: <your_code>
```
IMPORTANT: Make sure to:
1. Use proper indentation (4 spaces) for all multi-line fields
2. Use the | character for multi-line text fields
3. Keep single-line fields without the | character

### END

"""
    response = call_llm(system="",user=prompt)
    # print(response)