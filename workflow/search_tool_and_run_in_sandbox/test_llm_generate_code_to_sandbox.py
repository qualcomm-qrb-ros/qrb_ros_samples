
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# VIBE CODING WARNING:
# This is a minimal Python sandbox prototype for controlled/educational use.
# It is NOT a complete security boundary. Do not run untrusted code in production.

import os
import sys
import time
import tempfile
import subprocess
import yaml

# POSIX-only resource limits (Linux/Unix). Not available on Windows.
try:
    import resource
except ImportError:
    resource = None


def _limit_resources(memory_mb: int, cpu_seconds: int, nofile: int = 64, fsize_mb: int = 16, nproc: int = 64):
    """
    Apply OS-level resource caps in the child process (POSIX).
    Intended for use via subprocess preexec_fn.
    """
    if resource is None:
        return  # No resource module (likely Windows)

    # Cap virtual memory (approximate RAM usage ceiling)
    resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024, memory_mb * 1024 * 1024))
    # Cap CPU time (seconds)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    # Cap maximum file size (limits disk writes)
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_mb * 1024 * 1024, fsize_mb * 1024 * 1024))
    # Cap number of open file descriptors
    resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))
    # Cap number of child processes (mitigates fork bombs)
    resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))

    # Put process in its own group for easier termination
    try:
        os.setpgrp()
    except Exception:
        pass


def run_user_code(
    code: str,
    input_data: str = "",
    timeout_sec: int = 2,
    memory_mb: int = 128,
    cpu_seconds: int = 2,
):
    """
    Execute user Python code in a restricted subprocess.
    Captures stdout/stderr and enforces time/memory limits.

    Returns:
        dict: {
            stdout (str),
            stderr (str),
            returncode (int),
            duration (float),
            timed_out (bool)
        }
    """
    start = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        user_code_path = os.path.join(tmpdir, "user_code.py")

        # Write user code to a temp file
        with open(user_code_path, "w", encoding="utf-8") as f:
            f.write(code)

        # Run Python in a more isolated mode:
        # -I: isolated mode (ignores user site-packages and env vars)
        # -S: do not import site automatically
        # -u: unbuffered I/O for reliable output capture
        cmd = [sys.executable, "-I", "-S", "-u", user_code_path]

        try:
            result = subprocess.run(
                cmd,
                input=input_data,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=tmpdir,          # confined working dir
                env={},              # empty environment for fewer side effects
                start_new_session=True,  # separate session for termination control
                # preexec_fn=lambda: _limit_resources(memory_mb, cpu_seconds),
            )
            # duration = time.time() - start
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                # "duration": duration,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as e:
            # On timeout, report and mark as timed_out
            # duration = time.time() - start
            return {
                "stdout": e.stdout or "",
                "stderr": (e.stderr or "") + "\n[timeout] Process exceeded time limit.",
                "returncode": -1,
                # "duration": duration,
                "timed_out": True,
            }


if __name__ == "__main__":
    question = "create 10 files named from test_text01 to test_text10"
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
        print("no code needed")
        exit()
    print(act_dict["code_block"])
    
    demo_code = act_dict["code_block"]

    tool_res = run_user_code(demo_code, input_data="", timeout_sec=1, memory_mb=64, cpu_seconds=1)
    print("=== stdout ===")
    print(tool_res["stdout"])
    print("=== stderr ===")
    print(tool_res["stderr"])
    print("=== returncode ===")
    print(tool_res["returncode"])
    
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