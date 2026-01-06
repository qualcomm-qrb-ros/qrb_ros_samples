
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
                preexec_fn=lambda: _limit_resources(memory_mb, cpu_seconds),
            )
            duration = time.time() - start
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "duration": duration,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as e:
            # On timeout, report and mark as timed_out
            duration = time.time() - start
            return {
                "stdout": e.stdout or "",
                "stderr": (e.stderr or "") + "\n[timeout] Process exceeded time limit.",
                "returncode": -1,
                "duration": duration,
                "timed_out": True,
            }


if __name__ == "__main__":
    # Quick smoke test: print, read input, sleep
    demo_code = r'''
import time
print("Hello from sandbox")
s = input()
print("You typed:", s)
time.sleep(0.5)
'''
    res = run_user_code(demo_code, input_data="Shenzhen\n", timeout_sec=1, memory_mb=64, cpu_seconds=1)
    print("=== stdout ===")
    print(res["stdout"])
    print("=== stderr ===")
    print(res["stderr"])
    print("=== returncode ===")
    print(res["returncode"])