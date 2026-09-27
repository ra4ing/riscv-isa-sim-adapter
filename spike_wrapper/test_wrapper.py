#!/usr/bin/env python3
"""Exercise argument integrity, query timeout, and executor-worker death."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import spike_wrapper


def process_state(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return None


def wait_for_pid(path, worker=None):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().strip():
            return int(path.read_text())
        if worker is not None and worker.poll() is not None:
            raise AssertionError(f"worker exited early: {worker.returncode}")
        time.sleep(0.01)
    raise AssertionError("fake Spike was not started")


def wait_for_exit(pid):
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        if process_state(pid) in (None, "Z", "X"):
            return
        time.sleep(0.01)
    raise AssertionError(f"Spike process {pid} survived worker/query termination")


with tempfile.TemporaryDirectory(prefix="spike-wrapper-test-") as tmp:
    root = Path(tmp)
    stub = root / "spike"
    stub.write_text(
        '#!/bin/sh\n'
        'printf "%s\\n" "$$" > "$SPIKE_TEST_PID"\n'
        'if [ "$SPIKE_TEST_MODE" = hang ]; then exec sleep 117; fi\n'
        'printf "%s\\n" "$@"\n'
    )
    stub.chmod(0o755)
    old_path = os.environ["PATH"]
    old_pid = os.environ.get("SPIKE_TEST_PID")
    old_mode = os.environ.get("SPIKE_TEST_MODE")
    os.environ["PATH"] = f"{root}:{old_path}"
    os.environ["SPIKE_TEST_PID"] = str(root / "spike.pid")
    try:
        os.environ["SPIKE_TEST_MODE"] = "normal"
        debug = "until pc 0x123; literal spaces"
        result = spike_wrapper.debug_cmd_str_elf_file("/tmp/fake.elf", debug, "rv64g")
        assert result == f"-d\n--isa=rv64g\n--debug-cmd-from-string={debug}\n/tmp/fake.elf\n", result
        result = spike_wrapper.debug_cmd_file_elf_file("/tmp/fake.elf", "file with spaces", "rv64g")
        assert result == "-d\n--isa=rv64g\n--debug-cmd=file with spaces\n/tmp/fake.elf\n", result

        os.environ["SPIKE_TEST_MODE"] = "hang"
        pid_file = root / "spike.pid"
        pid_file.unlink()
        start = time.monotonic()
        try:
            spike_wrapper.debug_cmd_str_elf_file("/tmp/fake.elf", "until", "rv64g")
            raise AssertionError("hung query returned without a timeout")
        except RuntimeError as exc:
            assert "spike query timed out" in str(exc), str(exc)
        assert 1.5 <= time.monotonic() - start < 4
        wait_for_exit(wait_for_pid(pid_file))

        pid_file.unlink()
        worker = subprocess.Popen(
            [sys.executable, "-c", "import spike_wrapper; "
             "spike_wrapper.debug_cmd_str_elf_file('/tmp/fake.elf', 'until', 'rv64g')"],
            env=os.environ.copy(), cwd=Path(__file__).parent,
        )
        child = None
        try:
            child = wait_for_pid(pid_file, worker)
            assert process_state(child) not in (None, "Z", "X")
            os.kill(worker.pid, signal.SIGKILL)
            worker.wait(timeout=2)
            wait_for_exit(child)
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.wait()
            if child is not None and process_state(child) not in (None, "Z", "X"):
                os.kill(child, signal.SIGKILL)
    finally:
        os.environ["PATH"] = old_path
        if old_pid is None:
            os.environ.pop("SPIKE_TEST_PID", None)
        else:
            os.environ["SPIKE_TEST_PID"] = old_pid
        if old_mode is None:
            os.environ.pop("SPIKE_TEST_MODE", None)
        else:
            os.environ["SPIKE_TEST_MODE"] = old_mode
print("Spike argv, query timeout, and worker-death cleanup: OK")
