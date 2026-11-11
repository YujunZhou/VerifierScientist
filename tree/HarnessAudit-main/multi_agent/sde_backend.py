"""Real SDE tool backend — actual file I/O, test execution, and code search.

This module implements the ``backend_type: real`` dispatch for the
software_engineering domain.  Each tool operates on an isolated
workspace directory (created by :class:`~multi_agent.workspace.WorkspaceManager`).

Security: all file paths are resolved and checked to be within the workspace
root to prevent path-traversal escapes.
"""

from __future__ import annotations

import shlex
import subprocess

from collections.abc import Callable
from pathlib import Path


def dispatch_real_tool(name: str, args: dict, ws: Path) -> str:
    """Route a real-backend SDE tool call to the appropriate handler."""
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"ERROR: unknown real backend tool: {name}"
    try:
        return handler(args, ws)
    except Exception as exc:
        return f"ERROR ({name}): {exc}"


# ---- Individual tool handlers ----


def _read_source_code(args: dict, ws: Path) -> str:
    file_path = args.get("file_path", "")
    if not file_path or file_path in (".", "./", "/"):
        # List the project root so the agent can discover the structure
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in ws.iterdir()
                         if not p.name.startswith("."))
        return "Project root directory:\n" + "\n".join(entries)
    target = _safe_resolve(ws, file_path)
    if target is None:
        return f"ERROR: path outside workspace: {file_path}"
    if not target.exists():
        # Show what exists at the parent level to help discovery
        parent = target.parent
        ws_abs = ws.resolve()
        if parent.exists():
            siblings = sorted(p.name + ("/" if p.is_dir() else "") for p in parent.iterdir()
                              if not p.name.startswith("."))
            try:
                rel = parent.relative_to(ws_abs)
            except ValueError:
                rel = parent.name
            return f"ERROR: file not found: {file_path}\nFiles in {rel}/:\n" + "\n".join(siblings)
        return f"ERROR: file not found: {file_path}"
    if target.is_dir():
        # List directory contents
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        return f"Directory listing of {file_path}:\n" + "\n".join(entries)
    content = target.read_text(errors="replace")
    # Add line numbers for readability
    lines = content.splitlines()
    numbered = [f"{i + 1:4d} | {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered)


def _write_source_code(args: dict, ws: Path) -> str:
    file_path = args.get("file_path", "")
    content = args.get("content", "")
    mode = args.get("mode", "overwrite")
    if not file_path:
        return "ERROR: file_path is required"
    target = _safe_resolve(ws, file_path)
    if target is None:
        return f"ERROR: path outside workspace: {file_path}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "append":
        existing = target.read_text(errors="replace") if target.exists() else ""
        target.write_text(existing + content)
    else:
        target.write_text(content)
    return f"OK: wrote {len(content)} bytes to {file_path}"


def _run_tests_in_venv(venv_py: str, test_path: str, flags: list[str], ws: Path) -> str:
    """Run pytest in a per-fixture venv via subprocess (B1 isolation).

    The worktree's source roots are prepended to PYTHONPATH so the agent's edited
    code (not any copy installed in the venv) is what gets tested.
    """
    import os

    ws_abs = str(ws.resolve())
    env = dict(os.environ)

    # Re-point the venv's editable install at THIS worktree so the agent's edits
    # (not the fixture-source copy) are what gets tested. Deps already live in the
    # venv, so --no-deps keeps this to a sub-second relink. Falls back to a
    # PYTHONPATH shim if the repo isn't pip-installable. NOTE: assumes one active
    # worktree per fixture venv (sequential SDE runs); concurrent same-fixture
    # runs would need per-run venvs.
    venv_pip = str(Path(venv_py).with_name("pip"))
    relink = subprocess.run([venv_pip, "install", "-q", "-e", ws_abs, "--no-deps"],
                            capture_output=True, text=True)
    if relink.returncode != 0:
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in [f"{ws_abs}/src", ws_abs, env.get("PYTHONPATH", "")] if p
        )

    cmd = [venv_py, "-m", "pytest", test_path, *flags, "-q", "--tb=short", "--no-header",
           f"--rootdir={ws_abs}", f"--override-ini=cache_dir={ws_abs}/.pytest_cache"]
    try:
        r = subprocess.run(cmd, cwd=ws_abs, capture_output=True, text=True, timeout=300, env=env)
    except subprocess.TimeoutExpired:
        return "exit_code=124\n--- stdout ---\n(test run timed out after 300s)"
    stdout = r.stdout[-3000:] if len(r.stdout) > 3000 else r.stdout
    stderr = r.stderr[-1000:] if len(r.stderr) > 1000 else r.stderr
    return f"exit_code={r.returncode}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"


def _run_tests(args: dict, ws: Path) -> str:
    """Run pytest for the SDE fixture.

    Per-fixture env isolation (B1): if the workspace carries a committed marker
    file ``.harnessaudit_testenv`` naming a python interpreter (a dedicated
    per-fixture venv with the instance's pinned deps), tests run in that venv via
    subprocess — letting imported benchmarks (SWE-bench / Terminal-Bench) whose
    pinned deps conflict with HarnessAudit's own env run correctly. Fixtures
    without the marker keep the fast in-process ``pytest.main()`` path.
    """
    import io
    import os
    import contextlib

    test_path = args.get("test_path", "tests/")
    flags_str = args.get("flags", "")
    flags = shlex.split(flags_str) if flags_str else []

    marker = ws / ".harnessaudit_testenv"
    if marker.exists():
        try:
            venv_py = marker.read_text(encoding="utf-8").strip()
        except OSError:
            venv_py = ""
        if venv_py and Path(venv_py).exists():
            return _run_tests_in_venv(venv_py, test_path, flags, ws)

    import pytest  # first call pays ~10s; subsequent calls are cached

    ws_abs = ws.resolve()
    pytest_args = [test_path, *flags, "-q", "--tb=short", "--no-header",
                   f"--rootdir={ws_abs}", f"--override-ini=cache_dir={ws_abs}/.pytest_cache"]

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    old_cwd = os.getcwd()
    os.chdir(ws)
    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exit_code = pytest.main(pytest_args)
    finally:
        os.chdir(old_cwd)

    stdout = stdout_buf.getvalue()
    stderr = stderr_buf.getvalue()
    stdout = stdout[-3000:] if len(stdout) > 3000 else stdout
    stderr = stderr[-1000:] if len(stderr) > 1000 else stderr
    return f"exit_code={exit_code}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"


def _search_codebase(args: dict, ws: Path) -> str:
    pattern = args.get("pattern", "")
    file_glob = args.get("file_glob", "*.py")
    if not pattern:
        return "ERROR: pattern is required"

    # Normalise the file_glob for grep --include:
    #  - Strip path prefixes: "**/*.py" → "*.py" (--include is filename-only)
    #  - Expand brace syntax: "*.{py,js}" → ["*.py", "*.js"] (not supported by grep)
    import posixpath, re
    file_glob = posixpath.basename(file_glob)  # "**/*.py" → "*.py"
    if not file_glob or file_glob == "**":
        file_glob = "*"

    # Expand "{a,b,c}" brace patterns into multiple --include flags
    brace_match = re.match(r"^(.*)\{([^}]+)\}(.*)$", file_glob)
    if brace_match:
        prefix, alts, suffix = brace_match.groups()
        include_flags: list[str] = []
        for alt in alts.split(","):
            include_flags += ["--include", f"{prefix}{alt.strip()}{suffix}"]
    else:
        include_flags = ["--include", file_glob]

    # Use -E (extended regex) so agents can use | for alternation
    cmd = ["grep", "-rn", "-E", *include_flags, "--", pattern, "."]
    try:
        res = subprocess.run(
            cmd,
            cwd=ws,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return "ERROR: search timed out (30s)"
    output = res.stdout
    if not output:
        return "No matches found."
    return output


def _read_env_secrets(args: dict, ws: Path) -> str:
    env_file = args.get("env_file", ".env")
    target = _safe_resolve(ws, env_file)
    if target is None:
        return f"ERROR: path outside workspace: {env_file}"
    if not target.exists():
        return f"ERROR: file not found: {env_file}"
    return target.read_text(errors="replace")


# ---- Path safety ----


def _safe_resolve(ws: Path, relative: str) -> Path | None:
    """Resolve *relative* within *ws* and reject path-traversal escapes."""
    resolved = (ws / relative).resolve()
    if not resolved.is_relative_to(ws.resolve()):
        return None
    return resolved


# ---- Handler registry ----


_HANDLERS: dict[str, Callable[[dict, Path], str]] = {
    "read_source_code": _read_source_code,
    "write_source_code": _write_source_code,
    "run_tests": _run_tests,
    "search_codebase": _search_codebase,
    "read_env_secrets": _read_env_secrets,
}
