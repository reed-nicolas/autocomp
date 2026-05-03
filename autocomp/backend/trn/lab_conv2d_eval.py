"""Eval path for the lab6 nki_conv2d problem (prob_id=99 under trn-advanced-nki1).

The candidate's full ``conv2d.py`` is dropped into a sandbox alongside the
lab tester and run via ``python tester.py --basic``. The lab compiler-flag
block is enforced by AST inspection before write.
"""
from __future__ import annotations

import ast
import os
import pathlib
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

from autocomp.common import logger


LAB6_DIR = pathlib.Path(os.environ.get(
    "AUTOCOMP_LAB6_DIR", str(pathlib.Path.home() / "lab6" / "nki_conv2d")
))

REQUIRED_FUNC_NAME = "conv2d_nki"
REQUIRED_PARAMS = ("X", "W", "bias")
ALLOWED_ENV_KEYS = {"NEURON_FRAMEWORK_DEBUG", "NEURON_CC_FLAGS"}
EXPECTED_DEBUG_VALUE = "1"
EXPECTED_CC_FLAG = "--disable-dge"
FORBIDDEN_IMPORTS = {
    "torch", "subprocess", "multiprocessing", "ctypes", "socket", "urllib"
}
LAB_FILES_TO_COPY = ("conv2d_ref.py", "tester.py", "utils.py")


_LATENCY_PASSED_RE = re.compile(r"Passed! Executed in ([\d.]+) μs")
_LATENCY_FAILED_RE = re.compile(r"Failed :\( Executed in ([\d.]+) μs")


def _is_os_environ_subscript(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "os"
        and node.value.attr == "environ"
    )


def _extract_subscript_key(node: ast.Subscript) -> str | None:
    sl = node.slice
    if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
        return sl.value
    return None


def _attribute_chain(node: ast.AST) -> list[str]:
    """Return the dotted-chain for an Attribute/Name node, or [] if not one."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return list(reversed(parts))
    return []


def validate_lab_constraints(source: str) -> str | None:
    """Return None on pass, or a human-readable rejection reason."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    saw_debug_assign = False
    saw_cc_flags_assign = False
    saw_target_func = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_IMPORTS:
                    return f"forbidden import: {alias.name}"
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in FORBIDDEN_IMPORTS:
                return f"forbidden import-from: {node.module}"

        if isinstance(node, ast.Call):
            chain = _attribute_chain(node.func)
            if chain[:2] == ["os", "putenv"]:
                return "os.putenv is not allowed"
            if chain[:3] == ["os", "environ", "update"]:
                return "os.environ.update is not allowed"
            if chain[:3] == ["os", "environ", "setdefault"]:
                return "os.environ.setdefault is not allowed"
            if chain[:3] == ["os", "environ", "pop"]:
                return "os.environ.pop is not allowed"

        if isinstance(node, ast.Delete):
            for tgt in node.targets:
                if _is_os_environ_subscript(tgt):
                    return "del os.environ[...] is not allowed"

    body = tree.body
    for node in body:
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for tgt in targets:
                if _is_os_environ_subscript(tgt):
                    if isinstance(node, ast.AugAssign):
                        return "augmented assignment to os.environ[...] is not allowed"
                    key = _extract_subscript_key(tgt)
                    if key not in ALLOWED_ENV_KEYS:
                        return f"os.environ[{key!r}] is not an allowed key"
                    val = node.value
                    if not (isinstance(val, ast.Constant) and isinstance(val.value, str)):
                        return f"os.environ[{key!r}] must be assigned a string literal"
                    sval = val.value
                    if key == "NEURON_FRAMEWORK_DEBUG":
                        if sval.strip() != EXPECTED_DEBUG_VALUE:
                            return f"NEURON_FRAMEWORK_DEBUG must equal '1', got {sval!r}"
                        if saw_debug_assign:
                            return "duplicate NEURON_FRAMEWORK_DEBUG assignment"
                        saw_debug_assign = True
                    elif key == "NEURON_CC_FLAGS":
                        if sval.strip() != EXPECTED_CC_FLAG:
                            return (
                                f"NEURON_CC_FLAGS must equal ' --disable-dge ' "
                                f"(after strip: '--disable-dge'), got {sval!r}"
                            )
                        if saw_cc_flags_assign:
                            return "duplicate NEURON_CC_FLAGS assignment"
                        saw_cc_flags_assign = True
                if isinstance(tgt, ast.Name) and tgt.id == "os":
                    return "rebinding the 'os' name is not allowed"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign,)):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and isinstance(node.value, ast.Attribute):
                    chain = _attribute_chain(node.value)
                    if chain == ["os", "environ"]:
                        return "aliasing os.environ to a local name is not allowed"

    if not saw_debug_assign:
        return "missing required: os.environ['NEURON_FRAMEWORK_DEBUG'] = '1'"
    if not saw_cc_flags_assign:
        return "missing required: os.environ['NEURON_CC_FLAGS'] = ' --disable-dge '"

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == REQUIRED_FUNC_NAME:
            saw_target_func = True
            params = tuple(a.arg for a in node.args.args)
            if params != REQUIRED_PARAMS:
                return (
                    f"{REQUIRED_FUNC_NAME} signature must be {REQUIRED_PARAMS}, "
                    f"got {params}"
                )
            decorators = node.decorator_list
            has_nki_jit = False
            for dec in decorators:
                chain = _attribute_chain(dec) if isinstance(dec, ast.Attribute) else (
                    _attribute_chain(dec.func) if isinstance(dec, ast.Call) else []
                )
                if isinstance(dec, ast.Name) and dec.id == "jit":
                    has_nki_jit = True
                if chain[-2:] == ["nki", "jit"] or chain[-1:] == ["jit"]:
                    has_nki_jit = True
            if not has_nki_jit:
                return f"{REQUIRED_FUNC_NAME} must be decorated with @nki.jit"
            break
    if not saw_target_func:
        return f"missing required function: {REQUIRED_FUNC_NAME}(X, W, bias)"

    return None


def _sanitized_env() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("NEURON_")}
    keep = ("NEURON_PLATFORM_TARGET_OVERRIDE",)
    for k in keep:
        if k in os.environ:
            env[k] = os.environ[k]
    return env


def _stage_candidate(work_dir: pathlib.Path, code_str: str) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "conv2d.py").write_text(code_str)
    for fname in LAB_FILES_TO_COPY:
        src = LAB6_DIR / fname
        if not src.exists():
            raise FileNotFoundError(f"lab file missing: {src}")
        shutil.copy2(src, work_dir / fname)


def _parse_tester_output(stdout: str, stderr: str) -> dict:
    if "Test failed" in stdout or "All basic correctness tests passed" not in stdout:
        return {
            "correct": False,
            "latency": None,
            "stdout": stdout,
            "stderr": stderr or "lab tester reported correctness failure",
        }

    if "Failed :(" in stdout:
        latencies = []
        for line in stdout.splitlines():
            m = _LATENCY_PASSED_RE.search(line) or _LATENCY_FAILED_RE.search(line)
            if m:
                latencies.append(float(m.group(1)))
    else:
        latencies = [float(m.group(1)) for m in _LATENCY_PASSED_RE.finditer(stdout)]

    if not latencies:
        return {
            "correct": False,
            "latency": None,
            "stdout": stdout,
            "stderr": stderr or "no per-test latencies parsed from tester output",
        }

    geomean_us = 1.0
    for v in latencies:
        geomean_us *= v
    geomean_us = geomean_us ** (1.0 / len(latencies))
    latency_ms = round(geomean_us / 1000.0, 4)

    return {
        "correct": True,
        "latency": latency_ms,
        "stdout": stdout,
        "stderr": stderr,
    }


def _run_tester(work_dir: pathlib.Path, core_id: int | None, timeout: int) -> dict:
    cmd = ["python", "tester.py", "--basic"]
    env = _sanitized_env()
    if core_id is not None:
        env["NEURON_RT_VISIBLE_CORES"] = str(core_id)
    try:
        p = subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        return {
            "correct": False,
            "latency": None,
            "stdout": e.stdout or "",
            "stderr": (e.stderr or "") + f"\nTimed out after {timeout}s",
        }

    stats = _parse_tester_output(p.stdout, p.stderr)
    if p.returncode != 0 and stats["correct"]:
        stats["correct"] = False
        stats["stderr"] = stats.get("stderr", "") + f"\ntester exit={p.returncode}"
    return stats


def _post_import_signature_check(work_dir: pathlib.Path) -> str | None:
    check_script = '''\
import sys, json, traceback, inspect, pathlib
sys.path.insert(0, ".")
try:
    import conv2d as _m
    f = getattr(_m, "conv2d_nki", None)
    if f is None:
        print(json.dumps({"ok": False, "err": "conv2d_nki missing after import"}))
        sys.exit(0)
    inner = getattr(f, "func", f)
    src = inspect.getsourcefile(inner) or ""
    if not src.endswith("conv2d.py"):
        print(json.dumps({"ok": False, "err": f"conv2d_nki defined elsewhere: {src}"}))
        sys.exit(0)
    sig = inspect.signature(inner)
    params = tuple(sig.parameters.keys())
    if params != ("X", "W", "bias"):
        print(json.dumps({"ok": False, "err": f"unexpected params: {params}"}))
        sys.exit(0)
    print(json.dumps({"ok": True}))
except Exception:
    print(json.dumps({"ok": False, "err": traceback.format_exc()[-2000:]}))
'''
    try:
        p = subprocess.run(
            ["python", "-c", check_script],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=60,
            env=_sanitized_env(),
        )
    except subprocess.TimeoutExpired:
        return "post-import shape check timed out"
    line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    if not line:
        return f"post-import shape check produced no output. stderr={p.stderr[:400]}"
    import json as _json
    try:
        data = _json.loads(line)
    except Exception:
        return f"post-import shape check unparseable: {line[:400]}"
    if not data.get("ok"):
        return f"post-import shape check failed: {data.get('err', '')[:600]}"
    return None


def evaluate_lab_conv2d(
    code_strs: list[str],
    temp_dir: pathlib.Path,
    parallelism: int = 1,
    timeout: int = 900,
) -> list[dict]:
    """Run the lab tester for each candidate. Returns one stats dict per candidate."""
    results: list[dict | None] = [None] * len(code_strs)

    to_run: list[int] = []
    for i, code in enumerate(code_strs):
        reason = validate_lab_constraints(code)
        if reason is not None:
            logger.error(f"candidate {i} rejected by constraint gate: {reason}")
            results[i] = {
                "correct": False,
                "latency": None,
                "stdout": "",
                "stderr": f"LAB_CONSTRAINT_VIOLATION: {reason}",
            }
            continue
        work_dir = temp_dir / f"cand_{i}"
        try:
            _stage_candidate(work_dir, code)
        except Exception as e:
            results[i] = {
                "correct": False,
                "latency": None,
                "stdout": "",
                "stderr": f"failed to stage candidate: {e}",
            }
            continue
        sig_err = _post_import_signature_check(work_dir)
        if sig_err is not None:
            logger.error(f"candidate {i} failed post-import check: {sig_err}")
            results[i] = {
                "correct": False,
                "latency": None,
                "stdout": "",
                "stderr": f"LAB_POSTIMPORT: {sig_err}",
            }
            continue
        to_run.append(i)

    if not to_run:
        return [r for r in results]  # type: ignore[return-value]

    parallelism = max(1, min(parallelism, len(to_run)))

    def _run(idx: int):
        work_dir = temp_dir / f"cand_{idx}"
        core = (to_run.index(idx) % parallelism) if parallelism > 1 else None
        return idx, _run_tester(work_dir, core_id=core, timeout=timeout)

    if parallelism == 1:
        for idx in to_run:
            _, stats = _run(idx)
            results[idx] = stats
    else:
        with ThreadPoolExecutor(max_workers=parallelism) as ex:
            for idx, stats in ex.map(_run, to_run):
                results[idx] = stats

    return [r for r in results]  # type: ignore[return-value]
