"""Subprocess wrapper for the local `claude` CLI as an LLM provider."""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass


_RATE_LIMIT_RE = re.compile(r"rate.?limit|429|503|connection|timed out", re.IGNORECASE)
_AUTH_RE = re.compile(r"not.?logged.?in|unauthor|auth(?:enticate)?|invalid.?key", re.IGNORECASE)


_PROCESS_BUCKET_LOCK = threading.Lock()
_PROCESS_BUCKET: threading.BoundedSemaphore | None = None
_PROCESS_BUCKET_SIZE: int | None = None


def _get_process_bucket(size: int) -> threading.BoundedSemaphore | None:
    """Return a process-global semaphore sized to ``size`` (the largest
    requested cap wins). Used to honor a single shared rate-limit budget
    across plan / code LLMClients that live in different event loops."""
    global _PROCESS_BUCKET, _PROCESS_BUCKET_SIZE
    if size <= 0:
        return None
    with _PROCESS_BUCKET_LOCK:
        if _PROCESS_BUCKET is None or _PROCESS_BUCKET_SIZE != size:
            _PROCESS_BUCKET = threading.BoundedSemaphore(size)
            _PROCESS_BUCKET_SIZE = size
        return _PROCESS_BUCKET


class ClaudeCodeError(RuntimeError):
    pass


class ClaudeCodeAuthError(ClaudeCodeError):
    pass


class ClaudeCodeRateLimitError(ClaudeCodeError):
    pass


@dataclass
class ClaudeCodeResult:
    content: str
    stderr: str
    duration_s: float


class ClaudeCodeBackend:
    """Drives the local `claude --print` CLI as if it were an LLM provider.

    Concurrent invocations are bounded by an internal semaphore (per-instance,
    not per-process), so multiple `LLMClient` instances each get their own
    bucket. The bucket size defaults to 8 but is overridable via
    ``AUTOCOMP_CLAUDE_MAX_CONCURRENT``.
    """

    def __init__(
        self,
        model: str,
        claude_path: str | None = None,
        timeout: int | None = None,
        claude_config_dir: str | None = None,
        max_concurrent: int | None = None,
        max_retries: int = 3,
        skip_permissions: bool | None = None,
    ):
        self.model = model

        env_path = os.environ.get("CLAUDE_BINARY")
        resolved = claude_path or env_path or shutil.which("claude")
        if not resolved:
            raise ClaudeCodeError(
                "claude binary not found on PATH; set CLAUDE_BINARY or install Claude Code"
            )
        self.claude_path = resolved

        env_timeout = os.environ.get("AUTOCOMP_CLAUDE_TIMEOUT")
        self.timeout = timeout if timeout is not None else (
            int(env_timeout) if env_timeout else 600
        )

        env_dir = os.environ.get("CLAUDE_CONFIG_DIR")
        self.claude_config_dir = claude_config_dir or env_dir

        env_concurrent = os.environ.get("AUTOCOMP_CLAUDE_MAX_CONCURRENT")
        self.max_concurrent = max_concurrent if max_concurrent is not None else (
            int(env_concurrent) if env_concurrent else 8
        )

        env_skip = os.environ.get("AUTOCOMP_CLAUDE_SKIP_PERMISSIONS")
        if skip_permissions is None:
            self.skip_permissions = bool(env_skip and env_skip not in ("0", "", "false", "False"))
        else:
            self.skip_permissions = skip_permissions

        self.max_retries = max_retries
        self._semaphore: asyncio.Semaphore | None = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore

    @staticmethod
    def _format_payload(messages: list[dict]) -> tuple[str | None, str]:
        """Split messages into (system_prompt, stdin_payload).

        System messages are concatenated into the --system-prompt arg.
        Non-system messages are turned into a tagged transcript fed to stdin.
        """
        system_parts: list[str] = []
        body_parts: list[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content")
            if not content:
                continue
            if isinstance(content, list):
                pieces: list[str] = []
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text")
                        if text:
                            pieces.append(text)
                content = "\n".join(pieces)
            if role == "system":
                system_parts.append(content)
            elif role == "assistant":
                body_parts.append(f"### assistant\n{content}")
            elif role == "tool":
                body_parts.append(f"### tool\n{content}")
            else:
                body_parts.append(f"### user\n{content}")
        system_prompt = "\n\n".join(p for p in system_parts if p) or None
        payload = "\n\n".join(body_parts).strip()
        if not payload:
            payload = "(no user input)"
        return system_prompt, payload

    def _build_cmd(self, system_prompt: str | None) -> list[str]:
        cmd = [self.claude_path, "--print", "--model", self.model]
        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])
        if self.skip_permissions:
            cmd.append("--dangerously-skip-permissions")
        return cmd

    def _build_env(self) -> dict:
        env = dict(os.environ)
        if self.claude_config_dir:
            env["CLAUDE_CONFIG_DIR"] = os.path.expanduser(self.claude_config_dir)
        return env

    async def complete(self, messages: list[dict]) -> ClaudeCodeResult:
        sem = self._get_semaphore()
        process_bucket = _get_process_bucket(self.max_concurrent)
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                async with sem:
                    if process_bucket is not None:
                        await asyncio.to_thread(process_bucket.acquire)
                    try:
                        return await self._invoke(messages)
                    finally:
                        if process_bucket is not None:
                            process_bucket.release()
            except ClaudeCodeAuthError:
                raise
            except ClaudeCodeRateLimitError as e:
                last_exc = e
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(2 ** attempt)
            except ClaudeCodeError as e:
                last_exc = e
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(2 ** attempt)
        assert last_exc is not None
        raise last_exc

    async def _invoke(self, messages: list[dict]) -> ClaudeCodeResult:
        system_prompt, payload = self._format_payload(messages)
        cmd = self._build_cmd(system_prompt)
        env = self._build_env()

        t0 = time.perf_counter()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(payload.encode("utf-8")),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            try:
                await proc.wait()
            except Exception:
                pass
            raise ClaudeCodeRateLimitError(
                f"claude --print timed out after {self.timeout}s"
            )

        duration = time.perf_counter() - t0
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            if _AUTH_RE.search(stderr) and not _RATE_LIMIT_RE.search(stderr):
                raise ClaudeCodeAuthError(
                    f"claude auth failure (exit={proc.returncode}): {stderr.strip()[:400]}"
                )
            if _RATE_LIMIT_RE.search(stderr):
                raise ClaudeCodeRateLimitError(
                    f"claude transient error (exit={proc.returncode}): {stderr.strip()[:400]}"
                )
            raise ClaudeCodeError(
                f"claude exited {proc.returncode}: {stderr.strip()[:800]}"
            )

        if not stdout.strip():
            raise ClaudeCodeRateLimitError(
                f"claude produced empty output; stderr={stderr.strip()[:400]}"
            )

        return ClaudeCodeResult(content=stdout, stderr=stderr, duration_s=duration)
