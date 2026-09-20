import asyncio
import subprocess
import sys
import tempfile
import time
from pathlib import Path


async def run_python_in_sandbox(code: str, timeout_sec: int = 3) -> dict:
    """在临时目录中执行一段python代码 并返回结构化的结果"""

    def _run() -> dict:
        start = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "main.py"
            script_path.write_text(code, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                    env={"PYTHONIOENCODING": "utf-8"},
                )
                duration_ms = int((time.perf_counter() - start) * 1000)
                stderr = proc.stderr.strip()
                if proc.returncode == 0:
                    error_type = None
                    error_message = ""
                elif "SyntaxError" in stderr:
                    error_type = "syntax_error"
                    error_message = "Python 代码存在语法错误。"
                else:
                    error_type = "runtime_error"
                    error_message = "Python 代码运行时抛出了异常。"
                return {
                    "ok": proc.returncode == 0,
                    "exit_code": proc.returncode,
                    "status": "success",
                    "duration_ms": duration_ms,
                    "error_type": error_type,
                    "error_message": error_message,
                    "stdout": proc.stdout.strip(),
                    "stderr": stderr,
                    "sandbox": {
                        "cwd": tmpdir,
                        "timeout_sec": timeout_sec,
                    },
                }
            except subprocess.TimeoutExpired:
                duration_ms = int((time.perf_counter() - start) * 1000)
                return {
                    "ok": False,
                    "exit_code": None,
                    "stdout": "",
                    "stderr": f"Execution timed out after {timeout_sec}s",
                    "error_type": "timeout",
                    "error_message": f"Python 代码执行超时，超过 {timeout_sec} 秒仍未结束。",
                    "duration_ms": duration_ms,
                    "sandbox": {
                        "cwd": tmpdir,
                        "timeout_sec": timeout_sec,
                    },
                }

    return await asyncio.to_thread(_run)
