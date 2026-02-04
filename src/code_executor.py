import subprocess
import sys
import tempfile
import os
from typing import Dict, Any, Optional


class CodeExecutor:
    def execute_code(self, code: str, language: str = "python", timeout: int = 30) -> Dict[str, Any]:
        if language == "python":
            return self._execute_python(code, timeout)
        elif language == "javascript":
            return self._execute_javascript(code, timeout)
        elif language == "bash":
            return self._execute_bash(code, timeout)
        else:
            return {"status": "error", "message": f"Unsupported language: {language}"}

    def _execute_python(self, code: str, timeout: int) -> Dict[str, Any]:
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()
            )
            return {
                "status": "success" if result.returncode == 0 else "error",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "message": "Execution timed out"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _execute_javascript(self, code: str, timeout: int) -> Dict[str, Any]:
        try:
            result = subprocess.run(
                ["node", "-e", code],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return {
                "status": "success" if result.returncode == 0 else "error",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except FileNotFoundError:
            return {"status": "error", "message": "Node.js not found"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _execute_bash(self, code: str, timeout: int) -> Dict[str, Any]:
        try:
            result = subprocess.run(
                code,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()
            )
            return {
                "status": "success" if result.returncode == 0 else "error",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "message": "Execution timed out"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def execute_file(self, filepath: str, language: str = "python", timeout: int = 30) -> Dict[str, Any]:
        if not os.path.exists(filepath):
            return {"status": "error", "message": "File not found"}
        
        with open(filepath, 'r') as f:
            code = f.read()
        
        return self.execute_code(code, language, timeout)
