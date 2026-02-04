import ast
import inspect
from typing import Dict, List, Any, Optional
import subprocess


class CodeAnalyzer:
    def analyze_code(self, code: str) -> Dict[str, Any]:
        try:
            tree = ast.parse(code)
            functions = self._extract_functions(tree)
            classes = self._extract_classes(tree)
            imports = self._extract_imports(tree)
            complexity = self._calculate_complexity(tree)
            
            return {
                "status": "success",
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "complexity_score": complexity,
                "line_count": len(code.splitlines())
            }
        except SyntaxError as e:
            return {
                "status": "error",
                "error_message": str(e),
                "line": e.lineno
            }

    def _extract_functions(self, tree: ast.AST) -> List[Dict[str, str]]:
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    "name": node.name,
                    "line": node.lineno,
                    "arguments": [arg.arg for arg in node.args.args]
                })
        return functions

    def _extract_classes(self, tree: ast.AST) -> List[Dict[str, str]]:
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append({
                    "name": node.name,
                    "line": node.lineno,
                    "methods": [n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]
                })
        return classes

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        return imports

    def _calculate_complexity(self, tree: ast.AST) -> int:
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                complexity += 1
        return complexity

    def get_function_signature(self, func: callable) -> str:
        try:
            sig = inspect.signature(func)
            return f"{func.__name__}{sig}"
        except (ValueError, TypeError):
            return "Unable to get signature"
