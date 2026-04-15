"""Python sandbox tool for arithmetic and logic."""

import ast
import sys
import io
import signal
from contextlib import contextmanager
from typing import Dict, Any
from ..agent.params import PYTHON_TOOL_TIMEOUT


class TimeoutError(Exception):
    """Timeout exception."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Execution timeout")


class PythonTool:
    """Execute Python code in a constrained sandbox."""
    
    ALLOWED_IMPORTS = {'math', 'itertools', 'collections', 're', 'json'}
    
    def __init__(self):
        self.timeout = PYTHON_TOOL_TIMEOUT
    
    def execute(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code safely.
        
        Args:
            code: Python code to execute
            
        Returns:
            Result dict with success, data, and error
        """
        # Validate code
        if not self._validate_code(code):
            return {
                "success": False,
                "data": {},
                "error": "Code validation failed: forbidden constructs or imports"
            }
        
        # Count lines
        lines = [l for l in code.split('\n') if l.strip() and not l.strip().startswith('#')]
        if len(lines) > 50:
            return {
                "success": False,
                "data": {},
                "error": f"Code too long: {len(lines)} lines (max 50)"
            }
        
        try:
            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)
            
            # Execute in restricted namespace
            namespace = self._create_namespace()
            exec(code, namespace)
            
            # Cancel timeout
            signal.alarm(0)
            
            # Get output
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            # Extract result variable if exists
            result = namespace.get('result', output.strip())
            
            return {
                "success": True,
                "data": {
                    "result": result,
                    "output": output,
                    "type": type(result).__name__
                },
                "error": ""
            }
            
        except TimeoutError:
            signal.alarm(0)
            sys.stdout = old_stdout
            return {
                "success": False,
                "data": {},
                "error": f"Execution timeout ({self.timeout}s)"
            }
        except Exception as e:
            signal.alarm(0)
            sys.stdout = old_stdout
            return {
                "success": False,
                "data": {},
                "error": f"{type(e).__name__}: {str(e)}"
            }
    
    def _validate_code(self, code: str) -> bool:
        """Validate code for safety."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        
        # Check for forbidden constructs
        for node in ast.walk(tree):
            # No imports outside allowed list
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] not in self.ALLOWED_IMPORTS:
                        return False
            
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] not in self.ALLOWED_IMPORTS:
                    return False
            
            # No file operations, subprocess, eval, etc.
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in {'eval', 'exec', 'compile', '__import__', 'open'}:
                        return False
        
        return True
    
    def _create_namespace(self) -> Dict[str, Any]:
        """Create restricted namespace for execution."""
        import math
        import itertools
        import collections
        import re
        import json
        
        namespace = {
            'math': math,
            'itertools': itertools,
            'collections': collections,
            're': re,
            'json': json,
            '__builtins__': {
                'abs': abs,
                'all': all,
                'any': any,
                'bin': bin,
                'bool': bool,
                'chr': chr,
                'dict': dict,
                'divmod': divmod,
                'enumerate': enumerate,
                'filter': filter,
                'float': float,
                'hex': hex,
                'int': int,
                'len': len,
                'list': list,
                'map': map,
                'max': max,
                'min': min,
                'oct': oct,
                'ord': ord,
                'pow': pow,
                'print': print,
                'range': range,
                'reversed': reversed,
                'round': round,
                'set': set,
                'sorted': sorted,
                'str': str,
                'sum': sum,
                'tuple': tuple,
                'zip': zip,
            }
        }
        
        return namespace
