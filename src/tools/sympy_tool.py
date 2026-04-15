"""SymPy tool for symbolic mathematics."""

from typing import Dict, Any
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr


class SympyTool:
    """Symbolic math using SymPy."""
    
    def execute(self, expression: str = "", operation: str = "solve", **kwargs) -> Dict[str, Any]:
        """
        Execute symbolic math operation.
        
        Args:
            expression: Mathematical expression
            operation: Operation type (solve, simplify, expand, factor, etc.)
            **kwargs: Additional arguments
            
        Returns:
            Result dict
        """
        try:
            if operation == "solve":
                return self._solve(expression, kwargs.get("variable", "x"))
            elif operation == "simplify":
                return self._simplify(expression)
            elif operation == "expand":
                return self._expand(expression)
            elif operation == "factor":
                return self._factor(expression)
            elif operation == "check_equality":
                return self._check_equality(expression, kwargs.get("target", "0"))
            else:
                return {
                    "success": False,
                    "data": {},
                    "error": f"Unknown operation: {operation}"
                }
        except Exception as e:
            return {
                "success": False,
                "data": {},
                "error": f"{type(e).__name__}: {str(e)}"
            }
    
    def _solve(self, expression: str, variable: str) -> Dict[str, Any]:
        """Solve equation for variable."""
        expr = parse_expr(expression)
        var = sp.Symbol(variable)
        solutions = sp.solve(expr, var)
        
        return {
            "success": True,
            "data": {
                "result": [str(sol) for sol in solutions],
                "solutions": solutions,
                "type": "solutions"
            },
            "error": ""
        }
    
    def _simplify(self, expression: str) -> Dict[str, Any]:
        """Simplify expression."""
        expr = parse_expr(expression)
        simplified = sp.simplify(expr)
        
        return {
            "success": True,
            "data": {
                "result": str(simplified),
                "simplified": simplified,
                "type": "simplified"
            },
            "error": ""
        }
    
    def _expand(self, expression: str) -> Dict[str, Any]:
        """Expand expression."""
        expr = parse_expr(expression)
        expanded = sp.expand(expr)
        
        return {
            "success": True,
            "data": {
                "result": str(expanded),
                "expanded": expanded,
                "type": "expanded"
            },
            "error": ""
        }
    
    def _factor(self, expression: str) -> Dict[str, Any]:
        """Factor expression."""
        expr = parse_expr(expression)
        factored = sp.factor(expr)
        
        return {
            "success": True,
            "data": {
                "result": str(factored),
                "factored": factored,
                "type": "factored"
            },
            "error": ""
        }
    
    def _check_equality(self, expr1: str, expr2: str) -> Dict[str, Any]:
        """Check if two expressions are equal."""
        e1 = parse_expr(expr1)
        e2 = parse_expr(expr2)
        
        diff = sp.simplify(e1 - e2)
        equal = diff == 0
        
        return {
            "success": True,
            "data": {
                "result": equal,
                "equal": equal,
                "difference": str(diff),
                "type": "equality_check"
            },
            "error": ""
        }
