"""Production-grade calculator utility for LangChain tools."""

from __future__ import annotations

import ast
import operator

Number = int | float

_ALLOWED_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorError(ValueError):
    """Raised when the input expression is invalid or unsafe."""


def calculate_expression(expression: str) -> str:
    """Safely evaluate a simple arithmetic expression and return string output."""
    sanitized = expression.strip()
    if not sanitized:
        raise CalculatorError("Expression must not be empty.")

    try:
        node = ast.parse(sanitized, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError("Invalid arithmetic expression.") from exc

    result = _evaluate_ast(node.body)
    return str(result)


def _evaluate_ast(node: ast.AST) -> Number:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINARY_OPERATORS:
            raise CalculatorError(f"Unsupported operator: {op_type.__name__}")

        left = _evaluate_ast(node.left)
        right = _evaluate_ast(node.right)
        try:
            return _ALLOWED_BINARY_OPERATORS[op_type](left, right)
        except ZeroDivisionError as exc:
            raise CalculatorError("Division by zero is not allowed.") from exc

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARY_OPERATORS:
            raise CalculatorError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _evaluate_ast(node.operand)
        return _ALLOWED_UNARY_OPERATORS[op_type](operand)

    raise CalculatorError("Only numeric arithmetic expressions are allowed.")
