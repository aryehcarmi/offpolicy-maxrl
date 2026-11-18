"""Binary Countdown reward for controlled experiments."""
import ast
import operator
import re
from collections import Counter
from fractions import Fraction

_OPERATIONS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def verify(text: str, numbers: list[int], target: int) -> bool:
    """Require a single final answer, integer leaves and exact rational equality."""
    if text.count("<answer>") != 1 or text.count("</answer>") != 1:
        return False
    match = re.search(r"<answer>([^<>]*)</answer>\s*\Z", text)
    if match is None:
        return False
    expr = match[1].strip()
    if len(expr) > 256 or not re.fullmatch(r"[0-9+*/()\s-]+", expr):
        return False
    # Bound work before parsing and reject different spellings of literal integers.
    tokens = re.findall(r"[0-9]+", expr)
    if Counter(tokens) != Counter(map(str, numbers)):
        return False

    def value(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return Fraction(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATIONS:
            return _OPERATIONS[type(node.op)](value(node.left), value(node.right))
        raise ValueError("only binary arithmetic and integer operands are allowed")

    try:
        return value(ast.parse(expr, mode="eval").body) == target
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError):
        return False
