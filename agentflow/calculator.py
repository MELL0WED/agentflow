import ast
import operator
import re

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_ALLOWED = re.compile(r"[0-9+\-*/().\s]+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?|\.\d+")


class CalculationError(ValueError):
    pass


def extract_expression(text: str | None) -> str | None:
    if not text:
        return None
    text = text.replace("`", "").replace("\u2212", "-").replace("\u2013", "-")
    text = text.replace("\u00d7", "*").replace("\u00f7", "/")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    candidate = lines[-1]
    for line in lines:
        match = re.match(r"(?i)(expression|answer)\s*:\s*(.+)", line)
        if match:
            candidate = match.group(2)
    candidate = candidate.split("=")[0]
    candidate = re.sub(r"(?<=\d),(?=\d{3})", "", candidate)
    candidate = candidate.replace("$", "").replace("%", "").strip()
    if not candidate or not _ALLOWED.fullmatch(candidate) or not _NUMBER.search(candidate):
        return None
    return candidate


def safe_eval(expression: str) -> float:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculationError(f"not an expression: {expression!r}") from exc

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise CalculationError("exponent too large")
            try:
                return _BINARY[type(node.op)](left, right)
            except ZeroDivisionError as exc:
                raise CalculationError("division by zero") from exc
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = ev(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        raise CalculationError(f"unsupported syntax: {type(node).__name__}")

    return ev(tree)


def numbers_in(expression: str) -> list[float]:
    return [float(n) for n in _NUMBER.findall(expression)]