"""表达式安全求值：AST 白名单校验 + 受限数学命名空间.

DSL 派生参数（``expr``）与计算节点公式共用本求值器：

- 输入表达式经 :mod:`ast` 解析并按白名单校验节点类型（算术/比较/布尔
  运算、常量、白名单函数调用），属性访问、下标、推导式、lambda、
  赋值等任意可产生副作用的构造一律拒绝；
- 求值命名空间仅含安全数学函数与调用方注入的变量，内建置空
  （``__builtins__`` 为空字典），无 import/exec 通道。
"""

from __future__ import annotations

import ast
import math
from typing import Any, Mapping

from .errors import ParamError

__all__ = ["SAFE_MATH_NAMESPACE", "expr_names", "safe_eval"]


def expr_names(expr: str) -> set[str]:
    """提取表达式中全部标识符（Name 节点），供依赖分析使用."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return set()
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


#: 安全数学函数/常量命名空间（标量公式够用；数组运算由 P3 计算节点扩展）
SAFE_MATH_NAMESPACE: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sqrt": math.sqrt,
    "cbrt": math.cbrt if hasattr(math, "cbrt") else None,
    "exp": math.exp,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "floor": math.floor,
    "ceil": math.ceil,
    "hypot": math.hypot,
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}
SAFE_MATH_NAMESPACE = {k: v for k, v in SAFE_MATH_NAMESPACE.items() if v is not None}

#: 白名单 AST 节点（表达式外层）
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Call,
    ast.Name,
    ast.Constant,
)
#: 白名单双目运算符
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
#: 白名单单目运算符
_ALLOWED_UNARY = (ast.UAdd, ast.USub, ast.Not, ast.Invert)
#: 白名单比较运算符
_ALLOWED_CMPOPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
#: 白名单布尔运算符
_ALLOWED_BOOLOPS = (ast.And, ast.Or)


def safe_eval(expr: str, namespace: Mapping[str, Any] | None = None) -> Any:
    """安全求值表达式；语法错误/非法构造/未知名称抛 :class:`ParamError`.

    :param expr: 表达式文本（如 ``"height ** 3 / 12"``）。
    :param namespace: 变量绑定（参数名 -> 值），叠加安全数学命名空间。
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ParamError(f"表达式语法错误: {expr!r} ({exc.msg})") from exc
    _validate_tree(tree, expr)
    merged: dict[str, Any] = {**SAFE_MATH_NAMESPACE, **dict(namespace or {})}
    try:
        return eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, merged)
    except ZeroDivisionError as exc:
        raise ParamError(f"表达式除零: {expr!r}") from exc
    except NameError as exc:
        raise ParamError(f"表达式引用未定义名称: {expr!r} ({exc})") from exc
    except (TypeError, ValueError) as exc:
        raise ParamError(f"表达式求值失败: {expr!r} ({exc})") from exc


def _validate_tree(tree: ast.AST, expr: str) -> None:
    """递归校验 AST 节点均在白名单内（walk 会产出运算符/上下文叶子，一并校验）."""
    # 运算符与访问上下文叶子节点（BinOp.op / UnaryOp.op / Compare.ops / BoolOp.op / Name.ctx）
    leaf_ok = (*_ALLOWED_BINOPS, *_ALLOWED_UNARY, *_ALLOWED_CMPOPS, *_ALLOWED_BOOLOPS, ast.Load)
    for node in ast.walk(tree):
        if isinstance(node, leaf_ok):
            continue
        if isinstance(node, _ALLOWED_NODES):
            if isinstance(node, ast.BinOp) and not isinstance(node.op, _ALLOWED_BINOPS):
                raise ParamError(f"表达式含不支持的运算符: {expr!r}")
            if isinstance(node, ast.UnaryOp) and not isinstance(node.op, _ALLOWED_UNARY):
                raise ParamError(f"表达式含不支持的运算符: {expr!r}")
            if isinstance(node, ast.Compare) and not all(isinstance(op, _ALLOWED_CMPOPS) for op in node.ops):
                raise ParamError(f"表达式含不支持的比较运算: {expr!r}")
            if isinstance(node, ast.BoolOp) and not isinstance(node.op, _ALLOWED_BOOLOPS):
                raise ParamError(f"表达式含不支持的布尔运算: {expr!r}")
            continue
        raise ParamError(f"表达式含非法构造 {type(node).__name__}: {expr!r}")
