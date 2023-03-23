from __future__ import annotations

from sbmath.tree import Node, Wildcard, Value, Function, PythonFunction, Variable
from sbmath.parser import parse

from sbmath._utils import debug

_sum_pat: Node = parse("[u]+[v]")
_prod_pat: Node = parse("[u]*[v]")
_div_pat: Node = parse("[u]/[v]")


def diff(expression: Node | Function, variable: Variable) -> Node:
    if isinstance(expression, Function):
        if isinstance(expression, NodeExpression):
            return diff(expression(variable), variable).reduce()
        if isinstance(expression, PythonFunction):
            raise NotImplementedError("todo: derivative of functions (chain rule)")

    m = Wildcard("_", constant_with=variable).matches(expression)
    if m:
        return Value(0.0)

    m = variable.matches(expression)
    if m:
        return Value(1.0)

    m = _sum_pat.matches(expression)
    if m:
        return (diff(m.wildcards["u"], variable) + diff(m.wildcards["v"], variable)).reduce()

    m = _prod_pat.matches(expression)
    if m and not m.weak:
        u = m.wildcards["u"]
        v = m.wildcards["v"]
        # product rule
        return (diff(u, variable) * v + u * diff(v, variable)).reduce()

    m = _div_pat.matches(expression)
    if m:
        u = m.wildcards["u"]
        v = m.wildcards["v"]
        # quotient rule
        return ((diff(u, variable) * v - u * diff(v, variable)) / (v ** 2)).reduce()

    m = (Wildcard("u") ** Wildcard("k", constant_with=variable)).matches(expression)
    if m:
        u = m.wildcards["u"]
        k = m.wildcards["k"]
        return (k * (u ** (k - 1))).reduce()

    # TODO: functions with multiple parameters
    # TODO: function operating on raw nodes in expression tree
    return Wildcard("diff", expr=expression, var=variable)
