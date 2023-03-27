from __future__ import annotations

from sbmath.tree import Node, Wildcard, Value, Function, PythonFunction, Variable, NodeFunction
from sbmath.parser import parse

from sbmath._utils import debug
from sbmath.tree.context.std import std

_sum_pat: Node = parse("[u]+[v]")
_prod_pat: Node = parse("[u]*[v]")
_div_pat: Node = parse("[u]/[v]")
_func_pat: Node = parse("[func]([arg])")


_derivatives: dict[Function, Function] = {
    std.functions["abs"]: Function.from_expression(parse("abs(x)/x"), parse("x")),
    std.functions["sqrt"]: Function.from_expression(parse("1 / ( 2*sqrt(x) )"), parse("x")),
    std.functions["exp"]: std.functions["exp"],
    std.functions["ln"]: Function.from_expression(parse("1/x"), parse("x")),
    std.functions["log"]: Function.from_expression(parse("1/(ln(10)*x)"), parse("x")),
}


def diff(expression: Node, variable: Variable) -> Node:
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

    # redundant with next match but faster in most cases
    m = (Wildcard("u") ** Wildcard("k", constant_with=variable)).matches(expression)
    if m:
        u = m.wildcards["u"]
        k = m.wildcards["k"]
        return (k * (u ** (k - 1))).reduce()

    # m = (Wildcard("u") ** Wildcard("k", constant_with=variable)).matches(expression)
    # if m:
    #     u = m.wildcards["u"]
    #     k = m.wildcards["k"]
    #     return (k * (u ** (k - 1))).reduce()

    m = (Wildcard("u") ** Wildcard("k")).matches(expression)
    if m:
        u = m.wildcards["u"]
        v = m.wildcards["v"]
        # u^v = exp( v*ln(u) ) = exp(w) => ( u^v )' = ( exp(w) )'
        w = v * std.functions["ln"](u)
        return (diff(w, variable) * std.functions["exp"](w)).reduce()

    m = _func_pat.matches(expression)
    if m:
        func = m.functions_wildcards["func"]
        arg = m.wildcards["arg"]
        if isinstance(func, NodeFunction):
            return diff(func(arg).reduce(), variable)
        if func in _derivatives.keys():
            return (diff(arg, variable) * _derivatives[func](arg)).reduce()

    # TODO: functions with multiple parameters
    # TODO: function operating on raw nodes in expression tree
    return Wildcard("diff", expr=expression, var=variable)


_prod_sum_pat = parse("[k]*([a]+[b])")
_sum_prod_pat = parse("[k]*[a]+[k]*[b]")


def expand(expression: Node):
    return expression.replace(_prod_sum_pat, _sum_prod_pat)