from __future__ import annotations

import math

from sbmath.tree import Node, Wildcard, Value, Function, PythonFunction, Variable, NodeFunction, AddAndSub
from sbmath.parser import parse

from sbmath._utils import debug
from sbmath.tree.context.std import std

_sum_pat: Node = parse("[u]+[v]")
_prod_pat: Node = parse("[u]*[v]")
_div_pat: Node = parse("[u]/[v]")
_func_pat: Node = parse("[func]([arg])")

_prod_sum_pat = parse("[k]*([a]+[b])")
_sum_prod_pat = parse("[k]*[a]+[k]*[b]")
_binom_pat = parse("([a]+[b])^[n]")

_common_factor_pat = parse("[k]*[a, eval=1]+[k]*[b, eval=1]+[c]")
_no_common_factor_pat = parse("[k]*([a]+[b])+[c]")

_common_factor2_pat = parse("[k]*[a, eval=1]+[k]*[b, eval=1]")
_no_common_factor2_pat = parse("[k]*([a]+[b])")

_common_factor1_pat = parse("[k]*[a, eval=1]+[k]")
_no_common_factor1_pat = parse("[k]*([a]+1)")



_derivatives: dict[Function, Function] = {
    std.functions["abs"]: Function.from_expression(parse("abs(x)/x"), parse("x")),
    std.functions["sqrt"]: Function.from_expression(parse("1 / ( 2*sqrt(x) )"), parse("x")),
    std.functions["exp"]: std.functions["exp"],
    std.functions["ln"]: Function.from_expression(parse("1/x"), parse("x")),
    std.functions["log"]: Function.from_expression(parse("1/(ln(10)*x)"), parse("x")),
}


def _diff_no_reduce(expression: Node, variable: Variable) -> Node:
    m = Wildcard("_", constant_with=variable).matches(expression)
    if m:
        result = Value(0.0)
        result.context = expression.context

        return result

    m = variable.matches(expression)
    if m:
        result = Value(1.0)
        result.context = expression.context

        return result

    m = _sum_pat.matches(expression)
    if m:
        result = _diff_no_reduce(m.wildcards["u"], variable) + _diff_no_reduce(m.wildcards["v"], variable)
        result.context = expression.context

        return result

    m = _prod_pat.matches(expression)
    if m and not m.weak:
        u = m.wildcards["u"]
        v = m.wildcards["v"]
        # product rule
        result = _diff_no_reduce(u, variable) * v + u * _diff_no_reduce(v, variable)
        result.context = expression.context

        return result

    m = _div_pat.matches(expression)
    if m:
        u = m.wildcards["u"]
        v = m.wildcards["v"]
        # quotient rule
        result = (_diff_no_reduce(u, variable) * v - u * _diff_no_reduce(v, variable)) / (v ** 2)
        result.context = expression.context

        return result

    # redundant with next match but faster in most cases
    m = (Wildcard("u") ** Wildcard("k", constant_with=variable)).matches(expression)
    if m:
        u = m.wildcards["u"]
        k = m.wildcards["k"]
        result = (k * _diff_no_reduce(u, variable) * (u ** (k - 1)))
        result.context = expression.context

        return result

    # m = (Wildcard("u") ** Wildcard("k", constant_with=variable)).matches(expression)
    # if m:
    #     u = m.wildcards["u"]
    #     k = m.wildcards["k"]
    #     return (k * (u ** (k - 1))).reduce()

    m = (Wildcard("u") ** Wildcard("v")).matches(expression)
    if m:
        u = m.wildcards["u"]
        v = m.wildcards["v"]
        # u^v = exp( v*ln(u) ) = exp(w) => ( u^v )' = ( exp(w) )'
        w = v * std.functions["ln"](u)
        result = _diff_no_reduce(w, variable) * std.functions["exp"](w)
        result.context = expression.context

        return result

    m = _func_pat.matches(expression)
    if m:
        func = m.functions_wildcards["func"]
        arg = m.wildcards["arg"]
        if isinstance(func, NodeFunction):
            result = _diff_no_reduce(func(arg).reduce(depth=1), variable)
            result.context = expression.context
            return result
        if func in _derivatives.keys():
            result = (_diff_no_reduce(arg, variable) * _derivatives[func](arg).reduce(depth=1))
            result.context = expression.context
            return result

    # TODO: functions with multiple parameters
    # TODO: function operating on raw nodes in expression tree
    result = Wildcard("diff", expr=expression, var=variable)
    result.context = expression.context

    return result


def diff(expression: Node, variable: Variable) -> Node:
    return _diff_no_reduce(expression, variable).reduce()


def expand(expression: Node) -> Node:
    m = _binom_pat.matches(expression)
    if m is not None and isinstance(m.wildcards["n"], Value) and m.wildcards["n"].data.is_integer():
        n = int(m.wildcards["n"].data)
        a = m.wildcards["a"]
        b = m.wildcards["b"]
        # binomial formula
        expression = AddAndSub.add(*(math.comb(n, k) * a ** (n-k) * b ** k for k in range(n+1))).reduce()

    # the pattern matching is powerful enough to support more complex expansions
    return expression.replace(_prod_sum_pat, _sum_prod_pat)


def simplify(expression: Node):
    expression = expression.reduce()
    expression = expression.replace(_common_factor_pat, _no_common_factor_pat)
    expression = expression.replace(_common_factor2_pat, _no_common_factor2_pat)
    expression = expression.replace(_common_factor1_pat, _no_common_factor1_pat)

    return expression.reduce()
