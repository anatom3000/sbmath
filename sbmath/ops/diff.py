from __future__ import annotations

from sbmath.expression import Expression, Wildcard, Function, Variable, ExpressionFunction
from sbmath.parser import parse

from sbmath.std import std

_sum_pat: Expression = parse("[u]+[v]")
_prod_pat: Expression = parse("[u]*[v]")
_div_pat: Expression = parse("[u]/[v]")
_func_pat: Expression = parse("[func]([arg])")


_x = parse("x")

_derivatives: dict[Function, Function] = {
    std.functions["abs"]: Function.from_expression(parse("abs(x)/x"), _x),
    std.functions["sqrt"]: Function.from_expression(parse("1 / ( 2*sqrt(x) )"), _x),
    std.functions["exp"]: std.functions["exp"],
    std.functions["ln"]: Function.from_expression(parse("1/x"), _x),
    std.functions["log"]: Function.from_expression(parse("1/(ln(10)*x)"), _x),
}


def _diff_no_reduce(expression: Expression, variable: Variable) -> Expression:
    m = Wildcard("_", constant_with=variable).matches(expression)
    if m:
        result = Expression.from_float(0.0)
        result.context = expression.context

        return result

    m = variable.matches(expression)
    if m:
        result = Expression.from_float(1.0)
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
        if isinstance(func, ExpressionFunction):
            result = _diff_no_reduce(func(arg).reduce(depth=1), variable)
            result.context = expression.context
            return result
        if func in _derivatives.keys():
            result = (_diff_no_reduce(arg, variable) * _derivatives[func](arg).reduce(depth=1))
            result = result.reduce(depth=1)
            result.context = expression.context
            return result

    # TODO: function with multiple parameters
    # TODO: function operating on expressions in expressions
    result = Wildcard("diff", expr=expression, var=variable)
    result.context = expression.context

    return result


def diff(expression: Expression, variable: Variable) -> Expression:
    return _diff_no_reduce(expression, variable).reduce()

__all__ = ['diff']
