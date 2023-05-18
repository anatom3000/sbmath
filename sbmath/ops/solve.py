from __future__ import annotations

from sbmath.ops.polynomial import match_polynomial_from_predicate, find_roots
from sbmath.ops.simplify import simplify
from sbmath.expression import Equality, FunctionWildcard, Wildcard, Value, Expression
from sbmath.std import std

inverse_functions = {
    std.functions["exp"]: std.functions["ln"],
    std.functions["ln"]: std.functions["exp"],
}


def to_autonomous(equation: Equality) -> Expression:
    return equation.left - equation.right


def _solve_no_reduce(equation: Equality, unknown: Variable) -> list[Expression]:
    equation = simplify(equation)  # should have unfolded any ExpressionFunction in the equation

    m = Equality(unknown, Wildcard("x", constant_with=unknown)).matches(equation)
    if m:
        return [m.wildcards["x"]]

    m = Equality(FunctionWildcard("f", Wildcard("u", variable_with=unknown)),
                 Wildcard("v", constant_with=unknown)).matches(equation)
    if m:
        f, u, v = m.functions_wildcards["f"], m.wildcards["u"], m.wildcards["v"]
        if f not in inverse_functions.keys():
            return []  # TODO: specify when a solution exists but could not be found

        inv_f = inverse_functions[f]

        return solve(Equality(u, inv_f(v)), unknown)

    autonomous = simplify(to_autonomous(equation))

    m = (Wildcard("u", variable_with=unknown) * Wildcard("v", variable_with=unknown)).matches(autonomous)
    if m:
        u, v = m.wildcards["u"], m.wildcards["v"]
        return solve(Equality(u, Value(0)), unknown) + solve(Equality(v, Value(0)), unknown)

    polynomial = match_polynomial_from_predicate(autonomous, unknown)
    if polynomial is not None:
        return find_roots(polynomial)

    raise NotImplementedError(f"could not solve the equation. {equation = }, ")


def solve(equation: Equality, unknown: Variable) -> Sequence[Expression]:
    return [simplify(solution) for solution in _solve_no_reduce(equation, unknown)]


__all__ = ['solve']