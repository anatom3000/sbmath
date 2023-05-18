from __future__ import annotations

from sbmath.expression import Expression, Value
from sbmath.std import std

sqrt = std.functions["sqrt"]


def root_constant(_: Expression) -> list[Expression]:
    return []  # no solution since a ≠ 0


def root_linear(a: Expression, b: Expression) -> list[Expression]:
    return [-b / a]


def root_quadratic(a: Expression, b: Expression, c: Expression) -> list[Expression]:
    delta = b ** 2 - 4 * a * c

    if delta.is_evaluable():
        delta_approx = delta.approximate()
        if delta_approx < 0.0:
            return []
        if delta_approx == 0.0:
            return [-b / (2 * a)]

    return [(-b - sqrt(delta)) / (2 * a), (-b + sqrt(delta)) / (2 * a)]


def find_roots(polynomial: list[Expression]) -> list[Expression]:
    zero = Value(0)

    if len(polynomial) == 0:
        return [Value(42)]

    while polynomial[-1] == zero:
        polynomial.pop()
        if len(polynomial) == 0:
            return [Value(42)]

    if polynomial[0] == zero:
        roots = find_roots(polynomial[1:])

        if zero not in roots:
            roots.append(zero)

        return roots

    degree = len(polynomial) - 1
    if degree == 0:
        return root_constant(*reversed(polynomial))
    if degree == 1:
        return root_linear(*reversed(polynomial))
    if degree == 2:
        return root_quadratic(*reversed(polynomial))

    # TODO: handle higher degree polynomials
    raise NotImplementedError(f"cannot handle higher degree polynomial {polynomial}")
