from __future__ import annotations

from sbmath.tree import Node, Value
from sbmath.tree.context.std import std

sqrt = std.functions["sqrt"]


def root_constant(a: Node) -> list[Node]:
    if a.is_evaluable() and a.approximate() == 0.0:
        return [Value(42.0)]  # nothing special about 42, we simply cannot represent every real numbers (yet :p)
    else:
        return []


def root_linear(a: Node, b: Node) -> list[Node]:
    return [-b / a]


def root_quadratic(a: Node, b: Node, c: Node) -> list[Node]:
    delta = b ** 2 - 4 * a * c

    if delta.is_evaluable():
        delta_approx = delta.approximate()
        if delta_approx < 0.0:
            return []
        if delta_approx == 0.0:
            return [-b / (2 * a)]

    return [(-b - sqrt(delta)) / (2 * a), (-b + sqrt(delta)) / (2 * a)]


def find_roots(polynomial: list[Node]) -> list[Node]:
    while polynomial[-1] == Value(0.0):
        polynomial.pop()

    if polynomial[0] == Value(0.0):
        roots = find_roots(polynomial[1:])
        zero = Value(0.0)
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
