from __future__ import annotations

from typing import Optional

import sbmath.ops.polynomial.core
from sbmath.expression import Expression

VariableExponents = tuple[int, ...]
Monomial = tuple[tuple[int, ...], Expression]


def exponents_mul(a: VariableExponents, b: VariableExponents) -> VariableExponents:
    return tuple(i + j for i, j in zip(a, b))


def exponents_div(a: VariableExponents, b: VariableExponents) -> Optional[VariableExponents]:
    result = []
    for i, j in zip(a, b):
        d = i - j
        if d < 0:
            return None
        result.append(d)

    return tuple(result)


def exponents_lcm(a: VariableExponents, b: VariableExponents) -> Optional[VariableExponents]:
    return tuple(max(i, j) for i, j in zip(a, b))


def exponents_lexicographic_max(a: VariableExponents, b: VariableExponents) -> VariableExponents:
    return a if a >= b else b


def term_div(a: Monomial, b: Monomial) -> Optional[Monomial]:
    divided_exponents = exponents_div(a[0], b[0])
    if divided_exponents is None:
        return None

    return divided_exponents, (a[1] / b[1]).reduce()


def polynomial_monomial_product(polynomial: sbmath.ops.polynomial.core.Polynomial,
                                exponents: VariableExponents) -> sbmath.ops.polynomial.core.Polynomial:
    from sbmath.ops.polynomial.core import Polynomial
    # print(f"{polynomial = }")
    # print(f"{exponents = }")

    return Polynomial(
        {exponents_mul(pexponents, exponents): pcoefficient for (pexponents, pcoefficient) in polynomial.terms.items()},
        polynomial.variables)


__all__ = ['exponents_mul', 'exponents_div', 'exponents_lcm', 'exponents_lexicographic_max', 'term_div', 'polynomial_monomial_product']
