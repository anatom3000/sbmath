from __future__ import annotations

from collections.abc import Callable

from sbmath.ops.simplify import simplify
from sbmath.expression import Expression, Wildcard, Value

from sbmath._utils import debug
from sbmath.parser import parse

from sbmath.ops.polynomial.core import Polynomial

_add_pat = parse("[a]+[b]")
_mul_pat = parse("[a]*[b]")
_monomial_pat = parse("[a]^[n, integer_value: 1]")
_zero = Value(0)
_one = Value(1)


def _extract_binary_operator(value: Expression, pattern: Expression) -> list[Expression]:
    terms = []
    m = pattern.matches(value)
    while m:
        terms.append(m.wildcards['a'])
        value = m.wildcards['b']
        m = pattern.matches(value)
    terms.append(value)

    return terms


def match_polynomial_from_predicate(value: Expression, predicate: Callable[[Expression], bool]) -> Polynomial:
    terms = _extract_binary_operator(simplify(value), _add_pat)

    polynomial = Polynomial.zero([])
    variable_indices: dict[Expression, int] = {}
    for term in terms:
        coefficient: Expression = _one
        monomial: dict[Expression, int] = {}

        factors = _extract_binary_operator(term, _mul_pat)
        for factor in factors:

            m = _monomial_pat.matches(factor)
            if m and predicate(m.wildcards['a']):
                monomial[m.wildcards['a']] = m.wildcards['n'].data
            elif predicate(factor):
                monomial[factor] = 1
            else:
                coefficient *= factor

        exponents = [0, ] * len(polynomial.variables)
        for variable, exponent in monomial.items():
            if variable in polynomial.variables:
                exponents[variable_indices[variable]] = exponent
            else:
                variable_indices[variable] = len(polynomial.variables)
                polynomial = polynomial.add_variable(variable)
                exponents.append(exponent)

        exponents = tuple(exponents)

        polynomial.terms[exponents] = (polynomial.terms.get(exponents, _zero) + coefficient).reduce()

    return polynomial


def match_polynomial(value: Expression, variables: list[Expression]):
    return match_polynomial_from_predicate(value, lambda x: x in variables)

if __name__ == '__main__':
    from sbmath.expression import Variable

    x = parse('x')

    p = match_polynomial_from_predicate(parse('x**2+x*exp(x)+exp(x)*ln(x)**4+2*ln(x)'), lambda var: var.contains(x))
    print(p)

