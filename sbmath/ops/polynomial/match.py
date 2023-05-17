from __future__ import annotations

from sbmath.ops.simplify import simplify
from sbmath.tree import Node, Wildcard, Value

from sbmath._utils import debug
from sbmath.parser import parse

from sbmath.ops.polynomial.core import Polynomial

_add_pat = parse("[a]+[b]")


def _match_all(patterns: Iterable[Node], value: Node):
    m = None
    for pattern in patterns:
        m = pattern.matches(value)
        if m is None:
            return None

    return m


# note: returns (variable_index, variable_exponent)
def _match_partial_monomial(monomial: Node, variables: list[Node]) -> Optional[tuple[int, int]]:
    exponent = Wildcard("p", integer_value=Value(1))
    for index, var in enumerate(variables):
        m = (var ** exponent).matches(monomial)
        if m:
            return index, m.wildcards['p'].data

    for index, var in enumerate(variables):
        m = var.matches(monomial)
        if m:
            return index, 1

    return None


def _match_monomial(monomial: Node, variables: list[Node]) -> Optional[Polynomial]:
    for i in range:
        pass


# TODO: adapt _match_polynomial to multi-variate polynomials
def _match_polynomial(polynomial: Node, variables: list[Node]) -> Optional[Polynomial]:
    m = _add_pat.matches(polynomial)
    if m:
        return _match_polynomial(m.wildcards["a"], variables) + _match_polynomial(m.wildcards["b"], variables)

    term = Wildcard("k", constant_with=variable) * variable ** Wildcard("n", eval=Value(1))
    m = term.matches(polynomial)
    if m:
        n = m.wildcards["n"].approximate()
        if not n.is_integer():
            return None

        n = int(n)

        return n * [None] + [m.wildcards["k"]]

    term = Wildcard("k", constant_with=variable) * variable
    m = term.matches(polynomial)
    if m:
        return [None, m.wildcards["k"]]

    term = variable
    m = term.matches(polynomial)
    if m:
        return [None, Value(1)]

    if m:
        return Polynomial.constant(k, variables)

    return None


def match_polynomial(polynomial: Node, variables: list[Node]) -> Optional[list[Node]]:
    raw_coefficients = _match_polynomial(simplify(polynomial), variables)
    if raw_coefficients is None:
        return None

    return [Value(0) if c is None else simplify(c) for c in raw_coefficients]
