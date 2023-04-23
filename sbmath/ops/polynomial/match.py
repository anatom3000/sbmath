from __future__ import annotations

from sbmath.ops.simplify import simplify
from sbmath.tree import Node, Wildcard, Value

from sbmath._utils import debug
from sbmath.parser import parse

_add_pat = parse("[a]+[b]")


def _add_polynomials(p: Optional[list[Optional[Node]]], q: Optional[list[Optional[Node]]]):
    if p is None or q is None:
        return None

    smallest_length = min(len(p), len(q))

    common_coefficients = []
    for a, b in zip(p, q):
        if a is None:
            if b is None:
                common_coefficients.append(None)
            else:
                common_coefficients.append(b)
        else:
            if b is None:
                common_coefficients.append(a)
            else:
                common_coefficients.append(a+b)

    return common_coefficients + p[smallest_length:] + q[smallest_length:]


def _match_polynomial(polynomial: Node, variable: Node) -> Optional[list[Optional[Node]]]:
    m = _add_pat.matches(polynomial)
    if m:
        return _add_polynomials(_match_polynomial(m.wildcards["a"], variable), _match_polynomial(m.wildcards["b"], variable))

    term = Wildcard("k", constant_with=variable) * variable ** Wildcard("n", eval=Value(1))
    m = term.matches(polynomial)
    if m:
        n = m.wildcards["n"].approximate()
        if not n.is_integer():
            return None

        n = int(n)

        return n*[None] + [m.wildcards["k"]]

    term = variable ** Wildcard("n", eval=Value(1))
    m = term.matches(polynomial)
    if m:
        n = m.wildcards["n"].approximate()
        if not n.is_integer():
            return None

        n = int(n)

        return n*[None] + [Value(1.0)]

    term = Wildcard("k", constant_with=variable) * variable
    m = term.matches(polynomial)
    if m:
        return [None, m.wildcards["k"]]

    term = variable
    m = term.matches(polynomial)
    if m:
        return [None, Value(1.0)]

    term = Wildcard("c", constant_with=variable)
    m = term.matches(polynomial)
    if m:
        return [m.wildcards["c"]]

    return None


def match_polynomial(polynomial: Node, variable: Node) -> Optional[list[Node]]:
    raw_coefficients = _match_polynomial(simplify(polynomial), variable)
    if raw_coefficients is None:
        return None

    return [Value(0.0) if c is None else simplify(c) for c in raw_coefficients]
