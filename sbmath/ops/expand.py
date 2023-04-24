from __future__ import annotations

from sbmath import parse
from sbmath.tree import Node, AddAndSub

_prod_sum_pat = parse("[k]*([a]+[b])")
_sum_prod_pat = parse("[k]*[a]+[k]*[b]")
_binom_pat = parse("([a]+[b])^[n]")


def expand(expression: Node) -> Node:
    m = _binom_pat.matches(expression)
    if m is not None:
        n = m.wildcards["n"].reduce()
        if isinstance(n, Value):
            # binomial formula
            n = int(m.wildcards["n"].data)
            a = m.wildcards["a"]
            b = m.wildcards["b"]
        expression = AddAndSub.add(*(math.comb(n, k) * a ** (n - k) * b ** k for k in range(n + 1))).reduce()

    # the pattern matching is powerful enough to support more complex expansions
    return expression.replace(_prod_sum_pat, _sum_prod_pat)
