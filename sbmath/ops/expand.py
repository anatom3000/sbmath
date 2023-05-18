from __future__ import annotations

import math

from sbmath import parse
from sbmath.expression import Expression, AddAndSub, Value, MatchResult

_prod_sum_pat = parse("[k]*([a]+[b])")
_sum_prod_pat = parse("[k]*[a]+[k]*[b]")
_binom_pat = parse("([a]+[b]) ^ [n, integer_value: 1]")


def _binomial_expansion(m: MatchResult) -> Expression:
    a = m.wildcards["a"]
    b = m.wildcards["b"]

    n = m.wildcards["n"].data

    return AddAndSub.add(*(math.comb(n, k) * a ** (n - k) * b ** k for k in range(n + 1))).reduce()


def expand(expression: Expression) -> Expression:
    expression = expression.apply_on(_binom_pat, _binomial_expansion)

    # the pattern matching is powerful enough to support more complex expansions
    return expression.replace(_prod_sum_pat, _sum_prod_pat)

__all__ = ['expand']
