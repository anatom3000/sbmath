from sbmath import parse
from sbmath.tree import Node

_common_factor_pat = parse("[k]*[a, eval=1]+[k]*[b, eval=1]+[c]")
_no_common_factor_pat = parse("[k]*([a]+[b])+[c]")

_common_factor2_pat = parse("[k]*[a, eval=1]+[k]*[b, eval=1]")
_no_common_factor2_pat = parse("[k]*([a]+[b])")

_common_factor1_pat = parse("[k]*[a, eval=1]+[k]")
_no_common_factor1_pat = parse("[k]*([a]+1)")


def simplify(expression: Node):
    expression = expression.reduce()
    expression = expression.replace(_common_factor_pat, _no_common_factor_pat)
    expression = expression.replace(_common_factor2_pat, _no_common_factor2_pat)
    expression = expression.replace(_common_factor1_pat, _no_common_factor1_pat)

    return expression.reduce()
