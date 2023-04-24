from sbmath import parse
from sbmath.tree import Node

_common_factor_pat_with_extra = parse("[k]*[a, eval: 1]+[k]*[b, eval: 1]+[c]")
_no_common_factor_pat_with_extra = parse("[k]*([a]+[b])+[c]")

_common_factor_pat = parse("[k]*[a, eval: 1]+[k]*[b, eval: 1]")
_no_common_factor_pat = parse("[k]*([a]+[b])")

_implicit_common_factor_pat = parse("[k]*[a, eval: 1]+[k]")
_no_implicit_common_factor_pat = parse("[k]*([a]+1)")

_evaluable_common_factor_pat_with_extra = parse("[a]*[k, eval: 1]+[b]*[k, eval: 1]+[c]")
_no_evaluable_common_factor_pat_with_extra = parse("([a]+[b])*[k]+[c]")

_evaluable_common_factor_pat = parse("[a]*[k, eval: 1]+[b]*[k, eval: 1]")
_no_evaluable_common_factor_pat = parse("([a]+[b])*[k]")

_implicit_evaluable_common_factor_pat = parse("[a]*[k, eval: 1]+[a]")
_no_implicit_evaluable_common_factor_pat = parse("[a]*([k]+1)")


def simplify(expression: Node):
    expression = expression.reduce()

    expression = expression.replace(_common_factor_pat_with_extra, _no_common_factor_pat_with_extra)
    expression = expression.replace(_common_factor_pat, _no_common_factor_pat)
    expression = expression.replace(_implicit_common_factor_pat, _no_implicit_common_factor_pat)

    expression = expression.replace(_evaluable_common_factor_pat_with_extra, _no_evaluable_common_factor_pat_with_extra)
    expression = expression.replace(_evaluable_common_factor_pat, _no_evaluable_common_factor_pat)
    expression = expression.replace(_implicit_evaluable_common_factor_pat, _no_implicit_evaluable_common_factor_pat)

    return expression.reduce()
