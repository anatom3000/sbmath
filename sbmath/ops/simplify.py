from sbmath import parse
from sbmath.tree import Node

_common_factor_pat = parse("[k]*[a, eval: 1]+[k]*[b, eval: 1]")
_no_common_factor_pat = parse("[k]*([a]+[b])")

_implicit_common_factor_pat = parse("[k]*[a, eval: 1]+[k]")
_no_implicit_common_factor_pat = parse("[k]*([a]+1)")

_evaluable_common_factor_pat = parse("[a]*[k, eval: 1]+[b]*[k, eval: 1]")
_no_evaluable_common_factor_pat = parse("([a]+[b])*[k]")

_implicit_evaluable_common_factor_pat = parse("[a]*[k, eval: 1]+[a]")
_no_implicit_evaluable_common_factor_pat = parse("[a]*([k]+1)")

_common_power_pat = parse("[a]^[n, eval: 1]*[b]^[n, eval: 1]")
_no_common_power_pat = parse("([a]*[b])^[n]")

_mergeable_powers_pat = parse("([a]^[b])^[c]")
_merged_powers_pat = parse("[a]^([b]*[c])")

_common_power_base_pat = parse("[a]^[x, eval: 1]*[a]^[y, eval: 1]")
_no_common_power_base_pat = parse("[a]^([x, eval: 1]+[y, eval: 1])")


def simplify(expression: Node):
    # NOTE: use Node.reduce if you need fast simplification
    #       running simplify 10x takes 1400 ms whereas running Node.reduce 1000x takes less than 300 ms
    #       pattern matching is very expensive
    expression = expression.reduce()

    expression = expression.replace(_common_factor_pat, _no_common_factor_pat)
    expression = expression.replace(_implicit_common_factor_pat, _no_implicit_common_factor_pat)

    expression = expression.replace(_evaluable_common_factor_pat, _no_evaluable_common_factor_pat)
    expression = expression.replace(_implicit_evaluable_common_factor_pat, _no_implicit_evaluable_common_factor_pat)

    expression = expression.replace(_common_power_pat, _no_common_power_pat)

    expression = expression.replace(_mergeable_powers_pat, _merged_powers_pat)

    expression = expression.replace(_common_power_base_pat, _no_common_power_base_pat)

    return expression.reduce()

if __name__ == '__main__':
    from time import perf_counter

    expr = parse('x^2*x^4')

    now = perf_counter()
    for _ in range(1000):
        simplify(expr)
    print(round((perf_counter()-now)*1000), 'ms')
