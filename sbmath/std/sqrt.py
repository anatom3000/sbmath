from __future__ import annotations

import math

from sbmath.tree import PythonFunction, Node, Value, Wildcard
from sbmath.computation import integer

_factor_pat = Wildcard("a") * Wildcard("b")
_ratio_pat = Wildcard("a") / Wildcard("b")


class FunctionSqrt(PythonFunction):
    def reduce_func(self, argument: Node, depth: int, evaluate: bool) -> Optional[Node]:
        result = super().reduce_func(argument, depth, evaluate)

        if result is not None:
            return result

        if evaluate and argument.is_evaluable():
            argument = argument.evaluate()

        if isinstance(argument, Value):
            root = math.sqrt(argument.data)
            if root.is_integer():
                return Node.from_float(root)
            else:
                outside, inside = integer.decompose_sqrt(argument.data)
                if inside == 1:
                    return Node.from_float(outside)
                elif outside == 1:
                    return self(Value.from_float(inside))
                else:
                    return Value.from_float(outside) * self(Value.from_float(inside))

        m = _factor_pat.matches(argument)
        if m and not m.weak:
            sqrt_a = self(m.wildcards['a']).reduce(depth=depth-1, evaluate=evaluate)
            sqrt_b = self(m.wildcards['b']).reduce(depth=depth-1, evaluate=evaluate)
            return sqrt_a * sqrt_b

        m = _ratio_pat.matches(argument)
        if m:
            sqrt_a = self(m.wildcards['a']).reduce(depth=depth-1, evaluate=evaluate)
            sqrt_b = self(m.wildcards['b']).reduce(depth=depth-1, evaluate=evaluate)
            return sqrt_a / sqrt_b

        return None

    def __init__(self, parse):
        super().__init__(math.sqrt, {
            parse("[arg]^2"): parse("[arg]")
        })
