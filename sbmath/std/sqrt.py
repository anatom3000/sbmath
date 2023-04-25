from __future__ import annotations

import math

from sbmath.tree import PythonFunction, Node, Value
from sbmath.computation import integer


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
                    outside = 1
                    inside = 1
                    factorized = integer.prime_factorize(argument.data)
                    for p, k in factorized.items():
                        outside *= p**(k//2)
                        if p % 2 != 0:
                            inside *= p
                    if inside == 1:
                        return Node.from_float(outside)
                    elif outside == 1:
                        return self(Value.from_float(inside))
                    else:
                        return Value.from_float(outside) * self(Value.from_float(inside))

        return None

    def __init__(self, parse):
        super().__init__(math.sqrt, {
            parse("[arg]^2"): parse("[arg]")
        })
