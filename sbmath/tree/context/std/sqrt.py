from __future__ import annotations

import math

from sbmath.tree import PythonFunction, Node, Value


class FunctionSqrt(PythonFunction):
    def reduce_func(self, argument: Node, depth: int, evaluate: bool) -> Optional[Node]:
        result = super().reduce_func(argument, depth, evaluate)

        if result is not None:
            return result

        if evaluate and argument.is_evaluable():
            argument = argument.evaluate()
            if isinstance(argument, Value) and argument.data.is_integer():
                root = math.sqrt(argument.data)
                if root.is_integer():
                    return Value(root)

        return None

    def __init__(self, parse):
        super().__init__(math.sqrt, {
            parse("[arg]^2"): parse("[arg]")
        })
