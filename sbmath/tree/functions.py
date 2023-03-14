from __future__ import annotations

import copy
from abc import ABC, abstractmethod

from .base import Node, Variable, MatchResult, Value
from .._utils import debug, inc_indent, dec_indent


class FunctionNode(Node, ABC):
    name: str

    def same_function(self, other: Node) -> bool:
        return type(self) == type(other) and self.name == other.name

    def __eq__(self, other: Node) -> bool:
        return self.same_function(other) and self.argument == other.argument

    def _replace_identifiers(self, match_result: MatchResult) -> Node:
        return type(self)(self.argument._replace_identifiers(match_result))

    def replace(self, old: Node, new: Node):
        return new if old.matches(self) else type(self)(self.argument.replace(old, new))

    def contains(self, pattern: Node) -> bool:
        return pattern.matches(self) is not None or self.argument.contains(pattern)

    def _match_no_reduce(self, value: Node, state: MatchResult) -> Optional[MatchResult]:
        if not isinstance(value, type(self)):
            return None

        if not self.same_function(value):
            return None

        return self.argument.matches(value.argument, state)

    def matches(self, value: Node, state: MatchResult = None) -> Optional[MatchResult]:
        if state is None:
            state = MatchResult()

        no_reduce_state = self._match_no_reduce(value, copy.deepcopy(state))

        if no_reduce_state is not None:
            return no_reduce_state

        reduced_self = self.reduce()
        reduced_value = value.reduce()

        if not isinstance(reduced_self, BaseFunctionNode):
            return reduced_self.matches(reduced_value, state)

        return reduced_self._match_no_reduce(reduced_value, state)

    def is_evaluable(self) -> bool:
        return self.argument.is_evaluable()

    def __init__(self, argument: Node):
        self.argument = argument

    def __str__(self):
        return f"{self.name}({self.argument})"

    def __hash__(self):
        return hash(str(self))


class PythonFunctionNode(FunctionNode):
    source_fn: Callable[float, float]
    special_values: dict[Node, Node] = {}

    @property
    def name(self):
        return self.source_fn.__name__

    @staticmethod
    @abstractmethod
    def _should_evaluate(argument: Node) -> bool:
        pass

    def reduce_no_eval(self, depth=-1) -> Node:
        if depth == 0:
            return self

        reduced_arg = self.argument.reduce_no_eval(depth-1)

        if reduced_arg in self.special_values.keys():
            return self.special_values[reduced_arg]

        return type(self)(reduced_arg)

    def reduce(self, depth=-1) -> Node:
        if depth == 0:
            return self

        reduced_arg = self.argument.reduce(depth - 1)
        reduced_self = type(self)(reduced_arg)
        inc_indent()
        debug(f"{reduced_self = }", flag='match')
        debug(f"{reduced_arg = }", flag='match')
        dec_indent()
        if self._should_evaluate(reduced_arg):
            debug(f"should evaluate", flag='match')
            return reduced_self.evaluate()

        return reduced_self

    def evaluate(self) -> Node:
        if self.argument in self.special_values.keys():
            return self.special_values[self.argument]

        return Value(self.source_fn(self.argument.approximate()))

    def approximate(self) -> float:
        if self.argument in self.special_values.keys():
            return self.special_values[self.argument].approximate()

        return self.source_fn(self.argument.approximate())


import math


class Sin(PythonFunctionNode):
    source_fn = math.sin
    special_values = {Value(0.0): Value(0.0)}

    @staticmethod
    def _should_evaluate(argument: Node) -> bool:
        return False
