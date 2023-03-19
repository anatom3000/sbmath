from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Optional

from sbmath.tree.base import Node, Variable, MatchResult, Value
from sbmath.tree.context import MissingContextError
from sbmath._utils import debug, inc_indent, dec_indent


class Function(ABC):
    name: str

    @abstractmethod
    def reduce_func(self, argument: Node, depth: int) -> Optional[Node]:
        pass

    @abstractmethod
    def can_evaluate(self, argument: Node) -> bool:
        pass

    @abstractmethod
    def evaluate(self, argument: Node) -> Node:
        pass


class PythonFunction(Function):

    def reduce_func(self, argument: Node, _depth: int) -> Optional[Node]:
        for pattern, image in self.special_values.items():
            new = argument.morph(pattern, image)
            if new is not None:
                return new.reduce(-1)

        return None

    def can_evaluate(self, argument: Node) -> bool:
        return argument.is_evaluable() and any(pat.matches(argument) for pat in self.special_values.keys())

    def evaluate(self, argument: Node) -> Node:
        argument = argument.evaluate()

        for pattern, image in self.special_values.items():
            new = argument.morph(pattern, image)
            if new is not None:
                return new.evaluate()

        return Value(self.pyfunc(argument.approximate()))

    def __init__(self, func: Callable[[float], float], special_values: dict[Node, Node] = None, name: str = None):
        self.pyfunc = func
        if name is None:
            name = self.pyfunc.__name__
        self.name = name
        self.special_values = {} if special_values is None else special_values


class NodeFunction(Function):

    def reduce_func(self, argument: Node, depth: int) -> Optional[Node]:
        return self.body.replace(self.parameter, argument.reduce(depth-1)).reduce(depth)

    def can_evaluate(self, argument: Node) -> bool:
        return argument.is_evaluable()

    def evaluate(self, argument: Node) -> Node:
        return self.body.replace(self.parameter, argument.evaluate()).evaluate()

    def __init__(self, name: str, parameter: Node, body: Node):
        self.name = name
        self.parameter = parameter
        self.body = body


class FunctionApplication(Node):

    def approximate(self) -> float:
        return self.function.evaluate(self.argument).approximate()

    def evaluate(self) -> Node:
        return self.function.evaluate(self.argument)

    def reduce(self, depth=-1) -> Node:
        if depth == 0:
            return self

        if self.is_evaluable():
            return self.evaluate()

        r = self.function.reduce_func(self.argument, depth)
        if r is not None:
            return r

        return self.change_argument(self.argument.reduce(depth - 1))

    def is_evaluable(self) -> bool:
        try:
            return self.function.can_evaluate(self.argument)
        except MissingContextError:
            return False

    def __eq__(self, other: Node) -> bool:
        return isinstance(other, FunctionApplication) \
           and self._function == other._function \
           and self.argument == self.argument

    @property
    def function(self) -> Function:
        if isinstance(self._function, str):
            if self.context is None:
                raise MissingContextError(f"could not get function '{self._function}' without context")
            elif self._function not in self.context.functions:
                raise MissingContextError(f"context exists but function '{self._function}' is undefined")
            else:
                return self.context.functions[self._function]

        return self._function

    def _replace_identifiers(self, match_result: MatchResult) -> Node:
        return self.change_argument(self.argument._replace_identifiers(match_result))

    def change_argument(self, new_argument: Node):
        return type(self)(self._function, new_argument)

    def __init__(self, function: str | Function, argument: Node):
        self._function = function
        self.argument = argument

    def replace(self, old: Node, new: Node):
        return new if old.matches(self) else self.change_argument(self.argument.replace(old, new))

    def contains(self, pattern: Node) -> bool:
        return pattern.matches(self) is not None or self.argument.contains(pattern)

    def _match_no_reduce(self, value: Node, state: MatchResult) -> Optional[MatchResult]:
        if not isinstance(value, type(self)):
            return None

        if self.function != value.function:
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

        # noinspection PyProtectedMember
        return reduced_self._match_no_reduce(reduced_value, state)

    def __str__(self):
        try:
            func_name = self.function.name
        except MissingContextError:
            func_name = self._function

        return f"{func_name}({self.argument})"

    def __hash__(self):
        return hash(str(self))
