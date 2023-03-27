from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Optional

from sbmath.tree.base import Node, Variable, MatchResult, Value, Wildcard
from sbmath.tree.context import MissingContextError
from sbmath._utils import debug, inc_indent, dec_indent


class Function(ABC):
    name: str

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    def __str__(self):
        return f"function {self.name}"

    def __repr__(self):
        return str(self)

    def __call__(self, argument: Node) -> FunctionApplication:
        return FunctionApplication(self, argument)

    @abstractmethod
    def reduce_func(self, argument: Node, depth: int) -> Optional[Node]:
        pass

    @abstractmethod
    def can_evaluate(self, argument: Node) -> bool:
        pass

    @abstractmethod
    def evaluate(self, argument: Node) -> Node:
        pass

    @classmethod
    def from_expression(cls, body: expression, parameter: Node, name: str = None) -> Function:
        if name is None:
            name = f"_anonymous_{hash((parameter, body))}"

        return NodeFunction(name, parameter, body)


class PythonFunction(Function):
    def __hash__(self):
        return hash((self.name, self.pyfunc, tuple(self.special_values)))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def _derivative(self) -> Function:
        return self._deriv

    def reduce_func(self, argument: Node, depth: int) -> Optional[Node]:
        for pattern, image in self.special_values.items():
            new = argument.morph(pattern, image)
            if new is not None:
                return new.reduce(depth-1)

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
    def __hash__(self):
        return hash((self.name, self.parameter, self.body))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def reduce_func(self, argument: Node, depth: int) -> Optional[Node]:
        return self.body.substitute(self.parameter, argument.reduce(depth-1)).reduce(depth)

    def can_evaluate(self, argument: Node) -> bool:
        return argument.is_evaluable()

    def evaluate(self, argument: Node) -> Node:
        return self.body.substitute(self.parameter, argument.evaluate()).evaluate()

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

        try:
            r = self.function.reduce_func(self.argument, depth)
            if r is not None:
                return r
        except MissingContextError:
            pass

        return self.change_argument(self.argument.reduce(depth - 1))

    def is_evaluable(self) -> bool:
        try:
            return self.function.can_evaluate(self.argument)
        except MissingContextError:
            return False

    def __eq__(self, other: Node) -> bool:
        return isinstance(other, FunctionApplication) \
           and self._function == other._function      \
           and self.argument == self.argument

    @property
    def function(self) -> Function:
        if isinstance(self._function, str):
            if self.context is None:
                raise MissingContextError(f"could not get function '{self._function}' without context")
            # print(self.context.__dict__)
            elif self._function not in self.context.functions.keys():
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

    def _replace_in_children(self, old_pattern: Node, new_pattern: Node) -> Node:
        return self.change_argument(self.argument.replace(old_pattern, new_pattern))

    def _substitute_in_children(self, pattern: Node, new: Node) -> Node:
        return self.change_argument(self.argument.substitute(pattern, new))

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

        if not isinstance(reduced_self, FunctionApplication):
            return reduced_self.matches(reduced_value, state)

        # noinspection PyProtectedMember
        return reduced_self._match_no_reduce(reduced_value, state)

    def __str__(self):
        if isinstance(self._function, str):
            func_name = self._function
        else:
            func_name = self._function.name

        return f"{func_name}({self.argument})"

    def __hash__(self):
        return hash(str(self))


class FunctionWildcard(Wildcard):
    def __eq__(self, other):
        return isinstance(other, FunctionWildcard)  \
            and self.name == other.name             \
            and self.argument == self.argument      \
            and self.constraints == self.constraints

    def change_argument(self, new_argument: Node):
        return type(self)(self.name, new_argument, **self.constraints)

    def reduce(self, depth=-1) -> Node:
        if depth == 0:
            return self

        return self.change_argument(self.argument.reduce(depth - 1))

    def _match_contraints(self, value: Node) -> bool:
        # TODO: function constraints

        return True

    def matches(self, value: Node, state: MatchResult = None) -> Optional[MatchResult]:
        if state is None:
            state = MatchResult()

        if not isinstance(value, FunctionApplication):
            return None

        maybe_new = self.argument.matches(value.argument, copy.deepcopy(state))
        if maybe_new:
            state = maybe_new
        else:
            maybe_new = self.argument.reduce().matches(value.argument.reduce(), copy.deepcopy(state))
            if maybe_new:
                state = maybe_new
            else:
                return None

        if self.name == '_':
            return state if self._match_contraints(value) else None

        if self.name in state.functions_wildcards.keys() and value.matches(state.functions_wildcards[self.name]) is None:
            return None

        if self._match_contraints(value):
            state.functions_wildcards[self.name] = value.function
            return state

        return None

    def _replace_identifiers(self, match_result: MatchResult) -> Node:
        return self.change_argument(self.argument._replace_identifiers(match_result))

    def __str__(self):
        func = f"[{self.name}, "

        for k, v in self.constraints.items():
            func += f"{k}={v}, "

        func = func[:-2] + "]"

        return f"{func}({self.argument})"

    @classmethod
    def from_wildcard(cls, wildcard: Wildcard, argument: Node):
        return cls(wildcard.name, argument, **wildcard.constraints)

    def __init__(self, name: str, argument: Node, **constraints: Node):
        super().__init__(name, **constraints)
        self.argument = argument

