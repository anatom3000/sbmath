from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Optional

from sbmath.expression.core import Expression, Variable, MatchResult, Value, Wildcard
from sbmath.expression.context import MissingContextError
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

    def __call__(self, argument: Expression) -> FunctionApplication:
        return FunctionApplication(self, argument)

    @abstractmethod
    def reduce_func(self, argument: Expression, depth: int, evaluate: bool) -> Optional[Expression]:
        pass

    @abstractmethod
    def can_evaluate(self, argument: Expression) -> bool:
        pass

    @abstractmethod
    def evaluate(self, argument: Expression) -> Expression:
        pass

    @classmethod
    def from_expression(cls, body: expression, parameter: Expression, name: str = None) -> Function:
        if name is None:
            name = f"_anonymous_{abs(hash((parameter, body)))}"

        return ExpressionFunction(name, parameter, body)


class PythonFunction(Function):
    def __hash__(self):
        return hash((self.name, self.pyfunc, tuple(self.special_values.items())))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def _derivative(self) -> Function:
        return self._deriv

    def reduce_func(self, argument: Expression, depth: int, evaluate: bool) -> Optional[Expression]:
        for pattern, image in self.special_values.items():
            debug(f"{argument = }, {pattern = }, {image = }", flag='reduce_func')
            new = argument.morph(pattern, image, evaluate=evaluate, reduce=False)
            if new is not None:
                # print(f"reducing {new = } from {self = }, {argument = }")
                return new.reduce(depth-1, evaluate=evaluate)

        return None

    def can_evaluate(self, argument: Expression) -> bool:
        return argument.is_evaluable()

    def evaluate(self, argument: Expression) -> Expression:
        return self(argument.evaluate()).reduce()

    def __init__(self, func: Callable[[float], float], special_values: dict[Expression, Expression] = None, name: str = None):
        self.pyfunc = func
        if name is None:
            name = self.pyfunc.__name__
        self.name = name
        self.special_values = {} if special_values is None else special_values


class ExpressionFunction(Function):
    def __hash__(self):
        return hash((self.name, self.parameter, self.body))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def reduce_func(self, argument: Expression, depth: int, evaluate: bool) -> Optional[Expression]:
        return self.body.substitute(self.parameter, argument.reduce(depth-1, evaluate=evaluate)).reduce(depth, evaluate=evaluate)

    def can_evaluate(self, argument: Expression) -> bool:
        return argument.is_evaluable()

    def evaluate(self, argument: Expression) -> Expression:
        return self.body.substitute(self.parameter, argument.evaluate()).evaluate()

    def __init__(self, name: str, parameter: Expression, body: Expression):
        self.name = name
        self.parameter = parameter
        self.body = body


class FunctionApplication(Expression):
    def approximate(self) -> float:
        return self.function.evaluate(self.argument).approximate()

    def evaluate(self) -> Expression:
        return self.function.evaluate(self.argument)

    def reduce(self, depth=-1, *, evaluate: bool = True) -> Expression:
        if depth == 0:
            return self

        try:
            r = self.function.reduce_func(self.argument, depth, evaluate=evaluate)
            if r is not None:
                return r
        except MissingContextError:
            pass

        return self.change_argument(self.argument.reduce(depth - 1, evaluate=evaluate))

    def is_evaluable(self) -> bool:
        try:
            return self.function.can_evaluate(self.argument)
        except MissingContextError:
            return False

    def __eq__(self, other: Expression) -> bool:
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

    def _replace_identifiers(self, match_result: MatchResult) -> Expression:
        return self.change_argument(self.argument._replace_identifiers(match_result))

    def change_argument(self, new_argument: Expression):
        new = type(self)(self._function, new_argument)
        new.context = self.context
        return new

    def __init__(self, function: str | Function, argument: Expression):
        self._function = function
        self.argument = argument

    def _replace_in_children(self, old_pattern: Expression, new_pattern: Expression, evaluate: bool, reduce: bool) -> Expression:
        return self.change_argument(self.argument.replace(old_pattern, new_pattern, evaluate=evaluate, reduce=reduce))

    def _substitute_in_children(self, pattern: Expression, new: Expression, evaluate: bool, reduce: bool) -> Expression:
        return self.change_argument(self.argument.substitute(pattern, new, evaluate=evaluate, reduce=reduce))

    def _apply_on_children(self, pattern: Expression, modifier: Callable[[MatchResult], Expression], evaluate: bool, reduce: bool) -> Expression:
        return self.change_argument(self.argument.apply_on(pattern, modifier, evaluate=evaluate, reduce=reduce))

    def contains(self, pattern: Expression, *, evaluate: bool = True, reduce: bool = True) -> bool:
        return pattern.matches(self, evaluate=evaluate, reduce=reduce) is not None or self.argument.contains(pattern, evaluate=evaluate, reduce=reduce)

    def _match_no_reduce(self, value: Expression, state: MatchResult, evaluate: bool, reduce: bool) -> Optional[MatchResult]:
        if not isinstance(value, type(self)):
            return None

        if self.function != value.function:
            return None

        return self.argument.matches(value.argument, state, evaluate=evaluate, reduce=reduce)

    def matches(self, value: Expression, state: MatchResult = None, *, evaluate: bool = True, reduce: bool = True) -> Optional[MatchResult]:
        if state is None:
            state = MatchResult()

        no_reduce_state = self._match_no_reduce(value, copy.deepcopy(state), evaluate, reduce)

        if no_reduce_state is not None and not (no_reduce_state.weak and reduce):
            dec_indent()
            return no_reduce_state

        if not reduce:
            return None

        reduced_self = self.reduce(evaluate=evaluate)
        reduced_value = value.reduce(evaluate=evaluate)

        # if not isinstance(reduced_self, FunctionApplication):
        #     return reduced_self.matches(reduced_value, state)

        # noinspection PyProtectedMember
        return reduced_self._match_no_reduce(reduced_value, state, evaluate, reduce)

    def __str__(self):
        if isinstance(self._function, str):
            func_name = self._function
        else:
            func_name = self._function.name

        return f"{func_name}({self.argument})"

    def __hash__(self):
        return hash((self.__class__.__name__, self._function, self.argument))


class FunctionWildcard(Wildcard):
    def __eq__(self, other):
        return isinstance(other, FunctionWildcard)  \
            and self.name == other.name             \
            and self.argument == self.argument      \
            and self.constraints == self.constraints

    def __hash__(self):
        constraints_hashes = tuple(sorted(hash((k, v)) for k, v in self.constraints.items()))

        return hash((self.__class__.__name__, self.name, self.argument, constraints_hashes))

    def change_argument(self, new_argument: Expression):
        return type(self)(self.name, new_argument, **self.constraints)

    def reduce(self, depth=-1, *, evaluate: bool = True) -> Expression:
        if depth == 0:
            return self

        return self.change_argument(self.argument.reduce(depth - 1, evaluate=evaluate))

    def _match_contraints(self, value: Expression, evaluate: bool, reduce: bool) -> bool:
        # TODO: function constraints
        return True

    def matches(self, value: Expression, state: MatchResult = None, *, evaluate: bool = True, reduce: bool = True) -> Optional[MatchResult]:
        if state is None:
            state = MatchResult()

        if not isinstance(value, FunctionApplication):
            return None

        maybe_new = self.argument.matches(value.argument, copy.deepcopy(state), evaluate=evaluate, reduce=reduce)
        if maybe_new:
            state = maybe_new
        else:
            maybe_new = self.argument.reduce(evaluate=evaluate).matches(value.argument.reduce(), copy.deepcopy(state), evaluate=evaluate, reduce=reduce)
            if maybe_new:
                state = maybe_new
            else:
                return None

        if self.name == '_':
            return state if self._match_contraints(value, evaluate, reduce) else None

        if self.name in state.functions_wildcards.keys() and value.matches(state.functions_wildcards[self.name], evaluate=evaluate, reduce=reduce) is None:
            return None

        if self._match_contraints(value, evaluate, reduce):
            try:
                state.functions_wildcards[self.name] = value.function
            except MissingContextError:
                state.functions_wildcards[self.name] = value._function
            return state

        return None

    def _replace_identifiers(self, match_result: MatchResult) -> Expression:
        if self.name in match_result.functions_wildcards:
            return FunctionApplication(match_result.functions_wildcards[self.name], self.argument._replace_identifiers(match_result))
        else:
            return self.change_argument(self.argument._replace_identifiers(match_result))

    def __str__(self):
        func = f"[{self.name}, "

        for k, v in self.constraints.items():
            func += f"{k}={v}, "

        func = func[:-2] + "]"

        return f"{func}({self.argument})"

    @classmethod
    def from_wildcard(cls, wildcard: Wildcard, argument: Expression):
        return cls(wildcard.name, argument, **wildcard.constraints)

    def __init__(self, name: str, argument: Expression, **constraints: Expression):
        super().__init__(name, **constraints)
        self.argument = argument


__all__ = ['Function', 'PythonFunction', 'ExpressionFunction', 'FunctionApplication', 'FunctionWildcard']
