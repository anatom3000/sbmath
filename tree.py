from __future__ import annotations

import copy
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from numbers import Real
from typing import Optional

import utils


class Node(ABC):

    def __neg__(self):
        if isinstance(self, Value):
            return Value(-self.data)
        else:
            return MulAndDiv.mul(Value(-1.0), self)

    def __add__(self, other) -> Node:
        if isinstance(other, Real):
            return AddAndSub.add(self, Value(float(other)))
        elif isinstance(other, Node):
            return AddAndSub.add(self, other)
        else:
            return NotImplemented

    def __radd__(self, other) -> Node:
        if isinstance(other, Real):
            return AddAndSub.add(Value(float(other)), self)
        elif isinstance(other, Node):
            return AddAndSub.add(other, self)
        else:
            return NotImplemented

    def __sub__(self, other) -> Node:
        if isinstance(other, Real):
            return AddAndSub.sub(self, Value(float(other)))
        elif isinstance(other, Node):
            return AddAndSub.sub(self, other)
        else:
            return NotImplemented

    def __rsub__(self, other) -> Node:
        if isinstance(other, Real):
            return AddAndSub.sub(Value(float(other)), self)
        elif isinstance(other, Node):
            return AddAndSub.sub(other, self)
        else:
            return NotImplemented

    def __mul__(self, other) -> Node:
        if isinstance(other, Real):
            return MulAndDiv.mul(self, Value(float(other)))
        elif isinstance(other, Node):
            return MulAndDiv.mul(self, other)
        else:
            return NotImplemented

    def __rmul__(self, other) -> Node:
        if isinstance(other, Real):
            return MulAndDiv.mul(Value(float(other)), self)
        elif isinstance(other, Node):
            return MulAndDiv.mul(other, self)
        else:
            return NotImplemented

    def __truediv__(self, other) -> Node:
        if isinstance(other, Real):
            return MulAndDiv.div(self, Value(float(other)))
        elif isinstance(other, Node):
            return MulAndDiv.div(self, other)
        else:
            return NotImplemented

    def __rtruediv__(self, other) -> Node:
        if isinstance(other, Real):
            return MulAndDiv.div(Value(float(other)), self)
        elif isinstance(other, Node):
            return MulAndDiv.div(other, self)
        else:
            return NotImplemented

    def __pow__(self, other) -> Node:
        if isinstance(other, Real):
            return Pow(self, Value(float(other)))
        elif isinstance(other, Node):
            return Pow(self, other)
        else:
            return NotImplemented

    def __rpow__(self, other) -> Node:
        if isinstance(other, Real):
            return Pow(Value(float(other)), self)
        elif isinstance(other, Node):
            return Pow(other, self)
        else:
            return NotImplemented

    def __str__(self) -> str:
        return f"Node({self.__class__.__name__})"

    def __repr__(self) -> str:
        return self.__str__()

    @abstractmethod
    def is_evaluable(self) -> bool:
        pass

    @abstractmethod
    def evaluate(self) -> float:
        pass  # TODO: make Node.evaluable return a Node (for more complex values that can only be approximated like √2)

    def matches(self, value: Node, state: MatchResult = None) -> Optional[MatchResult]:
        if state is None:
            state = MatchResult()

        return state if self == value else None

    @abstractmethod
    def _replace_identifiers(self, match_result: MatchResult) -> Node:
        pass

    def replace(self, old_pattern: Node, new_pattern: Node) -> Optional[Node]:
        m = old_pattern.matches(self)
        if m is None:
            return None

        try:
            new = new_pattern._replace_identifiers(m)
        except ReplacingError:
            return None
        return new

    @abstractmethod
    def reduce(self) -> Node:
        pass

    @abstractmethod
    def contains(self, pattern: Node) -> bool:
        pass

    def __contains__(self, item):
        return self.contains(item)


class Leaf(Node, ABC):
    def __eq__(self, other):
        return isinstance(other, type(self)) and self.data == other.data

    def __hash__(self):
        return hash(f"{type(self)}({hash(self.data)})")

    def __str__(self):
        return str(self.data)

    @abstractmethod
    def is_evaluable(self) -> bool:
        pass

    def reduce(self) -> Node:
        return self

    def contains(self, pattern: Node) -> bool:
        return pattern.matches(self) is not None

    def _replace_identifiers(self, match_result: MatchResult) -> Node:
        return self

    def __init__(self, data):
        super().__init__()
        self.data = data


class Value(Leaf):
    def __hash__(self):
        return hash(self.data)

    def __eq__(self, other):
        return isinstance(other, Node) and other.is_evaluable() and other.evaluate() == self.data

    def is_evaluable(self) -> bool:
        return True

    def evaluate(self) -> float:
        return self.data


class Variable(Leaf):
    def is_evaluable(self) -> bool:
        return False

    def evaluate(self) -> float:
        raise TypeError("can't evaluate a variable")


class Wildcard(Node):
    def contains(self, pattern: Node) -> bool:
        return False

    def _match_contraints(self, value: Node) -> bool:
        if "eval" in self.constraints.keys():
            if self.constraints["eval"] == Value(1.0) and not value.is_evaluable():
                return False
            if self.constraints["eval"] == Value(0.0) and value.is_evaluable():
                return False

        return True

    def matches(self, value: Node, state: MatchResult = None) -> Optional[MatchResult]:
        if state is None:
            state = MatchResult()

        if self.name == '_':
            return state if self._match_contraints(value) else None

        if self.name in state.wildcards.keys() and value.matches(state.wildcards[self.name]) is None:
            return None

        if self._match_contraints(value):
            state.wildcards[self.name] = value
            return state

        return None

    def _replace_identifiers(self, match_result: MatchResult) -> Node:
        if self.name not in match_result.wildcards:
            raise ReplacingError(f"name '{self.name}' not found in ID mapping")

        return match_result.wildcards[self.name]

    def is_evaluable(self) -> bool:
        return False

    def evaluate(self) -> float:
        raise TypeError("can't evaluate a wildcard")

    def reduce(self) -> Node:
        return self

    def __str__(self):
        text = f"[{self.name}, "

        for k, v in self.constraints.items():
            text += f"{k}={v}, "

        text = text[:-2] + "]"

        return text

    def __init__(self, name: str, **constraints: Node):
        self.name = name
        self.constraints = constraints


class BinOp(Node, ABC):
    name: str

    def matches(self, value: Node, state: MatchResult = None) -> Optional[MatchResult]:
        if self.is_evaluable() and value.is_evaluable() and self.evaluate() == value.evaluate():
            return state
        else:
            return None

    # noinspection PyProtectedMember
    def _replace_identifiers(self, match_result: MatchResult) -> Node:
        return type(self)(*(x._replace_identifiers(match_result) for x in self.values))

    def is_evaluable(self) -> bool:
        return all(c.is_evaluable() for c in self.values)

    def contains(self, pattern: Node) -> bool:
        return any(c.contains(pattern) for c in self.values)

    def __init__(self, *values):
        self.values = values

    @staticmethod
    @abstractmethod
    def evaluator(*values: float) -> float:
        pass

    def evaluate(self) -> float:
        if not self.is_evaluable():
            raise ValueError("cannot evaluate expression")

        return self.evaluator(*(v.evaluate() for v in self.values))

    def __str__(self):
        return '( ' + f' {self.name} '.join(map(lambda x: str(x), self.values)) + ' )'


class AdvancedBinOp(Node, ABC):
    base_operation_symbol: str
    inverse_operation_symbol: str

    identity: Node

    @staticmethod
    @abstractmethod
    def _invert_value(value: Node) -> Node:
        pass

    @staticmethod
    @abstractmethod
    def _repeat_value(value: Node, times: int) -> Node:
        pass

    @staticmethod
    @abstractmethod
    def _should_invert_value(value: float) -> bool:
        pass

    def reduce(self) -> Node:
        # FIXME: (2x).reduce() == x

        eval_part = type(self)(
            filter(lambda x: x.is_evaluable(), self.base_values),
            filter(lambda x: x.is_evaluable(), self.inverted_values)
        )

        eval_result = eval_part.evaluate()

        value_occurences = defaultdict(int)

        for value in filter(lambda x: not x.is_evaluable(), self.base_values):
            value_occurences[value.reduce()] += 1

        for value in filter(lambda x: not x.is_evaluable(), self.inverted_values):
            value_occurences[value.reduce()] -= 1

        base_values = []
        inverted_values = []

        for key, value in value_occurences.items():
            if value == 0:
                continue

            elif value == 1:
                base_values.append(key)

            elif value == -1:
                inverted_values.append(key)

            elif value > 1:
                base_values.append(self._repeat_value(key, value))
            else:
                inverted_values.append(self._repeat_value(self._invert_value(key), value))

        if (not base_values) and (not inverted_values):
            return Value(eval_result)

        if eval_result == self.identity.evaluate():
            if len(base_values) == 1 and len(inverted_values) == 0:
                return base_values[0]
            if len(base_values) == 0 and len(inverted_values) == 1:
                return inverted_values[0]

        elif self._should_invert_value(eval_result):
            inverted_values.append(self._invert_value(Value(eval_result)))
        else:
            base_values.append(Value(eval_result))

        return type(self)(base_values, inverted_values)

    def _match_evaluable(self, value: Node, state: MatchResult = None) -> Optional[MatchResult]:
        return state if self.evaluate() == value.evaluate() else None

    def _match_no_wildcards(self, value: AdvancedBinOp, state: MatchResult = None) \
            -> (MatchResult, AdvancedBinOp, AdvancedBinOp):
        remaining_value = type(self)()
        remaining_pattern = copy.deepcopy(self)
        new_state = copy.deepcopy(state)

        index: int = 0  # for linters
        for val_value in value.base_values:
            found = False
            for index, pat_value in enumerate(remaining_pattern.base_values):
                match_result = pat_value.matches(val_value, new_state)
                if match_result is not None:
                    new_state = match_result
                    found = True
                    break
            if found:
                # PyCharm complains but `index` is garanteed to be set
                # because the only way `found` can be True is if one of the loop executed at least once
                del remaining_pattern.base_values[index]
            else:
                for index, pat_value in enumerate(remaining_pattern.inverted_values):
                    match_result = pat_value.matches(self._invert_value(val_value), new_state)
                    if match_result is not None:
                        new_state = match_result
                        found = True
                        break
                if found:
                    del remaining_pattern.inverted_values[index]
                else:
                    remaining_value.base_values.append(val_value)

        for val_value in value.inverted_values:
            found = False
            for index, pat_value in enumerate(remaining_pattern.inverted_values):
                match_result = pat_value.matches(val_value, new_state)
                if match_result is not None:
                    new_state = match_result
                    found = True
                    break
            if found:
                # PyCharm complains but `index` is garanteed to be set
                # because the only way `found` can be True is if one of the loop executed at least once
                del remaining_pattern.inverted_values[index]
            else:
                for index, pat_value in enumerate(remaining_pattern.base_values):
                    match_result = pat_value.matches(self._invert_value(val_value), new_state)
                    if match_result is not None:
                        new_state = match_result
                        found = True
                        break
                if found:
                    del remaining_pattern.base_values[index]
                else:
                    remaining_value.inverted_values.append(val_value)

        return new_state, remaining_pattern, remaining_value

    def _remove_wildcard_match(self, value: Node, wildcard: Wildcard, match_table: utils.TwoWayMapping,
                               state: MatchResult) \
            -> (utils.TwoWayMapping, MatchResult):

        similars = [similar for similar in match_table.get_from_value(wildcard)
                    if len(list(match_table.get_from_key(similar))) == 1]

        if len(similars) == 0:
            state = wildcard.matches(value, state)
            match_table.remove_key(value)

        elif len(similars) == 1:
            state = wildcard.matches(similars[0], state)
            match_table.remove_key(similars[0])
        else:
            state = wildcard.matches(type(self)(base_values=similars), state)
            for s in similars:
                match_table.remove_key(s)

        # p = lambda a: ({k: a.get_from_key(k) for k in a.keys()}, {k: a.get_from_value(k) for k in a.values()})
        match_table.remove_value(wildcard)

        return match_table, state

    def _clean_up_single_wildcards(self, match_table: utils.TwoWayMapping, state: MatchResult) \
            -> Optional[(utils.TwoWayMapping, MatchResult)]:

        while any(len(match_table.get_from_key(k)) < 2 for k in match_table.keys()):
            for value in list(match_table.keys()):
                if value not in match_table.keys():
                    continue

                matches = list(match_table.get_from_key(value))

                if len(matches) == 0:
                    return None
                elif len(matches) == 1:
                    match_table, state = self._remove_wildcard_match(value, matches[0], match_table, state)

        return match_table, state

    def _match_wildcards(self, value: AdvancedBinOp, state: MatchResult) -> Optional[MatchResult]:
        match_table = utils.TwoWayMapping()
        for value in value.base_values:
            for wildcard in self.base_values:
                if wildcard.matches(value, copy.deepcopy(state)):
                    match_table.add(value, wildcard)

        result = self._clean_up_single_wildcards(match_table, state)
        if result is None:
            return None

        match_table, state = result

        for value in list(match_table.keys()):
            if value not in list(match_table.keys()):
                continue
            match_table, state = self._remove_wildcard_match(value, match_table.get_from_key(value)[0], match_table,
                                                             state)

        if len(list(match_table.keys())) != 0:
            return None

        return state

    def matches(self, value: Node, state: MatchResult = None) -> Optional[MatchResult]:
        if state is None:
            state = MatchResult()

        if self.is_evaluable() and value.is_evaluable():
            return self._match_evaluable(value, state)
        elif not isinstance(value, type(self)):
            return None

        no_wildcard_self = type(self)(
            list(filter(lambda x: not isinstance(x, Wildcard) and self.identity.matches(x) is None, self.base_values)),
            list(filter(lambda x: not isinstance(x, Wildcard) and self.identity.matches(x) is None,
                        self.inverted_values))
        )

        value: AdvancedBinOp  # I love Python's type system...

        state, remaining_pattern, remaining_value = no_wildcard_self._match_no_wildcards(value, state)

        if remaining_pattern.base_values or remaining_pattern.inverted_values:
            return None

        wildcard_self = type(self)(
            list(filter(lambda x: isinstance(x, Wildcard), self.base_values)),
            list(filter(lambda x: isinstance(x, Wildcard), self.inverted_values))
        )

        result = wildcard_self._match_wildcards(remaining_value, state)  # type: ignore

        return result

    def _replace_identifiers(self, match_result: MatchResult) -> Node:
        return type(self)(
            base_values=(x._replace_identifiers(match_result) for x in self.base_values),
            inverted_values=(x._replace_identifiers(match_result) for x in self.inverted_values)
        )

    def is_evaluable(self) -> bool:
        return all(c.is_evaluable() for c in itertools.chain(self.base_values, self.inverted_values))

    def contains(self, pattern: Node) -> bool:
        return pattern.matches(self) is not None or any(
            c.contains(pattern) for c in itertools.chain(self.base_values, self.inverted_values))

    def __str__(self):
        text = '( '

        text += f' {self.base_operation_symbol} '.join(map(str, self.base_values))

        if self.inverted_values:
            text += f' {self.inverse_operation_symbol} ' if self.base_values else f'{self.inverse_operation_symbol} '

        text += f' {self.inverse_operation_symbol} '.join(map(str, self.inverted_values))

        return text + ' )'

    def __init__(self, base_values: Iterable[Node] = None, inverted_values: Iterable[Node] = None):
        self.base_values: list[Node] = []
        self.inverted_values: list[Node] = []

        if base_values is not None:
            for val in base_values:
                if isinstance(val, type(self)):
                    self.base_values.extend(val.base_values)
                    self.inverted_values.extend(val.inverted_values)
                else:
                    self.base_values.append(val)

        if inverted_values is not None:
            for val in inverted_values:
                if isinstance(val, type(self)):
                    self.inverted_values.extend(val.base_values)
                    self.base_values.extend(val.inverted_values)
                else:
                    self.inverted_values.append(val)


class AddAndSub(AdvancedBinOp):
    base_operation_symbol = '+'

    inverse_operation_symbol = '-'

    identity = Value(0.0)

    @staticmethod
    def _invert_value(value: Node) -> Node:
        return -value

    @staticmethod
    def _repeat_value(value: Node, times: int) -> Node:
        return MulAndDiv.mul(Value(times), value)

    @staticmethod
    def _should_invert_value(value: float) -> bool:
        return value < 0.0

    @classmethod
    def add(cls, *values: Node):
        return cls(base_values=values)

    @classmethod
    def sub(cls, *values: Node, positive_first_node: bool = True):
        if positive_first_node:
            return cls(base_values=values[:1], inverted_values=values[1:])
        else:
            return cls(inverted_values=values)

    def evaluate(self) -> float:
        result = 0.0
        for c in self.base_values:
            result += c.evaluate()
        for c in self.inverted_values:
            result -= c.evaluate()

        return result


class MulAndDiv(AdvancedBinOp):
    base_operation_symbol = '*'
    inverse_operation_symbol = '/'

    identity = Value(1.0)

    @staticmethod
    def _repeat_value(value: Node, times: int) -> Node:
        return Pow(Value(times), value)

    @staticmethod
    def _invert_value(value: Node) -> Node:
        if isinstance(value, Value) and -1 <= value.data <= 1:
            return Value(1.0 / value.data)

        return MulAndDiv.div(Value(1.0), value)

    @staticmethod
    def _should_invert_value(value: float) -> bool:
        return -1 < value < 1

    @classmethod
    def mul(cls, *values: Node):
        return cls(base_values=values)

    @classmethod
    def div(cls, *values: Node, multiply_first_node: bool = True):
        if multiply_first_node:
            return cls(base_values=values[:1], inverted_values=values[1:])
        else:
            return cls(inverted_values=values)

    def evaluate(self) -> float:
        result = 1.0
        for c in self.base_values:
            result *= c.evaluate()
        for c in self.inverted_values:
            result /= c.evaluate()

        return result


class Pow(BinOp):
    name = '^'

    def reduce(self) -> Node:
        reduced_values = []
        can_reduce_evaluable = True
        for value in self.values[::-1]:
            value = value.reduce()
            evaluable = value.is_evaluable()

            if not evaluable:
                can_reduce_evaluable = False

            if can_reduce_evaluable and evaluable and reduced_values:
                reduced_values[-1] = Value((value ** reduced_values[-1]).evaluate())

            else:
                reduced_values.append(value)

        if len(reduced_values) == 1 and can_reduce_evaluable:
            return reduced_values[0]

        return Pow(*reduced_values[::-1])

    @staticmethod
    def evaluator(*values: float) -> float:
        result = 1.0
        for v in values[::-1]:
            result = v ** result
        return result


@dataclass
class MatchResult:
    wildcards: dict[str, Node] = field(default_factory=dict)


class ReplacingError(Exception):
    pass
