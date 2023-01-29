from __future__ import annotations

import copy
import itertools
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

Numerics = (float, int)


class Node(ABC):
    def __neg__(self):
        if isinstance(self, Value):
            return Value(-self.data)
        else:
            return Mul(Value(-1.0), self)

    def __add__(self, other) -> Node:
        if isinstance(other, Numerics):
            return self + Value(float(other))
        elif isinstance(other, Node):
            return AddAndSub.add(self, other)
        else:
            return NotImplemented

    # TODO: fix (3 + Node) => Add(Node(...), 3) (should be (3 + Node) => Add(3, Node(...))
    __radd__ = __add__

    def __sub__(self, other) -> Node:
        if isinstance(other, Numerics):
            return self - Value(float(other))
        elif isinstance(other, Node):
            return AddAndSub.sub(self, other)
        else:
            return NotImplemented

    __rsub__ = __sub__

    def __mul__(self, other) -> Node:
        if isinstance(other, Numerics):
            return self * Value(float(other))
        elif isinstance(other, Node):
            return Mul(self, other)
        else:
            return NotImplemented

    def __truediv__(self, other) -> Node:
        if isinstance(other, Numerics):
            return self / Value(float(other))
        elif isinstance(other, Node):
            return Div(self, other)
        else:
            return NotImplemented

    def __pow__(self, other) -> Node:
        if isinstance(other, Numerics):
            return self ** Value(float(other))
        elif isinstance(other, Node):
            return Pow(self, other)
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
        pass

    def matches(self, value: Node, state: MatchResult = None) -> Optional[MatchResult]:
        if state is None:
            return MatchResult()

        return state if self == value else None

    def replace(self, pattern: Node, value: Node):
        pass

    @abstractmethod
    def contains(self, node: Node):
        pass

    def __contains__(self, item):
        return self.contains(item)


class Leaf(Node, ABC):
    def __eq__(self, other):
        return isinstance(other, type(self)) and self.data == other.data

    @abstractmethod
    def is_evaluable(self) -> bool:
        pass

    def contains(self, node: Node):
        return self.data == node.data

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.data if self.data is not None else ''})"


class Value(Leaf):
    def __eq__(self, other):
        return isinstance(other, Node) and other.is_evaluable() and other.evaluate() == self.data

    def contains(self, node: Node):
        return self == node

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
    def contains(self, node: Node):
        return False

    def _match_contraints(self, value: Node) -> bool:
        return True

    def matches(self, value: Node, state: MatchResult = None) -> Optional[MatchResult]:
        print(f"matching {self} to {value} ; {state}")
        if state is None:
            state = MatchResult()

        if self.name == '_':
            return state if self._match_contraints(value) else None

        if self.name in state.wildcards.keys() and not (value.matches(state.wildcards[self.name])):
            return None

        if self._match_contraints(value):
            state.wildcards[self.name] = value
            return state

        return None

    def is_evaluable(self) -> bool:
        return False

    def evaluate(self) -> float:
        raise TypeError("can't evaluate a wildcard")

    def __str__(self):
        return f"[{self.name}]"

    def __init__(self, name: str):
        self.name = name


class BinOp(Node, ABC):
    name: str

    def matches(self, value: Node, state: MatchResult = None) -> Optional[MatchResult]:
        if self.is_evaluable() and value.is_evaluable() and self.evaluate() == value.evaluate():
            return state
        else:
            return None

    def is_evaluable(self) -> bool:
        return all(c.is_evaluable() for c in self.values)

    def contains(self, node: Node):
        return any(c.contains(node) for c in self.values)

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


class AddAndSub(Node):
    name = "+"

    def _match_evaluable(self, value: Node, state: MatchResult = None) -> Optional[MatchResult]:
        return state if self.evaluate() == value.evaluate() else None

    def _match_no_wildcards(self, value: AddAndSub, state: MatchResult = None) \
            -> (MatchResult, AddAndSub, AddAndSub):
        remaining_value = AddAndSub()
        remaining_pattern = copy.deepcopy(self)
        new_state = copy.deepcopy(state)

        index: int = 0  # for linters
        for val_value in value.added_values:
            found = False
            for index, pat_value in enumerate(remaining_pattern.added_values):
                match_result = pat_value.matches(val_value, new_state)
                if match_result is not None:
                    new_state = match_result
                    found = True
                    break
            if found:
                # PyCharm complains but `index` is garanteed to be set
                # because the only way `found` can be True is if one of the loop executed at least once
                del remaining_pattern.added_values[index]
            else:
                for index, pat_value in enumerate(remaining_pattern.substracted_values):
                    match_result = pat_value.matches(-val_value, new_state)
                    if match_result is not None:
                        new_state = match_result
                        found = True
                        break
                if found:
                    del remaining_pattern.substracted_values[index]
                else:
                    remaining_value.added_values.append(val_value)

        for val_value in value.substracted_values:
            found = False
            for index, pat_value in remaining_pattern.substracted_values:
                match_result = pat_value.matches(val_value, new_state)
                if match_result is not None:
                    new_state = match_result
                    found = True
                    break
            if found:
                # PyCharm complains but `index` is garanteed to be set
                # because the only way `found` can be True is if one of the loop executed at least once
                del remaining_pattern.substracted_values[index]
            else:
                for index, pat_value in remaining_pattern.added_values:
                    match_result = pat_value.matches(-val_value, new_state)
                    if match_result is not None:
                        new_state = match_result
                        found = True
                        break
                if found:
                    del remaining_pattern.added_values[index]
                else:
                    remaining_value.substracted_values.append(val_value)

        return new_state, remaining_pattern, remaining_value

    def _match_wildcards(self, value: AddAndSub, state: MatchResult) -> Optional[MatchResult]:
        match_table: dict[int, list[int]] = {}
        for vindex, value in enumerate(value.added_values):
            match_table[vindex] = []
            for windex, wildcard in enumerate(self.added_values):
                if wildcard.matches(value, copy.deepcopy(state)):
                    match_table[vindex].append(windex)

        print(match_table)
        return None



    def matches(self, value: Node, state: MatchResult = None) -> Optional[MatchResult]:
        if state is None:
            state = MatchResult()

        if self.is_evaluable() and value.is_evaluable():
            return self._match_evaluable(value, state)
        elif not isinstance(value, AddAndSub):
            return None

        self: AddAndSub
        value: AddAndSub

        no_wildcard_self = AddAndSub(
            list(filter(lambda x: not isinstance(x, Wildcard), self.added_values)),
            list(filter(lambda x: not isinstance(x, Wildcard), self.substracted_values))
        )

        state, remaining_pattern, remaining_value = no_wildcard_self._match_no_wildcards(value, state)

        if remaining_pattern.added_values or remaining_pattern.substracted_values:
            return None

        if not (remaining_value.added_values or remaining_value.substracted_values):
            return state

        wildcard_self = AddAndSub(
            list(filter(lambda x: isinstance(x, Wildcard), self.added_values)),
            list(filter(lambda x: isinstance(x, Wildcard), self.substracted_values))
        )

        return wildcard_self._match_wildcards(remaining_value, state)

    def is_evaluable(self) -> bool:
        return all(c.is_evaluable() for c in itertools.chain(self.added_values, self.substracted_values))

    def contains(self, node: Node):
        return any(c.contains(node) for c in itertools.chain(self.added_values, self.substracted_values))

    def __str__(self):
        text = '( '

        text += ' + '.join(map(str, self.added_values))

        if self.substracted_values:
            text += ' - ' if self.added_values else '- '

        text += ' - '.join(map(str, self.substracted_values))

        return text + ' )'

    def __init__(self, added_values: Sequence[Node] = None, substracted_values: Sequence[Node] = None):
        self.added_values = []
        self.substracted_values = []

        if added_values is not None:
            for val in added_values:
                if isinstance(val, AddAndSub):
                    self.added_values.extend(val.added_values)
                    self.substracted_values.extend(val.substracted_values)
                else:
                    self.added_values.append(val)

        if substracted_values is not None:
            for val in substracted_values:
                if isinstance(val, AddAndSub):
                    self.substracted_values.extend(val.added_values)
                    self.added_values.extend(val.substracted_values)
                else:
                    self.substracted_values.append(val)

    @classmethod
    def add(cls, *values: Node):
        return cls(values)

    @classmethod
    def sub(cls, *values: Node, positive_first_node: bool = True):
        if positive_first_node:
            return cls(added_values=values[:1], substracted_values=values[1:])
        else:
            return cls(substracted_values=values)

    def evaluate(self) -> float:
        result = 0
        for c in self.added_values:
            result += c.evaluate()
        for c in self.substracted_values:
            result -= c.evaluate()

        return result


# TODO: merge Mul and Div into a single object (see AddAndMul)
class Mul(BinOp):
    name = '*'

    @staticmethod
    def evaluator(*values: float) -> float:
        result = 1
        for v in values:
            result *= v
        return result


class Div(BinOp):
    name = '/'

    @staticmethod
    def evaluator(*values: float) -> float:
        result = 1
        for v in values:
            result *= v
        return result


class Pow(BinOp):
    name = '^'

    @staticmethod
    def evaluator(*values: float) -> float:
        result = 1.0
        for v in values[::-1]:
            result = v ** result
        return result


@dataclass
class MatchResult:
    wildcards: dict[str, Node] = field(default_factory=dict)
    just_added: Optional[str] = None
