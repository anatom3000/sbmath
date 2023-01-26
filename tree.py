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
        return self * -1

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

    def matches(self, value: Node, state: MatchResult = None) -> Optional[MatchResult]:
        print(f"matching {self} to {value} ; {state}")
        if state is None:
            state = MatchResult()

        if self.name in state.wildcards.keys() and not (value.matches == state.wildcards[self.name]):
            return None
        else:
            print(f"matching {self} to {value} ; {state}")
            state.wildcards[self.name] = value

        return state

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

    def matches(self, value: Node, state: MatchResult = None) -> Optional[MatchResult]:
        if state is None:
            state = MatchResult()

        if not isinstance(value, AddAndSub):
            if self.is_evaluable() and value.is_evaluable():
                return state if self.evaluate() == self.evaluate() else None

        new_state = copy.deepcopy(state)

        pattern_add_values = set(range(len(self. added_values)))
        value_add_values   = set(range(len(value.added_values)))
        pattern_sub_values = set(range(len(self. substracted_values)))
        value_sub_values   = set(range(len(value.substracted_values)))

        for vindex, vchild in enumerate(value.added_values):
            for pindex, pchild in filter(lambda x: not isinstance(x[1], Wildcard), enumerate(self.added_values)):
                if pindex not in pattern_add_values:
                    continue

                # print(f"[ADD-ADD] Trying to match {pchild=} with {vchild=}")
                result = pchild.matches(vchild, new_state)
                if result is not None:
                    pattern_add_values.remove(pindex)
                    value_add_values.remove(vindex)
                    new_state = result
                    break

        for vindex, vchild in enumerate(value.substracted_values):
            for pindex, pchild in filter(lambda x: not isinstance(x[1], Wildcard), enumerate(self.substracted_values)):
                if pindex not in pattern_sub_values:
                    continue
                # print(f"[SUB-SUB] Trying to match {pchild=} with {vchild=}")
                result = pchild.matches(vchild, new_state)
                if result is not None:
                    pattern_sub_values.remove(pindex)
                    value_sub_values.remove(vindex)
                    new_state = result
                    break

        for vindex in set(value_add_values):
            for pindex, pchild in filter(lambda x: not isinstance(x[1], Wildcard), enumerate(self.substracted_values)):
                if pindex not in pattern_sub_values:
                    continue
                # print(f"[ADD-SUB] Trying to match {pchild=} with {value.added_values[vindex]=}")
                result: Optional[MatchResult] = pchild.matches(-value.added_values[vindex], new_state)
                if result is not None:
                    pattern_sub_values.remove(pindex)
                    value_add_values.remove(vindex)
                    new_state = result
                    break

        for vindex in set(value_sub_values):
            for pindex, pchild in filter(lambda x: not isinstance(x[1], Wildcard), enumerate(self.added_values)):
                if pindex not in pattern_add_values:
                    continue
                # print(f"[SUB-ADD] Trying to match {pchild=} with {value.added_values[vindex]=}")
                result: Optional[MatchResult] = pchild.matches(-value.substracted_values[vindex], new_state)
                if result is not None:
                    pattern_add_values.remove(pindex)
                    value_sub_values.remove(vindex)
                    new_state = result
                    break

        # TODO: handle matching with wildcards

        if pattern_add_values or pattern_sub_values or value_add_values or value_sub_values:
            print(f"{pattern_add_values = }, {pattern_sub_values = }, {value_add_values = }, {value_sub_values = }")
            return None

        return new_state

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
