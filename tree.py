from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable

Numerics = (float, int)

class Node(ABC):
    def __add__(self, other):
        if isinstance(other, Numerics):
            return self + Value(other)
        elif isinstance(other, Node):
            return Add(self, other)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Numerics):
            return self - Value(other)
        elif isinstance(other, Node):
            return Sub(self, other)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Numerics):
            return self * Value(other)
        elif isinstance(other, Node):
            return Mul(self, other)
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Numerics):
            return self / Value(other)
        elif isinstance(other, Node):
            return Div(self, other)
        else:
            return NotImplemented

    def __pow__(self, other):
        if isinstance(other, Numerics):
            return self ** Value(other)
        elif isinstance(other, Node):
            return Pow(self, other)
        else:
            return NotImplemented

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.children}"

    def __repr__(self) -> str:
        return self.__str__()

    def __init__(self, *children: Node):
        self.children = []
        t = type(self)
        for c in children:
            if type(c) is t:
                self.children.extend(c.children)
            else:
                self.children.append(c)

    def is_evaluatable(self) -> bool:
        return all(c.is_evaluatable() for c in self.children)

    @abstractmethod
    def evaluate(self) -> float:
        pass

    def matches(self, value: Node) -> bool:
        return type(value) == type(self) \
            and len(self.children) == len(value.children) \
            and all(sc.matches(vc) for sc, vc in zip(self.children, value.children))


class Leaf(Node, ABC):
    @abstractmethod
    def is_evaluatable(self) -> bool:
        pass

    def matches(self, value: Node) -> bool:
        return type(value) == type(self) and value.data == self.data

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.data if self.data is not None else ''})"


class BinOp(Node, ABC):
    evaluator: Callable[[float], float]
    name: str

    def evaluate(self) -> float:
        if not self.is_evaluatable():
            raise ValueError("cannot evaluate expression")

        return self.evaluator(*(v.data for v in self.children))

    def __str__(self):
        return '( ' + f' {self.name} '.join(map(lambda x: str(x), self.children)) + ' )'


class Value(Leaf):
    def is_evaluatable(self) -> bool:
        return True

    def evaluate(self) -> float:
        return self.data


class Variable(Leaf):
    def is_evaluatable(self) -> bool:
        return False

    def evaluate(self) -> float:
        raise TypeError("can't evaluate a variable")


class Wildcard(Leaf):
    def is_evaluatable(self) -> bool:
        return False

    def evaluate(self) -> float:
        raise TypeError("can't evaluate a wildcard")

    def __init__(self):
        super().__init__(None)

    def matches(self, value: Node) -> bool:
        return True


class Add(BinOp):
    name = "+"
    evaluator = sum

    def matches(self, value: Node) -> bool:
        if not isinstance(value, Add):
            return False

        contains_wildcard = any(isinstance(c, Wildcard) for c in self.children)

        self_children_count = len(self.children)
        value_children_count = len(value.children)

        if self_children_count == value_children_count:
            return any(all(sc.matches(vc) for sc, vc in zip(self.children, children))  # type: ignore
                       for children in itertools.permutations(value.children))

        if self_children_count < value_children_count and contains_wildcard:
            wildcard_capture_size = value_children_count - self_children_count

            for children in itertools.permutations(value.children):
                children = (Add(*children[:wildcard_capture_size]), *children[wildcard_capture_size:])
                if all(sc.matches(vc) for sc, vc in zip(self.children, children)):
                    return True

        return False


class Sub(BinOp):
    name = '-'

    @staticmethod
    def evaluator(*values: float):
        result = values[0]
        for v in values[1:]:
            result -= v
        return result


class Mul(BinOp):
    name = '*'

    def matches(self, value: Node) -> bool:
        raise NotImplementedError("Mul.matches")  # TODO: copy code from Add.matches() when polished enough

    @staticmethod
    def evaluator(*values: float):
        result = 1
        for v in values:
            result *= v
        return result


class Div(BinOp):
    name = '/'

    @staticmethod
    def evaluator(*values: float):
        result = 1
        for v in values:
            result *= v
        return result


class Pow(BinOp):
    name = '^'

    @staticmethod
    def evaluator(*values: float):
        result = 1
        for v in values[:-1]:
            result = v ** result
        return result
