from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable


class Node(ABC):
    def __init__(self, *childs: Node):
        self.childs = childs

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.childs}"

    def __repr__(self) -> str:
        return self.__str__()

    def is_evaluatable(self):
        return all(c.is_evaluatable() for c in self.childs)

    @abstractmethod
    def evaluate(self) -> float:
        pass


class Leaf(Node, ABC):

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

        return self.evaluator(*(v.data for v in self.childs))

    def __str__(self):
        return '(' + f' {self.name} '.join(*self.childs) + ' )'


class Value(Leaf):
    def is_evaluatable(self):
        return True

    def evaluate(self) -> float:
        return self.data


class Variable(Leaf):
    def is_evaluatable(self):
        return False

    def evaluate(self) -> float:
        raise TypeError("can't evaluate a variable")


class Wildcard(Leaf):
    def is_evaluatable(self):
        return False

    def evaluate(self) -> float:
        raise TypeError("can't evaluate a wildcard")

    def __init__(self):
        super().__init__(None)


class Add(BinOp):
    name = "+"
    evaluator = sum


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
