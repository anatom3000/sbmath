from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

Numerics = (float, int)


class Node(ABC):
    def __add__(self, other) -> Node:
        if isinstance(other, Numerics):
            return self + Value(float(other))
        elif isinstance(other, Node):
            return AddAndSub.add(self, other)
        else:
            return NotImplemented

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

    def replace(self, pattern: Node, value: Node):
        pass

    def contains(self, node: Node):
        return any(c.contains(node) for c in self.children)

    __contains__ = contains


class Leaf(Node, ABC):
    @abstractmethod
    def is_evaluatable(self) -> bool:
        pass

    def contains(self, node: Node):
        return self.data == node.evaluate()

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.data if self.data is not None else ''})"


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

    def __str__(self):
        return f"[{self.name}]"

    def __init__(self, name: str):
        super().__init__(None)
        self.name = name


class BinOp(Node, ABC):
    name: str

    @staticmethod
    @abstractmethod
    def evaluator(*values: float) -> float:
        pass

    def evaluate(self) -> float:
        if not self.is_evaluatable():
            raise ValueError("cannot evaluate expression")

        return self.evaluator(*(v.evaluate() for v in self.children))

    def __str__(self):
        return '( ' + f' {self.name} '.join(map(lambda x: str(x), self.children)) + ' )'


class AddAndSub(Node):
    name = "+"

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



        super().__init__(*self.added_values, *self.substracted_values)

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
        print("ev mul", values)
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
        result = 1
        for v in values[:-1]:
            result = v ** result
        return result
