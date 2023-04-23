from sbmath.tree import Node, MatchResult


class _FlagNode(Node):
    def __eq__(self, other: Node) -> bool:
        return isinstance(other, _FlagNode) and other._flag == self._flag

    def __init__(self, flag: str):
        self._flag = flag

    def is_evaluable(self) -> bool:
        return False

    def evaluate(self) -> Node:
        raise TypeError(f"cannot evaluate {flag}")

    def approximate(self) -> float:
        raise TypeError(f"cannot approximate {flag}")

    def _replace_identifiers(self, match_result: MatchResult) -> Node:
        return self

    def _replace_in_children(self, old_pattern: Node, new_pattern: Node, evaluate: bool, reduce: bool) -> Node:
        return self

    def _substitute_in_children(self, pattern: Node, new: Node, evaluate: bool, reduce: bool) -> Node:
        return self

    def reduce(self, depth=-1, *, evaluate: bool = True) -> Node:
        return self

    def __str__(self):
        return self._flag

    def contains(self, pattern: Node, *, evaluate: bool = True, reduce: bool = True) -> bool:
        return self == pattern

undefined = _FlagNode("undefined")
uncomputable = _FlagNode("uncomputable")
