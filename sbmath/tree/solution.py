from __future__ import annotations

from sbmath.tree.core import Node, MatchResult, Variable


class Solution(Node):
    def __eq__(self, other: Node) -> bool:
        return isinstance(other, Approximation) and self.definition == other.definition

    def is_evaluable(self) -> bool:
        return self.approximation is not None

    def evaluate(self) -> Node:
        if self.is_evaluable():
            return self

        raise ValueError("cannot evaluate an approximation with no value")

    def approximate(self) -> float:
        return self.approximation

    # noinspection PyProtectedMember
    def _replace_identifiers(self, match_result: MatchResult) -> Node:
        return Solution(
            self.definition._replace_identifiers(match_result),
            self.variable
        )

    def _replace_in_children(self, old_pattern: Node, new_pattern: Node, evaluate: bool, reduce: bool) -> Node:
        return Solution(
            self.definition.replace(old_pattern, new_pattern, evaluate=evaluate, reduce=reduce),
            self.variable
        )

    def _substitute_in_children(self, pattern: Node, new: Node, evaluate: bool, reduce: bool) -> Node:
        return Solution(
            self.definition.substitute(pattern, new, evaluate=evaluate, reduce=reduce),
            self.variable
        )

    def _matches_no_reduce(self, value: Node, state: MatchResult, evaluate: bool, reduce: bool) \
            -> Optional[MatchResult]:
        if not isinstance(value, Solution):
            return None

        return self.definition.matches(value.definition, state, evaluate=evaluate, reduce=reduce)

    def matches(self, value: Node, state: MatchResult = None, *, evaluate: bool = True, reduce: bool = True) \
            -> Optional[MatchResult]:

        if state is None:
            state = MatchResult()

        new_state = self._match_no_reduce(value, copy.deepcopy(state), evaluate, reduce)
        if new_state is not None:
            return new_state

        reduced_self = self.reduce(evaluate=evaluate)
        reduced_value = value.reduce(evaluate=evaluate)

        return reduced_self._match_no_reduce(reduced_value, state, evaluate, reduce)

    def reduce(self, depth=-1, *, evaluate: bool = True) -> Solution:
        if depth == 0:
            return self

        return Solution(
            self.definition.reduce(depth - 1, evaluate=evaluate),
            self.variable,
            self.approximation
        )

    def contains(self, pattern: Node, *, evaluate: bool = True, reduce: bool = True) -> bool:
        return pattern.matches(self, evaluate=evaluate, reduce=reduce) is not None \
            or self.definition.contains(pattern, evaluate=evaluate, reduce=reduce)

    def __init__(self, definition: Equality, variable: Variable, approximation: float = None, inaccuracy: float = None):
        self.definition = definition.substitute(variable, Variable("{__internal_defined_value}"))
        self.variable = variable

        self.approximation = approximation
        self.inaccuracy = inaccuracy

    def __str__(self):
        str_self = f"({{__internal_defined_value}} | {self.definition}"
        if self.approximation is not None:
            str_self += f" | ~= {self.approximation}"
            if self.inaccuracy is not None:
                str_self += f" ~ {self.inaccuracy}"

        str_self += ')'
        return str_self.replace("{__internal_defined_value}", f"{self.variable}")