from __future__ import annotations

import copy

from sbmath.tree.core import Node, MatchResult


class Equality(Node):

    def __eq__(self, other):
        if not isinstance(other, Equality):
            return NotImplemented

        return self.left == other.left and self.right == other.right \
            or self.right == other.left and self.left == other.right

    def reduce(self, depth=-1, *, evaluate: bool = True) -> Node:
        if depth == 0:
            return self

        reduced_left = self.left.reduce(depth - 1, evaluate=evaluate)
        reduced_right = self.right.reduce(depth - 1, evaluate=evaluate)

        return type(self)(reduced_left, reduced_right)

    def _match_no_reduce(self, value: Node, state: MatchResult, evaluate: bool, reduce: bool) -> Optional[MatchResult]:
        if not isinstance(value, type(self)):
            return None

        left_match = self.left.matches(value.left, state, evaluate=evaluate, reduce=reduce)
        if left_match is None:
            return None

        right_match = self.right.matches(value.right, left_match, evaluate=evaluate, reduce=reduce)

        if right_match is None:
            return None

        return right_match

    # noinspection PyProtectedMember
    def matches(self, value: Node, state: MatchResult = None, *, evaluate: bool = True, reduce: bool = True) \
            -> Optional[MatchResult]:

        if state is None:
            state = MatchResult()

        no_reduce_state = self._match_no_reduce(value, copy.deepcopy(state), evaluate, reduce)

        if no_reduce_state and not (no_reduce_state.weak and reduce):
            return no_reduce_state

        no_reduce_state_swapped = Equality(self.right, self.left)._match_no_reduce(value, copy.deepcopy(state),
                                                                                   evaluate, reduce)

        if no_reduce_state_swapped and not (no_reduce_state_swapped.weak and reduce):
            return no_reduce_state_swapped

        if not reduce:
            return None

        reduced_self = self.reduce(evaluate=evaluate)
        reduced_value = value.reduce(evaluate=evaluate)

        reduced_state = reduced_self._match_no_reduce(reduced_value, state, evaluate, reduce)

        if reduced_state and not (reduced_state.weak and reduce):
            return reduced_state

        reduced_swapped_state = reduced_self._match_no_reduce(reduced_value, state, evaluate, reduce)

        if reduced_swapped_state:
            return reduced_swapped_state

        if reduced_state:
            return reduced_state

        if no_reduce_state_swapped:
            return no_reduce_state_swapped

        if no_reduce_state:
            return no_reduce_state

        return None

    # noinspection PyProtectedMember
    def _replace_identifiers(self, match_result: MatchResult) -> Node:
        return type(self)(self.left._replace_identifiers(match_result), self.right._replace_identifiers(match_result))

    def _replace_in_children(self, old_pattern: Node, new_pattern: Node, evaluate: bool, reduce: bool) -> Node:
        return type(self)(
            self.left.replace(old_pattern, new_pattern, evaluate=evaluate, reduce=reduce),
            self.right.replace(old_pattern, new_pattern, evaluate=evaluate, reduce=reduce)
        )

    def _substitute_in_children(self, pattern: Node, new: Node, evaluate: bool, reduce: bool) -> Node:
        return type(self)(
            self.left.substitute(pattern, new, evaluate=evaluate, reduce=reduce),
            self.right.substitute(pattern, new, evaluate=evaluate, reduce=reduce)
        )

    def _apply_on_children(self, pattern: Node, modifier: Callable[[MatchResult], Node], evaluate: bool, reduce: bool) -> Node:
        return type(self)(
            self.left.apply_on(pattern, modifier, evaluate=evaluate, reduce=reduce),
            self.right.apply_on(pattern, modifier, evaluate=evaluate, reduce=reduce)
        )

    def is_evaluable(self) -> bool:
        return False

    def evaluate(self) -> Node:
        raise TypeError("cannot evaluate an equality")

    def contains(self, pattern: Node, *, evaluate: bool = True, reduce: bool = True) -> bool:
        return pattern.matches(self, evaluate=evaluate, reduce=reduce) is not None \
            or self.left.contains(pattern, evaluate=evaluate, reduce=reduce) \
            or self.right.contains(pattern, evaluate=evaluate, reduce=reduce)

    @property
    def context(self) -> Optional[Context]:
        return self._context

    @context.setter
    def context(self, new: Optional[Context]):
        self._context = new

        self.left.context = new
        self.right.context = new

    def __init__(self, left: Node, right: Node):
        self.left = left
        self.right = right

        self._context = left.context

    def approximate(self) -> float:
        raise TypeError("cannot approximate an equality")

    def __str__(self):
        return f"{self.left} = {self.right}"
