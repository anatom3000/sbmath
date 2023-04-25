from __future__ import annotations

import copy
import itertools

from collections import defaultdict

# typing modules
from collections.abc import Iterable
from typing import Optional
from numbers import Real

# abstraction modules
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from sbmath.tree.context import MissingContextError, Context
from sbmath import _utils
from sbmath.computation import fraction

# debug function
from sbmath._utils import debug, inc_indent, dec_indent


class Node(ABC):
    context: Optional[Context] = None

    def __neg__(self):
        result = -1.0 * self
        result.context = self.context
        return result

    def __add__(self, other) -> Node:
        if isinstance(other, Real):
            other = Node.from_float(float(other))
            other.context = self.context
            return AddAndSub.add(self, context)
        elif isinstance(other, Node):
            result = AddAndSub.add(self, other)
            result.context = self.context
            return result
        else:
            return NotImplemented

    def __radd__(self, other) -> Node:
        if isinstance(other, Real):
            result = AddAndSub.add(Node.from_float(float(other)), self)
            result.context = self.context
            return result
        elif isinstance(other, Node):
            result = AddAndSub.add(other, self)
            result.context = other.context
            return result
        else:
            return NotImplemented

    def __sub__(self, other) -> Node:
        if isinstance(other, Real):
            result = AddAndSub.sub(self, Node.from_float(float(other)))
            result.context = self.context
            return result
        elif isinstance(other, Node):
            result = AddAndSub.sub(self, other)
            result.context = self.context
            return result
        else:
            return NotImplemented

    def __rsub__(self, other) -> Node:
        if isinstance(other, Real):
            result = AddAndSub.sub(Node.from_float(float(other)), self)
            result.context = self.context
            return result
        elif isinstance(other, Node):
            result = AddAndSub.sub(other, self)
            result.context = other.context
            return result
        else:
            return NotImplemented

    def __mul__(self, other) -> Node:
        if isinstance(other, Real):
            result = MulAndDiv.mul(self, Node.from_float(float(other)))
            result.context = self.context
            return result
        elif isinstance(other, Node):
            result = MulAndDiv.mul(self, other)
            result.context = self.context
            return result
        else:
            return NotImplemented

    def __rmul__(self, other) -> Node:
        if isinstance(other, Real):
            result = MulAndDiv.mul(Node.from_float(float(other)), self)
            result.context = self.context
            return result
        elif isinstance(other, Node):
            result = MulAndDiv.mul(other, self)
            result.context = other.context
            return result
        else:
            return NotImplemented

    def __truediv__(self, other) -> Node:
        if isinstance(other, Real):
            result = MulAndDiv.div(self, Node.from_float(float(other)))
            result.context = self.context
            return result
        elif isinstance(other, Node):
            result = MulAndDiv.div(self, other)
            result.context = self.context
            return result
        else:
            return NotImplemented

    def __rtruediv__(self, other) -> Node:
        if isinstance(other, Real):
            result = MulAndDiv.div(Node.from_float(float(other)), self)
            result.context = self.context
            return result
        elif isinstance(other, Node):
            result = MulAndDiv.div(other, self)
            result.context = other.context
            return result
        else:
            return NotImplemented

    def __pow__(self, other) -> Node:
        if isinstance(other, Real):
            result = Pow(self, Node.from_float(float(other)))
            result.context = self.context
            return result
        elif isinstance(other, Node):
            result = Pow(self, other)
            result.context = self.context
            return result
        else:
            return NotImplemented

    def __rpow__(self, other) -> Node:
        if isinstance(other, Real):
            result = Pow(Node.from_float(float(other)), self)
            result.context = self.context
            return result
        elif isinstance(other, Node):
            result = Pow(other, self)
            result.context = other.context
            return result
        else:
            return NotImplemented

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.__dict__})"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self):
        return hash(str(self))

    @classmethod
    def from_float(cls, x: float):
        num, denom = fraction.float_to_fraction(x)
        if denom == 1:
            return Value(num)

        return MulAndDiv.div(Value(num), Value(denom))

    @abstractmethod
    def __eq__(self, other: Node) -> bool:
        pass

    @abstractmethod
    def is_evaluable(self) -> bool:
        pass

    @abstractmethod
    def evaluate(self) -> Node:
        pass

    @abstractmethod
    def approximate(self) -> float:
        pass

    def matches(self, value: Node, state: MatchResult = None, *, evaluate: bool = True, reduce: bool = True) -> \
            Optional[MatchResult]:
        if state is None:
            state = MatchResult()

        return state if self == value else None

    @abstractmethod
    def _replace_identifiers(self, match_result: MatchResult) -> Node:
        pass

    def morph(self, old_pattern: Node, new_pattern: Node, *, evaluate: bool = True, reduce: bool = True) \
            -> Optional[Node]:
        m = old_pattern.matches(self, evaluate=evaluate, reduce=reduce)
        if m is None:
            return None

        try:
            new = new_pattern._replace_identifiers(m)
        except ReplacingError:
            return None
        return new

    def replace(self, old_pattern: Node, new_pattern: Node, *, evaluate: bool = True, reduce: bool = True) -> Node:
        m = old_pattern.matches(self, evaluate=evaluate, reduce=reduce)
        if m:
            new = new_pattern._replace_identifiers(m)
        else:
            new = self

        return new._replace_in_children(old_pattern, new_pattern, evaluate, reduce)

    @abstractmethod
    def _replace_in_children(self, old_pattern: Node, new_pattern: Node, evaluate: bool, reduce: bool) -> Node:
        pass

    def substitute(self, pattern: Node, new: Node, *, evaluate: bool = True, reduce: bool = True) -> Node:
        m = pattern.matches(self, evaluate=evaluate, reduce=reduce)
        if m:
            return new._replace_identifiers(m)
        else:
            return self._substitute_in_children(pattern, new, evaluate, reduce)

    @abstractmethod
    def _substitute_in_children(self, pattern: Node, new: Node, evaluate: bool, reduce: bool) -> Node:
        pass

    @abstractmethod
    def reduce(self, depth=-1, *, evaluate: bool = True) -> Node:
        pass

    @abstractmethod
    def contains(self, pattern: Node, *, evaluate: bool = True, reduce: bool = True) -> bool:
        pass


class Leaf(Node, ABC):

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.data == other.data

        return NotImplemented

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return str(self.data)

    def reduce(self, depth=-1, *, evaluate: bool = True) -> Node:
        return self

    def contains(self, pattern: Node, *, evaluate: bool = True, reduce: bool = True) -> bool:
        return pattern.matches(self, evaluate=evaluate, reduce=reduce) is not None

    def _replace_identifiers(self, match_result: MatchResult) -> Node:
        return self

    def _replace_in_children(self, old_pattern: Node, new_pattern: Node, evaluate: bool, reduce: bool = True) -> Node:
        return self

    def _substitute_in_children(self, pattern: Node, new: Node, evaluate: bool, reduce: bool = True) -> Node:
        return self

    def __init__(self, data):
        self.data = data


class AdvancedBinOp(Node, ABC):
    base_operation_symbol: str
    inverse_operation_symbol: str
    constant_term_last: bool

    identity: Node
    absorbing_element: Optional[Node] = None

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
    def _should_invert_value(value: Node) -> bool:
        pass

    def reduce(self, depth=-1, *, evaluate: bool = True) -> Node:
        if depth == 0:
            return self

        inc_indent()

        if evaluate:
            eval_part = type(self)(
                filter(lambda x: x.is_evaluable(), self.base_values),
                filter(lambda x: x.is_evaluable(), self.inverted_values)
            )

            non_eval_part = type(self)(
                filter(lambda x: not x.is_evaluable(), self.base_values),
                filter(lambda x: not x.is_evaluable(), self.inverted_values)
            )

            inc_indent()
            debug(f"{eval_part = }", flag='reduce')
            debug(f"{non_eval_part = }", flag='reduce')

            eval_result = eval_part.evaluate()

            debug(f"{eval_result = }", flag='reduce')
            dec_indent()

            if isinstance(eval_result, type(self)):
                non_eval_part.base_values += eval_result.base_values
                non_eval_part.inverted_values += eval_result.inverted_values
            else:
                if self.constant_term_last:
                    non_eval_part.base_values.append(eval_result)
                else:
                    non_eval_part.base_values.insert(0, eval_result)
        else:
            non_eval_part = self

        debug(f"{non_eval_part = }", flag='reduce')

        value_occurences: defaultdict[Node, int] = defaultdict(int)

        for occurence in non_eval_part.base_values:
            reduced = occurence.reduce(depth - 1, evaluate=evaluate)
            if isinstance(reduced, type(non_eval_part)):
                v: Node
                for v in reduced.base_values:
                    value_occurences[v] += 1
                v: Node
                for v in reduced.inverted_values:
                    value_occurences[v] -= 1
            else:
                value_occurences[reduced] += 1

        debug(f"base no inverse {value_occurences = }", flag='reduce')

        for occurence in non_eval_part.inverted_values:
            reduced = occurence.reduce(depth - 1, evaluate=evaluate)

            if isinstance(reduced, type(non_eval_part)):
                for v in reduced.base_values:
                    value_occurences[v] -= 1
                for v in reduced.inverted_values:
                    value_occurences[v] += 1
            else:
                value_occurences[reduced] -= 1

        debug(f"final {value_occurences = }", flag='reduce')

        base_values = []
        inverted_values = []
        remaining_evaluable = type(self)((), ())

        for value, occurence in value_occurences.items():
            if occurence == 0:
                continue

            if value == self.identity:
                continue

            if value == self.absorbing_element:
                dec_indent()
                return self.absorbing_element

            if evaluate and value.is_evaluable():
                remaining_evaluable.base_values.append(self._repeat_value(value, occurence))
                continue

            # noinspection PyTypeChecker
            if self._should_invert_value(value):
                # noinspection PyTypeChecker
                value = self._invert_value(value)
                occurence *= -1

            if occurence == 1:
                base_values.append(value)
            elif occurence == -1:
                inverted_values.append(value)
            elif occurence > 1:
                base_values.append(self._repeat_value(value, occurence))
            else:
                inverted_values.append(self._repeat_value(self._invert_value(value), occurence))

        debug(f"before final evaluate: {base_values=}, {inverted_values=}, {remaining_evaluable=}", flag='reduce')

        if evaluate:
            evaluated = remaining_evaluable.evaluate()
            if evaluated == self.absorbing_element:
                return self.absorbing_element

            if evaluated != self.identity:
                if isinstance(evaluated, type(self)):
                    base_values.extend(val for val in evaluated.base_values if val != self.identity)
                    inverted_values.extend(val for val in evaluated.inverted_values if val != self.identity)
                elif self._should_invert_value(evaluated):
                    inverted_values.append(self._invert_value(evaluated))
                else:
                    base_values.append(evaluated)

        dec_indent()

        if len(base_values) == 0:
            if len(inverted_values) == 0:
                return self.identity
            elif len(inverted_values) == 1:
                return self._invert_value(inverted_values[0])
        elif len(base_values) == 1:
            if len(inverted_values) == 0:
                return base_values[0]

        return type(self)(base_values, inverted_values)

    # --- MATCH INTERNAL FUNCTIONS ---
    def _match_evaluable(self, value: Node, state: MatchResult = None) -> Optional[MatchResult]:
        return state if self.evaluate() == value.evaluate() else None

    def _match_no_wildcards(self, value: AdvancedBinOp, state: MatchResult, evaluate: bool, reduce: bool) \
            -> (MatchResult, AdvancedBinOp, AdvancedBinOp):
        remaining_value = type(self)()
        remaining_pattern = copy.deepcopy(self)
        new_state = copy.deepcopy(state)

        index: int = 0  # for linters
        for val_value in value.base_values:
            found = False
            for index, pat_value in enumerate(remaining_pattern.base_values):
                match_result = pat_value.matches(val_value, new_state, evaluate=evaluate, reduce=reduce)
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
                    match_result = pat_value.matches(self._invert_value(val_value), new_state, evaluate=evaluate,
                                                     reduce=reduce)
                    if match_result is not None:
                        new_state = match_result
                        found = True
                        state.weak = True
                        break
                if found:
                    del remaining_pattern.inverted_values[index]
                else:
                    remaining_value.base_values.append(val_value)

        for val_value in value.inverted_values:
            found = False
            for index, pat_value in enumerate(remaining_pattern.inverted_values):
                match_result = pat_value.matches(val_value, new_state, evaluate=evaluate, reduce=reduce)
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
                    match_result = pat_value.matches(self._invert_value(val_value), new_state, evaluate=evaluate,
                                                     reduce=reduce)
                    if match_result is not None:
                        new_state = match_result
                        found = True
                        state.weak = True
                        break
                if found:
                    del remaining_pattern.base_values[index]
                else:
                    remaining_value.inverted_values.append(val_value)

        return new_state, remaining_pattern, remaining_value

    def _remove_wildcard_match(self, value: Node, value_index: int, wildcard: Wildcard,
                               base_match_table: _utils.BiMultiDict, inverted_match_table: _utils.BiMultiDict,
                               state: MatchResult, evaluate: bool, reduce: bool, inverted: bool) \
            -> Optional[_utils.BiMultiDict, _utils.BiMultiDict, MatchResult]:

        debug(f"=> Start remove wildcard match:", flag='match_adv_wc')
        inc_indent()
        debug(f"{value = }", flag='match_adv_wc')
        debug(f"{wildcard = }", flag='match_adv_wc')
        debug(f"{base_match_table = }", flag='match_adv_wc')
        # debug(f"{inverted_match_table = }", flag='match_adv_wc')
        debug(f"{state = }", flag='match_adv_wc')
        # debug(f"{inverted = }", flag='match_adv_wc')
        dec_indent()

        base_match_table = copy.deepcopy(base_match_table)
        inverted_match_table = copy.deepcopy(inverted_match_table)

        base_similars = [(index, similar) for index, similar in base_match_table.get_from_value(wildcard)
                         if len(list(base_match_table.get_from_key((index, similar)))) == 1]

        inverted_similars = [(index, similar) for index, similar in inverted_match_table.get_from_value(wildcard)
                             if len(list(inverted_match_table.get_from_key((index, similar)))) == 1]

        # debug(f"Similars:", flag='match_adv_wc')
        # inc_indent()
        # debug(f"{base_similars = }", flag='match_adv_wc')
        # debug(f"{inverted_similars = }", flag='match_adv_wc')
        #
        # dec_indent()
        # debug(f"After copy:", flag='match_adv_wc')
        # inc_indent()
        #
        # debug(f"{base_match_table = }", flag='match_adv_wc')
        # debug(f"{inverted_match_table = }", flag='match_adv_wc')
        #
        # dec_indent()

        if len(base_similars) == 0 and len(inverted_similars) == 0:
            debug(f"[A]", flag='match_adv_wc')
            debug(f"0) {state = }", flag='match_adv_wc')
            state = wildcard.matches(value, state, evaluate=evaluate, reduce=reduce)
            if state is None:
                return None

            debug(f"1) {state = }", flag='match_adv_wc')

            if inverted:
                inverted_match_table.remove_key((value_index, value))
            else:
                base_match_table.remove_key((value_index, value))

            debug(f"2) {state = }", flag='match_adv_wc')

            base_match_table.try_remove_value(wildcard)
            inverted_match_table.try_remove_value(wildcard)

            debug(f"3) {state = }", flag='match_adv_wc')

            return base_match_table, inverted_match_table, state

        if (not inverted) and len(base_similars) == 1 and len(inverted_similars) == 0:
            debug(f"[B]", flag='match_adv_wc')

            debug(f"0) {state = }", flag='match_adv_wc')
            state = wildcard.matches(base_similars[0][1], state, evaluate=evaluate, reduce=reduce)
            if state is None:
                return None

            debug(f"1) {state = }", flag='match_adv_wc')

            base_match_table.remove_key(base_similars[0])
            base_match_table.remove_value(wildcard)
            inverted_match_table.try_remove_value(wildcard)

            debug(f"2) {state = }", flag='match_adv_wc')

            return base_match_table, inverted_match_table, state

        if inverted and len(inverted_similars) == 1 and len(base_similars) == 0:
            debug(f"[C]", flag='match_adv_wc')
            debug(f"0) {state = }", flag='match_adv_wc')

            state = wildcard.matches(inverted_similars[0][1], state, evaluate=evaluate, reduce=reduce)
            if state is None:
                return None

            debug(f"1) {state = }", flag='match_adv_wc')

            inverted_match_table.remove_key(inverted_similars[0])
            inverted_match_table.remove_value(wildcard)
            base_match_table.try_remove_value(wildcard)

            debug(f"2) {state = }", flag='match_adv_wc')

            return base_match_table, inverted_match_table, state

        if not inverted:
            combined_values = type(self)((v for _, v in base_similars), (v for _, v in inverted_similars))
        else:
            combined_values = type(self)((v for _, v in inverted_similars), (v for _, v in base_similars))

        debug(f"{combined_values = }", flag='match_adv_wc')
        debug(f"0) {state = }", flag='match_adv_wc')

        state = wildcard.matches(combined_values, state, evaluate=evaluate, reduce=reduce)
        if state is None:
            return None

        debug(f"1) {state = }", flag='match_adv_wc')

        for s in base_similars:
            base_match_table.remove_key(s)

        for s in inverted_similars:
            inverted_match_table.remove_key(s)

        base_match_table.try_remove_value(wildcard)
        inverted_match_table.try_remove_value(wildcard)

        return base_match_table, inverted_match_table, state

    def _clean_up_single_wildcards(self, base_match_table: _utils.BiMultiDict, inverted_match_table: _utils.BiMultiDict,
                                   state: MatchResult, evaluate: bool, reduce: bool) \
            -> Optional[(_utils.BiMultiDict, _utils.BiMultiDict, MatchResult)]:

        base_match_table = copy.deepcopy(base_match_table)
        inverted_match_table = copy.deepcopy(inverted_match_table)

        while any(len(base_match_table.get_from_key(k)) < 2 for k in base_match_table.keys()):
            debug(f"{[(k, base_match_table.get_from_key(k)) for k in base_match_table.keys()] = }", flag='match_adv_wc')
            for index, value in list(base_match_table.keys()):
                if (index, value) not in base_match_table.keys():
                    continue

                matches = list(base_match_table.get_from_key((index, value)))

                if len(matches) == 0:
                    return None
                elif len(matches) == 1:
                    r = self._remove_wildcard_match(value, index, matches[0], base_match_table, inverted_match_table,
                                                    state, evaluate, reduce,
                                                    inverted=False)
                    if r is None:
                        return None
                    base_match_table, inverted_match_table, state = r

        while any(len(inverted_match_table.get_from_key(k)) < 2 for k in inverted_match_table.keys()):
            for index, value in list(inverted_match_table.keys()):
                if (index, value) not in inverted_match_table.keys():
                    continue

                matches = list(inverted_match_table.get_from_key((index, value)))

                if len(matches) == 0:
                    return None
                elif len(matches) == 1:
                    r = self._remove_wildcard_match(value, index, matches[0], base_match_table, inverted_match_table,
                                                    state, evaluate, reduce,
                                                    inverted=True)
                    if r is None:
                        return None
                    base_match_table, inverted_match_table, state = r

        return base_match_table, inverted_match_table, state

    def _build_match_tables(self, value: AdvancedBinOp, state: MatchResult, evaluate: bool, reduce: bool) \
            -> Optional[tuple[_utils.BiMultiDict, _utils.BiMultiDict]]:

        remaining_wildcards = set(self.base_values + self.inverted_values)

        base_match_table = _utils.BiMultiDict()
        for index, val in enumerate(value.base_values):
            found_one = False
            for wildcard in self.base_values:
                r = wildcard.matches(val, copy.deepcopy(state), evaluate=evaluate, reduce=reduce)
                if r:
                    found_one = True
                    base_match_table.add((index, val), wildcard)
                    remaining_wildcards.discard(wildcard)

            if not found_one:
                for wildcard in self.inverted_values:
                    invval = self._invert_value(val)
                    r = wildcard.matches(invval, copy.deepcopy(state), evaluate=evaluate, reduce=reduce)
                    if r:
                        found_one = True
                        state.weak = True
                        base_match_table.add((-index, invval), wildcard)
                        remaining_wildcards.discard(wildcard)

            if not found_one:
                return None

        inverted_match_table = _utils.BiMultiDict()
        for index, val in enumerate(value.inverted_values):
            found_one = False
            for wildcard in self.inverted_values:
                r = wildcard.matches(val, copy.deepcopy(state), evaluate=evaluate, reduce=reduce)
                if r:
                    found_one = True
                    inverted_match_table.add((index, val), wildcard)
                    remaining_wildcards.discard(wildcard)

            if not found_one:
                for wildcard in self.base_values:
                    invval = self._invert_value(val)
                    r = wildcard.matches(invval, copy.deepcopy(state), evaluate=evaluate, reduce=reduce)
                    if r:
                        found_one = True
                        state.weak = True
                        inverted_match_table.add((-index, invval), wildcard)
                        remaining_wildcards.discard(wildcard)

            if not found_one:
                return None

        if len(remaining_wildcards) != 0:
            debug(f"{remaining_wildcards = }", flag='match_adv_wc')
            return None

        return base_match_table, inverted_match_table

    def _match_wildcards(self, value: AdvancedBinOp, state: MatchResult, evaluate: bool, reduce: bool) \
            -> Optional[MatchResult]:

        r = self._build_match_tables(value, state, evaluate, reduce)
        if r is None:
            debug(f"No match found, aborting...", flag='match_adv_wc')
            return None

        base_match_table, inverted_match_table = r

        debug(f"{base_match_table = }", flag='match_adv_wc')
        debug(f"{inverted_match_table = }", flag='match_adv_wc')

        result = self._clean_up_single_wildcards(base_match_table, inverted_match_table, state, evaluate, reduce)
        if result is None:
            debug(f"Something messed up while cleaning up single wildcards, aborting...", flag='match_adv_wc')
            return None

        base_match_table, inverted_match_table, state = result

        debug(f"Single wc clean up successful", flag='match_adv_wc')
        debug(f"{base_match_table = }", flag='match_adv_wc')
        debug(f"{inverted_match_table = }", flag='match_adv_wc')
        debug(f"{state = }", flag='match_adv_wc')

        debug(f"Starting to match base values", flag='match_adv_wc')
        inc_indent()
        for index, value in list(base_match_table.keys()):
            if value not in (v for _, v in base_match_table.keys()):
                debug(f"{value} was not found in {list(v for _, v in base_match_table.keys()) = }, skipping...",
                      flag='match_adv_wc')
                continue

            debug(f"====", flag='match_adv_wc')
            debug(f"attempting removal of wildcard match {value=} ", flag='match_adv_wc')
            inc_indent()
            debug(f"{base_match_table.get_from_key((index, value))[0] = }", flag='match_adv_wc')
            debug(f"{base_match_table = }", flag='match_adv_wc')
            debug(f"{inverted_match_table = }", flag='match_adv_wc')
            debug(f"{state = }", flag='match_adv_wc')

            r = self._remove_wildcard_match(value, index, base_match_table.get_from_key((index, value))[0],
                                            base_match_table,
                                            inverted_match_table, state, evaluate, reduce, False)
            dec_indent()
            if r is None:
                debug(f"Something went wrong while removing wildcard match, aborting...", flag='match_adv_wc')
                return None
            base_match_table, inverted_match_table, state = r
        dec_indent()

        debug(f"Starting to match inverted values", flag='match_adv_wc')
        inc_indent()
        for index, value in list(inverted_match_table.keys()):
            if value not in (v for _, v in inverted_match_table.keys()):
                debug(f"{value} was not found in {list(v for _, v in inverted_match_table.keys()) = }, skipping...",
                      flag='match_adv_wc')
                continue
            debug(f"attempting removal of wildcard match {value=} ", flag='match_adv_wc')
            inc_indent()
            debug(f"{inverted_match_table.get_from_key((index, value))[0] = }", flag='match_adv_wc')
            debug(f"{base_match_table = }", flag='match_adv_wc')
            debug(f"{inverted_match_table = }", flag='match_adv_wc')
            debug(f"{state = }", flag='match_adv_wc')
            inc_indent()
            r = self._remove_wildcard_match(value, index, inverted_match_table.get_from_key((index, value))[0],
                                            base_match_table,
                                            inverted_match_table, state, evaluate, reduce, True)

            dec_indent()
            dec_indent()
            if r is None:
                debug(f"Something went wrong while removing wildcard match, aborting...", flag='match_adv_wc')
                return None
            base_match_table, inverted_match_table, state = r
        dec_indent()

        debug(f"{base_match_table = }, {inverted_match_table = }", flag='match_adv_wc')

        if len(list(base_match_table.values())) != 0 or len(list(inverted_match_table.values())) != 0:
            return None

        return state

    def _match_no_reduce(self, value: Node, state: MatchResult, evaluate: bool, reduce: bool):
        if evaluate and self.is_evaluable() and value.is_evaluable():
            debug(f"both self and value are evaluable, matching direcly...", flag='match')
            return self._match_evaluable(value, state)

        if not isinstance(value, type(self)):
            debug(f"type of value is different than self, aborting...", flag='match')
            return None

        no_wildcard_self = type(self)(
            list(filter(lambda x: (not x.contains(_wildcard_wildcard)) and self.identity.matches(x, evaluate=evaluate,
                                                                                                 reduce=reduce) is None,
                        self.base_values)),
            list(filter(lambda x: (not x.contains(_wildcard_wildcard)) and self.identity.matches(x, evaluate=evaluate,
                                                                                                 reduce=reduce) is None,
                        self.inverted_values))
        )

        debug(f"{no_wildcard_self = }", flag='match')

        value: AdvancedBinOp  # I love Python's type system...

        state, remaining_pattern, remaining_value = no_wildcard_self._match_no_wildcards(value, state, evaluate, reduce)

        debug(f"Finished matching everything except wildcards:", flag='match')
        inc_indent()
        debug(f"{state = }", flag='match')
        debug(f"{remaining_pattern = }", flag='match')
        debug(f"{remaining_value = }", flag='match')
        dec_indent()

        if remaining_pattern.base_values or remaining_pattern.inverted_values:
            debug("Some non-wildcard did not get a corresponding value, aborting...", flag='match')
            return None

        wildcard_self = type(self)(
            list(filter(lambda x: x.contains(_wildcard_wildcard), self.base_values)),
            list(filter(lambda x: x.contains(_wildcard_wildcard), self.inverted_values))
        )

        debug(f"{wildcard_self = }", flag='match')

        result = wildcard_self._match_wildcards(remaining_value, state, evaluate, reduce)  # type: ignore

        debug(f"Finishing match, returning {result}", flag='match')

        return result

    # --- END MATCH INTERNAL FUNCTIONS ---

    def matches(self, value: Node, state: MatchResult = None, *, evaluate: bool = True, reduce: bool = True) -> \
            Optional[MatchResult]:

        debug(f"Matching pattern {self} with value {value}", flag='match')
        inc_indent()

        if state is None:
            state = MatchResult()

        no_reduce_state = self._match_no_reduce(value, copy.deepcopy(state), evaluate, reduce)

        if no_reduce_state is not None and not (no_reduce_state.weak and reduce):
            dec_indent()
            return no_reduce_state

        if not reduce:
            return None

        debug(f"Matching without reducing failed, reducing...", flag='match')

        reduced_self = self.reduce(evaluate=evaluate)
        reduced_value = value.reduce(evaluate=evaluate)

        debug(f"{reduced_self = }", flag='match')
        debug(f"{reduced_value = }", flag='match')

        if not isinstance(reduced_self, AdvancedBinOp):
            debug(f"reduced self is no longer an advanced binary operator, redirecting...", flag='match')
            m = reduced_self.matches(reduced_value, state, evaluate=evaluate, reduce=reduce)
            dec_indent()
            return m

        m = reduced_self._match_no_reduce(reduced_value, state, evaluate, reduce)
        if m is None and no_reduce_state:
            dec_indent()
            return no_reduce_state

        dec_indent()
        return m

    def _replace_identifiers(self, match_result: MatchResult) -> Node:
        return type(self)(
            base_values=(x._replace_identifiers(match_result) for x in self.base_values),
            inverted_values=(x._replace_identifiers(match_result) for x in self.inverted_values)
        )

    def _replace_in_children(self, old_pattern: Node, new_pattern: Node, evaluate: bool, reduce: bool) -> Node:
        return type(self)(
            base_values=(x.replace(old_pattern, new_pattern, evaluate=evaluate, reduce=reduce) for x in
                         self.base_values),
            inverted_values=(x.replace(old_pattern, new_pattern, evaluate=evaluate, reduce=reduce) for x in
                             self.inverted_values)
        )

    def _substitute_in_children(self, pattern: Node, new: Node, evaluate: bool, reduce: bool) -> Node:
        return type(self)(
            base_values=(x.substitute(pattern, new, evaluate=evaluate, reduce=reduce) for x in self.base_values),
            inverted_values=(x.substitute(pattern, new, evaluate=evaluate, reduce=reduce) for x in self.inverted_values)
        )

    def is_evaluable(self) -> bool:
        return all(c.is_evaluable() for c in itertools.chain(self.base_values, self.inverted_values))

    def contains(self, pattern: Node, *, evaluate: bool = True, reduce: bool = True) -> bool:
        return pattern.matches(self, evaluate=evaluate, reduce=reduce) is not None or any(
            c.contains(pattern, evaluate=evaluate, reduce=reduce) for c in
            itertools.chain(self.base_values, self.inverted_values))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        for value in self.base_values:
            if value not in other.base_values and self._invert_value(value) not in other.inverted_values:
                return False

        for value in self.inverted_values:
            if value not in other.inverted_values and self._invert_value(value) not in other.base_values:
                return False

        return True

    @staticmethod
    @abstractmethod
    def _put_parentheses(value: Node) -> bool:
        pass

    def __str__(self):

        text = f' {self.base_operation_symbol} '.join(
            f"({value})" if self._put_parentheses(value)
            else f"{value}"
            for value in self.base_values)

        if self.inverted_values:
            if not self.base_values:
                text += f"{self.identity}"

            text += f' {self.inverse_operation_symbol} ' if self.base_values else f'{self.inverse_operation_symbol} '

        text += f' {self.inverse_operation_symbol} '.join(
            f"({value})" if self._put_parentheses(value)
            else f"{value}"
            for value in self.inverted_values)

        return text

    def __hash__(self):
        return hash(str(self))

    @property
    def context(self) -> Optional[Context]:
        return self._context

    @context.setter
    def context(self, new: Optional[Context]):
        self._context = new

        for value in self.base_values:
            value.context = new
        for value in self.inverted_values:
            value.context = new

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

        if self.base_values:
            self._context = self.base_values[0].context
            return
        if self.inverted_values:
            self._context = self.inverted_values[0].context
            return


class BinOp(Node, ABC):
    name: str

    right_identity: Node

    left_absorbing_element: Optional[Node] = None
    left_absorbing_result: Optional[Node] = None

    right_absorbing_element: Optional[Node] = None
    right_absorbing_result: Optional[Node] = None

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return self.left == other.left and self.right == other.right

    def reduce(self, depth=-1, *, evaluate: bool = True) -> Node:
        if depth == 0:
            return self

        reduced_left = self.left.reduce(depth - 1, evaluate=evaluate)
        reduced_right = self.right.reduce(depth - 1, evaluate=evaluate)

        if evaluate:
            if reduced_right.is_evaluable():
                if reduced_right.evaluate() == self.right_absorbing_element:
                    return self.right_absorbing_result

                if reduced_left.is_evaluable():
                    if reduced_left.evaluate() == self.left_absorbing_element:
                        return self.left_absorbing_result

                    return type(self)(reduced_left, reduced_right).evaluate()

                if reduced_right.evaluate() == self.right_identity:
                    return reduced_left
        else:
            if isinstance(reduced_left, Value) and isinstance(reduced_right, Value):
                return Node.from_float(self.approximator(reduced_left.data, reduced_right.data))

            if reduced_right == self.right_identity:
                return reduced_left

        return type(self)(reduced_left, reduced_right)

    def _match_no_reduce(self, value: Node, state: MatchResult, evaluate: bool, reduce: bool) -> Optional[MatchResult]:

        if evaluate and self.is_evaluable() and value.is_evaluable():
            return state if reduced_self.evaluate() == reduced_value.evaluate() else None

        if not isinstance(value, type(self)):
            return None

        left_match = self.left.matches(value.left, state, evaluate=evaluate, reduce=reduce)
        if left_match is None:
            return None

        right_match = self.right.matches(value.right, left_match, evaluate=evaluate, reduce=reduce)

        if right_match is None:
            return None

        return right_match

    def matches(self, value: Node, state: MatchResult = None, *, evaluate: bool = True, reduce: bool = True) -> \
            Optional[MatchResult]:
        if state is None:
            state = MatchResult()

        new_state = self._match_no_reduce(value, copy.deepcopy(state), evaluate, reduce)
        if new_state is not None:
            return new_state

        reduced_self = self.reduce(evaluate=evaluate)
        reduced_value = value.reduce(evaluate=evaluate)

        if not isinstance(reduced_self, type(self)):
            return reduced_self.matches(reduced_value, state, evaluate=evaluate, reduce=reduce)

        return reduced_self._match_no_reduce(reduced_value, state, evaluate, reduce)

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

    def is_evaluable(self) -> bool:
        return self.left.is_evaluable() and self.right.is_evaluable()

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

    @staticmethod
    @abstractmethod
    def approximator(left: float, right: float) -> float:
        pass

    def approximate(self) -> float:
        return self.approximator(self.left.approximate(), self.right.approximate())

    @staticmethod
    @abstractmethod
    def _put_parentheses(value: Node) -> bool:
        pass

    def __str__(self):
        left = f"({self.left})" if self._put_parentheses(self.left) else f"{self.left}"
        right = f"({self.right})" if self._put_parentheses(self.right) else f"{self.right}"

        return f"{left} {self.name} {right}"

    def __hash__(self):
        return hash(str(self))


class Value(Leaf):

    def __init__(self, data: int):
        super().__init__(int(data))

    def __neg__(self):
        return Value(-self.data)

    def __str__(self):
        return str(self.data)

    def is_evaluable(self) -> bool:
        return True

    def evaluate(self) -> Node:
        return self

    def approximate(self) -> float:
        return float(self.data)


class Variable(Leaf):
    def is_evaluable(self) -> bool:
        return self.context is not None and (
                    self.data in self.context.variables.keys() or self.data in self.context.constants.keys())

    def evaluate(self) -> Node:
        if self.context is None:
            raise MissingContextError(f"can't evaluate variable '{self.data}' without context")

        is_constant = self.data not in self.context.constants.keys()
        is_variable = self.data not in self.context.variables.keys()

        if not is_constant and not is_variable:
            raise MissingContextError(f"context exists but variable '{self.data}' is undefined")

        if is_variable:
            return self.context.variables[self.data].evaluate()

        return self

    def reduce(self, depth=-1, *, evaluate: bool = True) -> Node:
        return self

    def approximate(self) -> float:
        if self.context is None:
            raise MissingContextError(f"can't approximate variable '{self.data}' without context")

        is_constant = self.data not in self.context.constants.keys()
        is_variable = self.data not in self.context.variables.keys()

        if not is_constant and not is_variable:
            raise MissingContextError(f"context exists but variable '{self.data}' is undefined")

        if is_constant:
            return self.context.constants[self.data].evaluate()

        return self.context.variables[self.data].evaluate()


class Wildcard(Node):
    def __eq__(self, other):
        if not isinstance(other, Wildcard):
            return NotImplemented

        return self.name == other.name and self.constraints == other.constraints

    def __hash__(self):
        return hash(str(self))

    def contains(self, pattern: Node, *, evaluate: bool = True, reduce: bool = True) -> bool:
        return pattern.matches(self, evaluate=evaluate, reduce=reduce) is not None

    def _match_contraints(self, value: Node, evaluate: bool, reduce: bool) -> bool:

        if "eval" in self.constraints.keys():
            if self.constraints["eval"] == Value(1) and not value.is_evaluable():
                return False
            if self.constraints["eval"] == Value(0) and value.is_evaluable():
                return False

        if "constant_with" in self.constraints.keys():
            if value.contains(self.constraints["constant_with"], evaluate=evaluate, reduce=reduce):
                return False

        if "variable_with" in self.constraints.keys():
            if not value.contains(self.constraints["variable_with"], evaluate=evaluate, reduce=reduce):
                return False

        if "wildcard" in self.constraints.keys():
            if self.constraints["wildcard"] == Value(1) and not isinstance(value, Wildcard):
                return False
            if self.constraints["wildcard"] == Value(0) and isinstance(value, Wildcard):
                return False

        if "explicit_value" in self.constraints.keys():
            if self.constraints["explicit_value"] == Value(1) and not isinstance(value, Value):
                return False
            if self.constraints["explicit_value"] == Value(0) and isinstance(value, Value):
                return False

        return True

    def matches(self, value: Node, state: MatchResult = None, *, evaluate: bool = True, reduce: bool = True) -> \
            Optional[MatchResult]:
        if state is None:
            state = MatchResult()

        if self.name == '_':
            return state if self._match_contraints(value, evaluate, reduce) else None

        if self.name in state.wildcards.keys() and value.matches(state.wildcards[self.name], evaluate=evaluate,
                                                                 reduce=reduce) is None:
            return None

        if self._match_contraints(value, evaluate, reduce):
            state.wildcards[self.name] = value
            return state

        return None

    def _replace_identifiers(self, match_result: MatchResult) -> Node:
        if self.name not in match_result.wildcards:
            return self

        return match_result.wildcards[self.name]

    def _replace_in_children(self, old_pattern: Node, new_pattern: Node, evaluate: bool, reduce: bool = True) -> Node:
        return self

    def _substitute_in_children(self, pattern: Node, new: Node, evaluate: bool, reduce: bool) -> Node:
        return self

    def is_evaluable(self) -> bool:
        return False

    def evaluate(self) -> Node:
        raise TypeError(f"can't evaluate {type(self)}")

    def approximate(self) -> float:
        raise TypeError(f"can't approximate {type(self)}")

    def reduce(self, depth=-1, *, evaluate: bool = True) -> Node:
        return self

    def __str__(self):
        text = f"[{self.name}, "

        for k, v in self.constraints.items():
            text += f"{k}: {v}, "

        text = text[:-2] + "]"

        return text

    @property
    def context(self) -> Optional[Context]:
        return self._context

    @context.setter
    def context(self, new: Optional[Context]):
        self._context = new

        for c in self.constraints.keys():
            self.constraints[c].context = new

    def __init__(self, name: str, **constraints: Node):
        self.name = name
        self.constraints = constraints

        self._context = None


_wildcard_wildcard = Wildcard('_', wildcard=Value(1))


class AddAndSub(AdvancedBinOp):
    base_operation_symbol = '+'
    inverse_operation_symbol = '-'
    constant_term_last = True

    identity = Value(0)

    @staticmethod
    def _invert_value(value: Node) -> Node:
        return -value

    @staticmethod
    def _repeat_value(value: Node, times: int) -> Node:
        return MulAndDiv.mul(Node.from_float(times), value)

    @staticmethod
    def _should_invert_value(value: Node) -> bool:
        if isinstance(value, Value):
            return value.data < 0

        if isinstance(value, AddAndSub):
            return len(value.base_values) > len(value.inverted_values)

        return False

    @classmethod
    def add(cls, *values: Node):
        return cls(base_values=values)

    @classmethod
    def sub(cls, *values: Node, positive_first_node: bool = True):
        if positive_first_node:
            return cls(base_values=values[:1], inverted_values=values[1:])
        else:
            return cls(inverted_values=values)

    def evaluate(self) -> Node:
        evaluated_fraction = [0, 1]
        others = Value(0)
        for c in self.base_values:
            ev = c.evaluate()
            if isinstance(ev, Value):
                evaluated_fraction[0] += ev.data
                continue

            if isinstance(ev, MulAndDiv) and len(ev.base_values) == len(ev.inverted_values) == 1:
                num = ev.base_values[0]
                denom = ev.inverted_values[0]
                if isinstance(num, Value) and isinstance(denom, Value):
                    num = num.data
                    denom = denom.data
                    evaluated_fraction = [(evaluated_fraction[0] * denom + num * evaluated_fraction[1]),
                                          (denom * evaluated_fraction[1])]
                    continue
            others += ev

        for c in self.inverted_values:
            ev = c.evaluate()
            if isinstance(ev, Value):
                evaluated_fraction[0] -= ev.data
                continue

            if isinstance(ev, MulAndDiv) and len(ev.base_values) == len(ev.inverted_values) == 1:
                num = ev.base_values[0]
                denom = ev.inverted_values[0]
                if isinstance(num, Value) and isinstance(denom, Value):
                    num = num.data
                    denom = denom.data
                    evaluated_fraction = [(evaluated_fraction[0] * denom - num * evaluated_fraction[1]),
                                          (denom * evaluated_fraction[1])]
                    continue
            others -= ev

        return (MulAndDiv.div(Value(evaluated_fraction[0]), Value(evaluated_fraction[1])) + others).reduce(
            evaluate=False)

    def approximate(self) -> float:
        result = 0.0
        for c in self.base_values:
            result += c.approximate()
        for c in self.inverted_values:
            result -= c.approximate()

        return result

    @staticmethod
    def _put_parentheses(value: Node) -> bool:
        # addition/substraction always have the lowest precedence
        return False

    def __neg__(self):
        result = AddAndSub(self.inverted_values, self.base_values)
        result.context = self.context
        return result

    def __add__(self, other):
        if isinstance(other, Real):
            result = AddAndSub(self.base_values + [Node.from_float(float(other))], self.inverted_values)
        elif isinstance(other, Node):
            result = AddAndSub(self.base_values + [other], self.inverted_values)
        else:
            return NotImplemented
        result.context = self.context
        return result

    def __radd__(self, other):
        if isinstance(other, Real):
            # noinspection PyTypeChecker
            result = AddAndSub([Value(other)] + self.base_values, self.inverted_values)
        elif isinstance(other, Node):
            result = AddAndSub([other] + self.base_values, self.inverted_values)
        else:
            return NotImplemented
        result.context = self.context
        return result

    def __sub__(self, other):
        if isinstance(other, Real):
            result = AddAndSub(self.base_values, self.inverted_values + [Node.from_float(float(other))])
        elif isinstance(other, Node):
            result = AddAndSub(self.base_values, self.inverted_values + [other])
        else:
            return NotImplemented
        result.context = self.context
        return result

    def __rsub__(self, other):
        if isinstance(other, Real):
            # noinspection PyTypeChecker
            result = AddAndSub([Value(other)] + self.inverted_values, self.base_values)
        elif isinstance(other, Node):
            result = AddAndSub([other] + self.inverted_values, self.base_values)
        else:
            return NotImplemented
        result.context = self.context
        return result


class MulAndDiv(AdvancedBinOp):
    base_operation_symbol = '*'
    inverse_operation_symbol = '/'
    constant_term_last = False

    identity = Value(1)
    absorbing_element = Value(0)

    @staticmethod
    def _repeat_value(value: Node, times: int) -> Node:
        return Pow(value, Value(times))

    @staticmethod
    def _invert_value(value: Node) -> Node:
        if isinstance(value, Value) and value.data in (-1, 1):
            return Value(value.data)

        return Value(1) / value

    @staticmethod
    def _should_invert_value(value: Node) -> bool:
        if isinstance(value, Value):
            return -1 < value.data < 1

        if isinstance(value, MulAndDiv):
            return len(value.base_values) > len(value.inverted_values)

        return False

    @classmethod
    def mul(cls, *values: Node):
        return cls(base_values=values)

    @classmethod
    def div(cls, *values: Node, multiply_first_node: bool = True):
        if multiply_first_node:
            return cls(base_values=values[:1], inverted_values=values[1:])
        else:
            return cls(inverted_values=values)

    def evaluate(self) -> Node:
        numerator = Value(1)
        numerator_value = Value(1)

        for c in self.base_values:
            ev = c.evaluate()
            if isinstance(ev, Value):
                numerator_value.data *= ev.data
                continue
            numerator *= ev

        denominator = Value(1)
        denominator_value = Value(1)
        for c in self.inverted_values:
            ev = c.evaluate()
            if isinstance(ev, Value):
                denominator_value.data *= ev.data
                continue
            denominator *= ev

        numerator_value, denominator_value = fraction.simplify_fraction(numerator_value.data, denominator_value.data)

        return ((numerator_value * numerator) / (denominator_value * denominator)).reduce(evaluate=False)

    def approximate(self) -> float:
        result = 1.0
        for c in self.base_values:
            result *= c.approximate()
        for c in self.inverted_values:
            result /= c.approximate()

        return result

    @staticmethod
    def _put_parentheses(value: Node) -> bool:
        return isinstance(value, AddAndSub)

    def __neg__(self):
        result = MulAndDiv(self.base_values + [Value(-1)], self.inverted_values)
        result.context = self.context
        return result

    def __mul__(self, other):
        if isinstance(other, Real):
            result = MulAndDiv(self.base_values + [Node.from_float(float(other))], self.inverted_values)
            result.context = self.context
            return result
        elif isinstance(other, Node):
            result = MulAndDiv(self.base_values + [other], self.inverted_values)
            result.context = self.context
            return result
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Real):
            # noinspection PyTypeChecker
            result = MulAndDiv([Node.from_float(other)] + self.base_values, self.inverted_values)
            result.context = self.context
            return result
        elif isinstance(other, Node):
            result = MulAndDiv([other] + self.base_values, self.inverted_values)
            result.context = self.context
            return result
        else:
            return NotImplemented

    def __div__(self, other):
        if isinstance(other, Real):
            result = MulAndDiv(self.base_values, self.inverted_values + [Node.from_float(float(other))])
            result.context = self.context
            return result
        elif isinstance(other, Node):
            result = MulAndDiv(self.base_values, self.inverted_values + [other])
            result.context = self.context
            return result
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, Real):
            # noinspection PyTypeChecker
            result = MulAndDiv([Node.from_float(other)] + self.inverted_values, self.base_values)
            result.context = self.context
            return result
        elif isinstance(other, Node):
            result = MulAndDiv([other] + self.inverted_values, self.base_values)
            result.context = self.context
            return result
        else:
            return NotImplemented

    def __str__(self):
        if (not self.inverted_values) and len(self.base_values) == 2:
            if self.base_values[0] == Node.from_float(-1):
                return f"-{self.base_values[1]}"
            if self.base_values[1] == Node.from_float(-1):
                return f"-{self.base_values[0]}"

        return super().__str__()


class Pow(BinOp):
    name = '^'

    right_identity = Value(1)

    left_absorbing_element = Value(1)
    left_absorbing_result = Value(1)

    right_absorbing_element = Value(0)
    right_absorbing_result = Value(1)

    def evaluate(self) -> Node:
        eval_left = self.left.evaluate()
        eval_right = self.right.evaluate()

        if isinstance(eval_right, Value):
            if isinstance(eval_left, Value):
                if eval_right.data < 0:
                    return 1 / Value(eval_left.data ** (-eval_right.data))
                return Value(eval_left.data ** eval_right.data)
            if isinstance(eval_left, MulAndDiv) and len(eval_left.base_values) == len(eval_left.inverted_values) == 1:
                num = eval_left.base_values[0]
                denom = eval_left.inverted_values[0]
                if isinstance(num, Value) and isinstance(denom, Value):
                    return (num ** eval_right).evaluate() / (denom ** eval_right).evaluate()

        return eval_left ** eval_right

    @staticmethod
    def approximator(left: float, right: float) -> float:
        return left ** right

    @staticmethod
    def _put_parentheses(value: Node) -> bool:
        return isinstance(value, AddAndSub) or isinstance(value, MulAndDiv)


@dataclass
class MatchResult:
    wildcards: dict[str, Node] = field(default_factory=dict)
    functions_wildcards: dict[str, Function] = field(default_factory=dict)
    weak: bool = False


class ReplacingError(Exception):
    pass
