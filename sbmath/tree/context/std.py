import copy
import math
from dataclasses import FrozenInstanceError

from sbmath.tree.context import Context
from sbmath.tree.functions import Function, PythonFunction
from sbmath.parser import parse as _parse
import sbmath.parser

class FrozenContextError(Exception):
    pass


class _StdFrozenContext:
    def __init__(self, name: str, context: Context):
        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "_name", name)

    @property
    def functions(self) -> dict[str, Function]:
        return self._context.functions.copy()

    @property
    def variables(self) -> dict[str, Function]:
        return self._context.variables.copy()

    def __call__(self) -> Context:
        return copy.deepcopy(self._context)

    def __setattr__(self, key, value):
        raise FrozenContextError(f"cannot modify {self._name}")


_std = Context()

parse = lambda x: _parse(x, _std)

_std.add_variable("pi", parse(math.pi))
_std.add_variable("tau", parse(math.tau))
_std.add_variable("e", parse(math.e))

# not the optimal implementation for abs since it will approximate any parameter
# TODO: better implementation that integrates better with the symbolic representation
_std.add_function(
    PythonFunction(abs)
)

_std.add_function(
    PythonFunction(math.sqrt, {
        parse("[arg]^2"): parse("[arg]")
    })
)

_std.add_function(
    PythonFunction(math.exp, {
        parse(0): parse(1),
        parse(1): parse("e"),
        parse("ln([arg])"): parse("[arg]")
    })
)

_std.add_function(
    PythonFunction(math.log, {
        parse(1): parse(0),
        parse("e"): parse(1),
        parse("exp([arg])"): parse("[arg]")
    }, "ln")
)

_std.add_function(
    PythonFunction(math.log10, {
        parse("10^[arg]"): parse("[arg]")
    }, "log")
)

std = _StdFrozenContext("std", _std)
del _std

sbmath.parser._DEFAULT_CONTEXT = std

__all__ = ['std']
