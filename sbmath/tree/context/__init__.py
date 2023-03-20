from __future__ import annotations

from dataclasses import dataclass, field

# from sbmath.tree import Function  # commented import for type annotations without circular imports


@dataclass
class Context:
    functions: dict[str, Function] = field(default_factory=dict)
    variables: dict[str, Node] = field(default_factory=dict)

    def add_function(self, func: Function):
        self.functions[func.name] = func

    def add_variable(self, name: str, value: Node):
        value.context = self
        self.variables[name] = value

    def __add__(self, other: Context) -> Context:
        if not isinstance(other, Context):
            return NotImplemented

        new = Context(
            functions={**self.functions.copy(), **other.functions.copy()},
            variables={**self.variables.copy(), **other.variables.copy()}
        )

        return new

    def __iadd__(self, other: Context):
        if not isinstance(other, Context):
            return NotImplemented

        self.functions.update(other.functions)
        self.variables.update(other.variables)


class MissingContextError(Exception):
    pass
