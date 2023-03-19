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
        self.variables[name] = value


class MissingContextError(Exception):
    pass
