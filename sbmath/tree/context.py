from __future__ import annotations

from dataclasses import dataclass

# from sbmath.tree import Function  # commented import for type annotations without circular imports


@dataclass
class Context:
    functions: dict[str, Function]

    def register_function(self, func: Function):
        self.functions[func.name] = func

    def get_function(self, name: str) -> Function:
        return self.functions[name]

    def delete_function(self, name: str):
        del self.functions[name]


class MissingContextError(Exception):
    pass
