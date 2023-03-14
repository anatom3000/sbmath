from dataclasses import dataclass

# from sbmath.tree import FunctionNode


@dataclass
class Context:
    functions: dict[str, FunctionNode]

    def register_function(self, func: FunctionNode):
        self.functions[func.name] = func

    def delete_function(self, name: str):
        del self.functions[name]
