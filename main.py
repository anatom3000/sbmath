# noinspection PyCompatibility
from parser import parse, ParsingError


def pattern_matching(print_back=False):
    # FIXME: `(x-9)(x+9)` does not match `([A]+[B])([A]-[B])`

    while True:
        try:
            expr = parse(input("Expression: "))
            if print_back: print(" =>", expr)
            pat = parse(input("Pattern: "))
            if print_back: print(" =>", pat)
        except ParsingError:
            expr = pat = None

        if expr is None or pat is None:
            print(" => Something went wrong...")
            continue

        m = pat.matches(expr)
        print(" =>", f"{m}")


def contains(print_back=False):
    while True:
        try:
            expr = parse(input("Expression: "))
            if print_back: print(" =>", expr)
            container = parse(input("Container: "))
            if print_back: print(" =>", container)
        except ParsingError:
            expr = container = None

        if expr is None or container is None:
            print(" => Something went wrong...")
            continue

        if container.contains(expr):
            print(f" => {expr} is in {container}")
        else:
            print(f" => {expr} is not in {container}")


def reduce(print_back=False):
    while True:
        try:
            expr = parse(input("Expression: "))
        except ParsingError:
            print(" => Something went wrong...")
            continue

        if expr is None:
            continue
        if print_back: print(" =>", expr)

        r = expr.reduce()
        print(f"R: {r}")


def find_and_replace(print_back=False):
    while True:
        try:
            expr = parse(input("Expression: "))
            if print_back: print(" =>", expr)
            oldp = parse(input("Current pattern: "))
            if print_back: print(" =>", oldp)
            newp = parse(input("New pattern: "))
            if print_back: print(" =>", newp)

        except ParsingError:
            print(" => Something went wrong...")
            continue

        if None in (expr, oldp, newp):
            continue

        result = expr.replace(oldp, newp)
        print(f" => {result}")


if __name__ == '__main__':
    reduce()
