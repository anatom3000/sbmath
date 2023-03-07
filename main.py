# noinspection PyCompatibility
from parser import parse, ParsingError

from repl import repl


def matches(print_back=False):
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
        print(" => ", f"{m}")


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


def eq(print_back=False):
    while True:
        try:
            expr1 = parse(input("Expression: "))
            if print_back: print(" =>", expr1)
            expr2 = parse(input("Expression: "))
            if print_back: print(" =>", expr2)
        except ParsingError:
            expr1 = expr2 = None

        if expr1 is None or expr2 is None:
            print(" => Something went wrong...")
            continue

        print(" =>", expr1 == expr2)


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


def reduce_no_eval(print_back=False):
    while True:
        try:
            expr = parse(input("Expression: "))
        except ParsingError:
            print(" => Something went wrong...")
            continue

        if expr is None:
            continue

        if print_back: print(" =>", expr)

        r = expr.reduce_no_eval()
        print(f"R: {r}")


def evaluate(print_back=False):
    while True:
        try:
            expr = parse(input("Expression: "))
        except ParsingError:
            print(" => Something went wrong...")
            continue

        if expr is None:
            continue

        if print_back: print(" =>", expr)

        r = expr.evaluate()
        print(f"E: {r}")


def approx(print_back=False):
    while True:
        try:
            expr = parse(input("Expression: "))
        except ParsingError:
            print(" => Something went wrong...")
            continue

        if expr is None:
            continue

        if print_back: print(" =>", expr)

        r = expr.approximate()
        print(f"A: {r}")


def replace(print_back=False):
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


# FIXME:
#  % match
#   Pattern: [a]+[b]
#   Expr: y+x
#   Traceback (most recent call last):
#     File "/home/anatom/Bureau/dev/sbmath/main.py", line 156, in <module>
#       repl()
#     File "/home/anatom/Bureau/dev/sbmath/main.py", line 152, in repl
#       shell()
#     File "/home/anatom/Bureau/dev/sbmath/repl.py", line 125, in repl
#       result = pat.matches(expr)
#     File "/home/anatom/Bureau/dev/sbmath/tree.py", line 679, in matches
#       result = wildcard_self._match_wildcards(remaining_value, state)  # type: ignore
#     File "/home/anatom/Bureau/dev/sbmath/tree.py", line 623, in _match_wildcards
#       r = self._remove_wildcard_match(value, base_match_table.get_from_key(value)[0], base_match_table,
#     File "/home/anatom/Bureau/dev/sbmath/tree.py", line 481, in _remove_wildcard_match
#       inverted_match_table.remove_value(wildcard)
#     File "/home/anatom/Bureau/dev/sbmath/utils.py", line 87, in remove_value
#       raise KeyError(f"unknown value {value}")
#   KeyError: 'unknown value [a]'


def debug_broken_match():
    e = parse("x+y")
    p = parse("[a]+[b]")

    print(p.matches(e))


if __name__ == '__main__':
    repl()
