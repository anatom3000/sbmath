# noinspection PyCompatibility
from parser import parse, ParsingError


def pattern_matching():
    while True:
        try:
            expr = parse(input("E: "))
            print(" =>", expr)
            pat = parse(input("P: "))
            print(" =>", pat)
        except ParsingError:
            expr = pat = None

        if expr is None or pat is None:
            print(" => Something went wrong...")
            continue

        m = pat.matches(expr)
        print(" =>", f"{m}")


def contains():
    while True:
        try:
            expr = parse(input("E: "))
            print(" =>", expr)
            container = parse(input("C: "))
            print(" =>", container)
        except ParsingError:
            expr = container = None

        if expr is None or container is None:
            print(" => Something went wrong...")
            continue

        if container.contains(expr):
            print(f" => {expr} is in {container}")
        else:
            print(f" => {expr} is not in {container}")


def reduce():
    while True:
        try:
            expr = parse(input("E: "))
        except ParsingError:
            print(" => Something went wrong...")
            continue

        if expr is None:
            continue
        print(" =>", expr)

        r = expr.reduce()
        print(f"R: {r}")


def find_and_replace():
    while True:
        try:
            expr = parse(input("E: "))
            print(" =>", expr)
            oldp = parse(input("O: "))
            print(" =>", oldp)
            newp = parse(input("N: "))
            print(" =>", newp)

        except ParsingError:
            print(" => Something went wrong...")
            continue

        if None in (expr, oldp, newp):
            continue

        result = expr.replace(oldp, newp)
        print(f" => {result}")


if __name__ == '__main__':
    pattern_matching()
