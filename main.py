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


if __name__ == '__main__':
    pattern_matching()
