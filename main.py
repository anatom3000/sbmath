# noinspection PyCompatibility
from parser import parse, ParsingError

if __name__ == '__main__':
    a = True
    while a:

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
