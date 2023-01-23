import sly

from parser import parse

if __name__ == '__main__':
    a = True
    while a:

        try:
            expr = parse(input("E: "))
            print(" =>", expr)
            pat = parse(input("P: "))
            print(" =>", pat)
        except parse.ParsingError:
            expr = pat = None

        if expr is None or pat is None:
            print(" => Something went wrong...")
            continue

        print(" => Matches:", pat.matches(expr))
