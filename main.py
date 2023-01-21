import sly

from parser import parse

if __name__ == '__main__':
    while True:

        try:
            expr = parse(input("E: "))
            print(" =>", expr)
            pat = parse(input("P: "))
            print(" =>", pat)
        except sly.lex.LexError:
            expr = pat = None

        if expr is None or pat is None:
            print(" => Something went wrong...")
            continue

        print(" => Matches:", pat.matches(expr))
