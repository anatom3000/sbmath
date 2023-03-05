# noinspection PyCompatibility
from parser import parse, ParsingError


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


def repl():
    RED = "\033[0;31m"
    YELLOW = "\033[1;33m"
    LIGHT_GREEN = "\033[1;32m"
    LIGHT_GRAY = "\033[0;37m"
    CYAN = "\033[0;36m"
    END = "\033[0m"

    operations = ('eq', 'approx', 'eval', 'reduce', 'match', 'replace', 'contains')

    print(f"{CYAN}Interactive shell (alpha){END}")
    print(f"{CYAN}Available operations:{END}{LIGHT_GREEN}", ', '.join(map(repr, operations[:-1])), 'and',
          repr(operations[-1]), END)

    while True:
        op = input(f"{LIGHT_GRAY}% {END}")

        if op not in operations:
            print(f"{RED}Operation not found!{END}", end=' ')
            print(f"{CYAN}Available operations:{END}{LIGHT_GREEN}", ', '.join(map(repr, operations[:-1])), 'and',
                  repr(operations[-1]), END)

            continue

        if op == 'eq':
            expr1 = parse(input(f'{LIGHT_GRAY}Expr 1: {END}'))

            if expr1 is None:
                continue
            expr2 = parse(input(f'{LIGHT_GRAY}Expr 2: {END}'))

            if expr2 is None:
                continue
            result = expr1 == expr2

        elif op == 'approx':
            expr = parse(input(f"{LIGHT_GRAY}Expr: {END}"))

            if expr is None:
                continue
            result = expr.approximate()

        elif op == 'eval':
            expr = parse(input(f"{LIGHT_GRAY}Expr: {END}"))

            if expr is None:
                continue
            result = expr.approximate()

        elif op == 'reduce':
            expr = parse(input(f"{LIGHT_GRAY}Expr: {END}"))

            if expr is None:
                continue
            result = expr.reduce()

        elif op == 'match':
            pat = parse(input(f"{LIGHT_GRAY}Pattern: {END}"))

            if pat is None:
                continue
            expr = parse(input(f"{LIGHT_GRAY}Expr: {END}"))

            if expr is None:
                continue

            result = pat.matches(expr)

        elif op == 'replace':
            pat_old = parse(input(f"{LIGHT_GRAY}Old attern: {END}"))

            if pat_old is None:
                continue
            pat_new = parse(input(f"{LIGHT_GRAY}New pattern: {END}"))

            if pat_new is None:
                continue
            expr = parse(input(f"{LIGHT_GRAY}Expr: {END}"))

            if expr is None:
                continue

            result = expr.replace(pat_old, pat_new)

        elif op == 'contains':
            expr = parse(input(f"{LIGHT_GRAY}Expr: {END}"))

            if expr is None:
                continue
            pat = parse(input(f"{LIGHT_GRAY}Pattern: {END}"))

            if pat is None:
                continue

            result = expr.contains(pat)
        else:
            raise RuntimeError("operation not properly handled")

        print(f" => {YELLOW}{result}{END}")


if __name__ == '__main__':
    repl()
