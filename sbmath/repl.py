import atexit
import re
import readline
import traceback

from . import _utils
from .parser import parse

from ._utils import debug

RED = "\033[0;31m"
YELLOW = "\033[1;33m"
LIGHT_GREEN = "\033[1;32m"
LIGHT_GRAY = "\033[0;37m"
CYAN = "\033[0;36m"
END = "\033[0m"

operations = ['eq', 'approx', 'eval', 'reduce', 'reduce_no_eval', 'match', 'replace', 'morph', 'contains', 'debug', 'exit']

start_text = f"""{CYAN}Interactive shell (alpha){END}
{CYAN}Available operations:{END}{LIGHT_GREEN} {', '.join(map(repr, operations))}{END}"""

BRACKETS = {'(': ')', '[': ']'}

histfile = '.sbmath_history'


def _completer(text, state):
    # get current line
    line = readline.get_line_buffer()

    # determine if we need to auto-insert a closing bracket
    if text in BRACKETS.keys() and not re.search(r'[)\]]', line, re.MULTILINE | re.DOTALL):
        return text + BRACKETS[text]

    # determine if we need to restrict completions to a closing bracket
    if re.search(r'[(\[]$', line, re.MULTILINE | re.DOTALL):
        matches = [BRACKETS[line[-1]]]
    else:
        # get list of completions
        completions = operations + list(BRACKETS.keys()) + list(BRACKETS.values())

        # filter completions by input text
        matches = [c for c in completions if c.startswith(text)]

    # return the next match
    return matches[state] if state < len(matches) else None


def repl():
    readline.parse_and_bind('tab: complete')  # Enable tab completion
    readline.set_completer(_completer)

    try:
        readline.read_history_file(histfile)  # Load previous history
    except FileNotFoundError:
        pass
    atexit.register(readline.write_history_file, histfile)  # Save new history

    print(start_text)

    while True:
        try:
            op = input(f"{LIGHT_GRAY}% {END}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print('\n')
            break

        if not op:
            continue

        if op not in operations:
            print(f"{RED}Operation not found!{END}", end=' ')
            print(f"{CYAN}Available operations:{END}{LIGHT_GREEN}", ', '.join(map(repr, operations[:-1])), 'and', repr(operations[-1]), END)

            continue

        if op == 'exit':
            break

        try:
            if op == 'eq':
                try:
                    expr1 = parse(input(f'{LIGHT_GRAY}Expr 1: {END}'))
                    debug(f" => {expr1}", flag='repl')
                except EOFError: break
                if expr1 is None:
                    continue
                try:
                    expr2 = parse(input(f'{LIGHT_GRAY}Expr 2: {END}'))
                    debug(f" => {expr2}", flag='repl')
                except EOFError: break
                if expr2 is None:
                    continue
                result = expr1 == expr2

            elif op == 'approx':
                try:
                    expr = parse(input(f"{LIGHT_GRAY}Expr: {END}"))
                    debug(f" => {expr}", flag='repl')
                except EOFError:
                    break
                if expr is None:
                    continue
                result = expr.approximate()

            elif op == 'eval':
                try:
                    expr = parse(input(f"{LIGHT_GRAY}Expr: {END}"))
                    debug(f" => {expr}", flag='repl')
                except EOFError:
                    break
                if expr is None:
                    continue
                result = expr.evaluate()

            elif op == 'reduce':
                try:
                    expr = parse(input(f"{LIGHT_GRAY}Expr: {END}"))
                    debug(f" => {expr}", flag='repl')

                    depth = input(f"{LIGHT_GRAY}Depth (leave blank for no limit): {END}")
                except EOFError:
                    break
                if expr is None:
                    continue

                if depth:
                    try:
                        depth = int(depth)
                    except ValueError:
                        continue
                else:
                    depth = -1

                result = expr.reduce(depth)

            elif op == 'reduce_no_eval':
                try:
                    expr = parse(input(f"{LIGHT_GRAY}Expr: {END}"))
                    debug(f" => {expr}", flag='repl')
                except EOFError:
                    break
                if expr is None:
                    continue
                result = expr.reduce_no_eval()

            elif op == 'match':
                try:
                    pat = parse(input(f"{LIGHT_GRAY}Pattern: {END}"))
                    debug(f" => {pat}", flag='repl')
                except EOFError:
                    break
                if pat is None:
                    continue
                try:
                    expr = parse(input(f"{LIGHT_GRAY}Expr: {END}"))
                    debug(f" => {expr}", flag='repl')
                except EOFError:
                    break
                if expr is None:
                    continue
                result = pat.matches(expr)

            elif op == 'replace':
                try:
                    old = parse(input(f"{LIGHT_GRAY}Old node: {END}"))
                    debug(f" => {old}", flag='repl')
                except EOFError:
                    break
                if old is None:
                    continue
                try:
                    new = parse(input(f"{LIGHT_GRAY}New node: {END}"))
                    debug(f" => {new}", flag='repl')
                except EOFError:
                    break
                if new is None:
                    continue
                try:
                    expr = parse(input(f"{LIGHT_GRAY}Expr: {END}"))
                    debug(f" => {expr}", flag='repl')
                except EOFError:
                    break
                if expr is None:
                    continue
                result = expr.replace(old, new)

            elif op == 'morph':
                try:
                    pat_old = parse(input(f"{LIGHT_GRAY}Old pattern: {END}"))
                    debug(f" => {pat_old}", flag='repl')
                except EOFError:
                    break
                if pat_old is None:
                    continue
                try:
                    pat_new = parse(input(f"{LIGHT_GRAY}New pattern: {END}"))
                    debug(f" => {pat_new}", flag='repl')
                except EOFError:
                    break
                if pat_new is None:
                    continue
                try:
                    expr = parse(input(f"{LIGHT_GRAY}Expr: {END}"))
                    debug(f" => {expr}", flag='repl')
                except EOFError:
                    break
                if expr is None:
                    continue
                result = expr.morph(pat_old, pat_new)

            elif op == 'contains':
                try:
                    expr = parse(input(f"{LIGHT_GRAY}Expr: {END}"))
                    debug(f" => {expr}", flag='repl')
                except EOFError:
                    break
                if expr is None:
                    continue
                try:
                    pat = parse(input(f"{LIGHT_GRAY}Pattern: {END}"))
                    debug(f" => {pat}", flag='repl')
                except EOFError:
                    break
                if pat is None:
                    continue
                result = expr.contains(pat)

            elif op == 'debug':
                _utils.DEBUG = not _utils.DEBUG

                if _utils.DEBUG:
                    result = "Debugging enabled!"
                else:
                    result = "Debugging disabled!"

            else:
                raise RuntimeError("operation not properly handled")

        except Exception as exc:
            print(f"{RED}An error occured during execution of operation:")

            print(f"{traceback.format_exc()}{END}")

            continue

        print(f" => {YELLOW}{result}{END}")
