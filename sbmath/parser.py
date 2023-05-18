from __future__ import annotations

from typing import Optional
from numbers import Real

import sly

import sbmath.expression as expression


# noinspection PyUnboundLocalVariable,PyUnresolvedReferences
class Lexer(sly.Lexer):
    tokens = {
        IDENT, NUMBER,
        PLUS, MINUS,
        POW,
        TIMES, DIVIDE,
        LPAREN, RPAREN,
        LBRACK, RBRACK,
        ARG_SEP, ARG_ASSIGN,
        EQUALS,
        DEFINED_EXPR_SEP, DEFINED_EXPR_APPROX, DEFINED_EXPR_INACCURACY,
    }

    # String containing ignored characters between tokens
    ignore = ' \t'

    # Regular expression rules for tokens
    IDENT = r'[a-zA-Z_][a-zA-Z0-9_]*'

    # https://stackoverflow.com/questions/12929308/python-regular-expression-that-matches-floating-point-numbers
    NUMBER = r'(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'

    PLUS = r'\+'
    MINUS = r'-'

    POW = r'(\*\*|\^)'  # either '**' or '^'

    TIMES = r'\*'
    DIVIDE = r'/'

    LPAREN = r'\('
    RPAREN = r'\)'

    LBRACK = r'\['
    RBRACK = r'\]'
    ARG_ASSIGN = r':'
    ARG_SEP = r','

    EQUALS = r'='

    DEFINED_EXPR_SEP = r'\|'
    DEFINED_EXPR_APPROX = r'\~='
    DEFINED_EXPR_INACCURACY = r'~'


_ = lambda _: (lambda _: None)  # fake value to make mypy happy


# noinspection PyUnresolvedReferences
class Parser(sly.Parser):
    # debugfile = "parser.out"

    def __init__(self, context: expression.Context = None):
        super().__init__()
        self.context = context
        self.convert_to_exp = self.context is not None and "exp" in self.context.functions

    # Get the token list from the lexer (required)
    tokens = Lexer.tokens

    # debugfile = 'parser.out'

    precedence = (
        ('nonassoc', EQUALS),
        ('left', PLUS, MINUS),
        ('left', TIMES, DIVIDE, IMPMUL),
        ('right', UMINUS),
        ('right', POW),
    )

    # Grammar rules and actions
    @_('expr PLUS expr')
    def expr(self, p):
        result = p.expr0 + p.expr1
        result.context = self.context
        return result

    @_('expr MINUS expr')
    def expr(self, p):
        result = p.expr0 - p.expr1
        result.context = self.context
        return result

    @_('expr TIMES expr')
    def expr(self, p):
        result = p.expr0 * p.expr1
        result.context = self.context
        return result

    @_('expr DIVIDE expr')
    def expr(self, p):
        result = p.expr0 / p.expr1
        result.context = self.context
        return result

    @_('MINUS expr %prec UMINUS')
    def expr(self, p):
        return -p.expr

    @_('PLUS expr %prec UMINUS')
    def expr(self, p):
        return p.expr

    @_('number')
    def expr(self, p):
        return p.number

    @_('ident')
    def expr(self, p):
        return p.ident

    @_('wildcard')
    def expr(self, p):
        return p.wildcard

    @_('exprblock')
    def expr(self, p):
        return p.exprblock

    @_('NUMBER')
    def number(self, p):
        result = expression.Expression.from_float(float(p.NUMBER))
        result.context = self.context
        return result

    @_('IDENT')
    def ident(self, p):
        result = expression.Variable(p.IDENT)
        result.context = self.context
        return result

    @_('expr EQUALS expr')
    def equality(self, p):
        result = expression.Equality(p.expr0, p.expr1)

        result.context = self.context
        return result

    @_('equality')
    def expr(self, p):
        return p.equality

    @_('DEFINED_EXPR_SEP DEFINED_EXPR_APPROX number DEFINED_EXPR_INACCURACY number')
    def definednode_approx(self, p):
        return p.number0, p.number1

    @_('DEFINED_EXPR_SEP DEFINED_EXPR_APPROX number')
    def definednode_approx(self, p):
        return p.number, None

    @_('')
    def definednode_approx(self, p):
        return None, None

    @_('LPAREN ident DEFINED_EXPR_SEP equality definednode_approx RPAREN')
    def expr(self, p):
        return expression.Solution(p.equality, p.ident, approximation=p.definednode_approx[0], inaccuracy=p.definednode_approx[1])

    @_('LBRACK IDENT RBRACK')
    def wildcard(self, p):
        result = expression.Wildcard(p.IDENT)
        result.context = self.context
        return result

    @_('IDENT ARG_ASSIGN expr')
    def wc_arg(self, p):
        return {p.IDENT: p.expr}

    @_('wc_arg')
    def wc_args(self, p):
        return p.wc_arg

    @_('wc_args ARG_SEP wc_arg')
    def wc_args(self, p):
        return {**p.wc_args, **p.wc_arg}

    @_("LBRACK IDENT ARG_SEP wc_args RBRACK")
    def wildcard(self, p):
        result = expression.Wildcard(p.IDENT, **p.wc_args)
        result.context = self.context
        return result

    @_('number ident POW expr %prec POW')
    def expr(self, p):
        result = expression.MulAndDiv.mul(p.number, expression.Pow(p.ident, p.expr))
        result.context = self.context
        return result

    @_('IDENT exprblock %prec IMPMUL')
    def expr(self, p):
        result = expression.FunctionApplication(p.IDENT, p.exprblock)
        result.context = self.context
        return result

    @_('wildcard exprblock %prec IMPMUL')
    def expr(self, p):
        result = expression.FunctionWildcard.from_wildcard(p.wildcard, p.exprblock)
        result.context = self.context
        return result

    @_('exprblock exprblock %prec IMPMUL')
    def expr(self, p):
        result = p.exprblock0 * p.exprblock1
        result.context = self.context
        return result

    @_('number ident %prec IMPMUL')
    def expr(self, p):
        result = p.number * p.ident
        result.context = self.context
        return result

    @_('number wildcard %prec IMPMUL')
    def expr(self, p):
        result = p.number * p.wildcard
        result.context = self.context
        return result

    @_('number exprblock %prec IMPMUL')
    def expr(self, p):
        result = p.number * p.exprblock
        result.context = self.context
        return result

    @_('LPAREN expr RPAREN')
    def exprblock(self, p):
        return p.expr

    @_('expr POW expr')
    def expr(self, p):
        if self.convert_to_exp and isinstance(p.expr0, expression.Variable) and p.expr0.data == 'e':
            result = expression.FunctionApplication("exp", p.expr1)
        else:
            result = p.expr0 ** p.expr1

        result.context = self.context
        return result


_lexer = Lexer()

# sentinel value to indicate the user did not provide a context to the parser
# we cannot use None since it can be passed to mean no context
ContextNotGiven = object()
_DEFAULT_CONTEXT = None  # is changed at runtime by `sbmath.expression.context.std` to prevent circular imports


def parse(data: str | Real, context: Optional[expression.Context] = ContextNotGiven) -> Optional[expression.Expression]:
    if context == ContextNotGiven:
        context = _DEFAULT_CONTEXT

    if isinstance(data, Real):
        result = expression.Expression.from_float(float(data))
        result.context = context
        return result

    if not data:
        return None

    parser = Parser(context)

    result = parser.parse(_lexer.tokenize(data))

    if result is None:
        return None

    if not isinstance(result, expression.Expression):
        raise ParsingError(f"parser returned {repr(result)} of type {type(result).__name__}, expected type Expression")

    return result


class ParsingError(Exception):
    pass


__all__ = ["parse", "ParsingError"]
