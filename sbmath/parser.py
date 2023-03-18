from __future__ import annotations

from typing import Optional
from numbers import Real

import sly

from sbmath import tree


# noinspection PyUnboundLocalVariable,PyUnresolvedReferences
class Lexer(sly.Lexer):
    tokens = {
        IDENT, NUMBER,
        PLUS, MINUS,
        POW,
        TIMES, DIVIDE,
        LPAREN, RPAREN,
        LBRACK, RBRACK,
        ARG_SEP, ARG_ASSIGN
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

    ARG_ASSIGN = r'='
    ARG_SEP = r','


_ = lambda _: (lambda _: None)  # fake value to make mypy happy


# noinspection PyUnresolvedReferences
class Parser(sly.Parser):
    def __init__(self, context: tree.Context = None):
        super().__init__()
        self.context = context

    # Get the token list from the lexer (required)
    tokens = Lexer.tokens

    # debugfile = 'parser.out'

    precedence = (
        ('left', PLUS, MINUS),
        ('left', IMPMUL),
        ('right', TIMES, DIVIDE),
        ('right', POW),
        ('left', UMINUS),
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

    @_('expr POW expr')
    def expr(self, p):
        result = p.expr0 ** p.expr1
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
        result = tree.Value(float(p.NUMBER))
        result.context = self.context
        return result

    @_('IDENT')
    def ident(self, p):
        result = tree.Variable(p.IDENT)
        result.context = self.context
        return result

    @_('LBRACK IDENT RBRACK')
    def wildcard(self, p):
        result = tree.Wildcard(p.IDENT)
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
        result = tree.Wildcard(p.IDENT, **p.wc_args)
        result.context = self.context
        return result

    @_('IDENT exprblock %prec IMPMUL')
    def expr(self, p):
        result = tree.FunctionApplication(p.IDENT, p.exprblock)
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

_lexer = Lexer()


def parse(data: str | Real, context: tree.Context = None) -> Optional[tree.Node]:
    if isinstance(data, Real):
        result = tree.Value(float(data))
        result.context = context
        return result

    if not data:
        return None

    parser = Parser(context)

    result = parser.parse(_lexer.tokenize(data))

    if not isinstance(result, tree.Node):
        raise ParsingError(f"parser returned {repr(result)} of type {type(result).__name__}, expected type Node")

    return result


class ParsingError(Exception):
    pass


__all__ = ["parse", "ParsingError"]
