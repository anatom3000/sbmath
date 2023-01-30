from __future__ import annotations

from typing import Optional

import sly

import tree


class ParsingError(Exception):
    pass


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


# noinspection PyUnresolvedReferences
class Parser(sly.Parser):
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
        return tree.AddAndSub.add(p.expr0, p.expr1)

    @_('expr MINUS expr')
    def expr(self, p):
        return tree.AddAndSub.sub(p.expr0, p.expr1)

    @_('expr TIMES expr')
    def expr(self, p):
        return tree.MulAndDiv.mul(p.expr0, p.expr1)

    @_('expr DIVIDE expr')
    def expr(self, p):
        return tree.MulAndDiv.div(p.expr0, p.expr1)

    @_('expr POW expr')
    def expr(self, p):
        return tree.Pow(p.expr0, p.expr1)

    @_('MINUS expr %prec UMINUS')
    def expr(self, p):
        if isinstance(p.expr, tree.Value):
            return tree.Value(-p.expr.data)
        else:
            return tree.Mul(tree.Value(-1.0), p.expr)

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
        return tree.Value(float(p.NUMBER))

    @_('IDENT')
    def ident(self, p):
        return tree.Variable(p.IDENT)

    @_('LBRACK IDENT RBRACK')
    def wildcard(self, p):
        return tree.Wildcard(p.IDENT)

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
        return tree.Wildcard(p.IDENT, **p.wc_args)

    @_('exprblock exprblock %prec IMPMUL')
    def expr(self, p):
        return tree.Mul(p.exprblock0, p.exprblock1)

    @_('number ident %prec IMPMUL')
    def expr(self, p):
        return tree.Mul(p.number, p.ident)

    @_('number wildcard %prec IMPMUL')
    def expr(self, p):
        return tree.Mul(p.number, p.wildcard)

    @_('number exprblock %prec IMPMUL')
    def expr(self, p):
        return tree.Mul(p.number, p.exprblock)

    @_('LPAREN expr RPAREN')
    def exprblock(self, p):
        return p.expr

_lexer = Lexer()
_parser = Parser()


def parse(data: str | tree.Numerics) -> Optional[tree.Node]:
    if isinstance(data, tree.Numerics):
        return tree.Value(float(data))

    if not data:
        return None

    result = _parser.parse(_lexer.tokenize(data))

    if not isinstance(result, tree.Node):
        raise ParsingError(f"parser returned {repr(result)} of type {type(result).__name__}, expected type Node")

    return result


__all__ = ["parse", "ParsingError"]
