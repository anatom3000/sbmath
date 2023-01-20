import sly

import tree


# noinspection PyUnboundLocalVariable
class Lexer(sly.Lexer):
    tokens = {
        IDENT, NUMBER,
        PLUS, MINUS,
        POW,
        TIMES, DIVIDE,
        LPAREN, RPAREN,
        LBRACK, RBRACK,
    }

    # String containing ignored characters between tokens
    ignore = ' \t'

    # Regular expression rules for tokens
    IDENT = r'[a-zA-Z][a-zA-Z0-9_]*'

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


class Parser(sly.Parser):
    # Get the token list from the lexer (required)
    tokens = Lexer.tokens

    debugfile = 'parser.out'

    # tokens = {
    #         IDENT, NUMBER,
    #         PLUS, MINUS,
    #         POW,
    #         TIMES, DIVIDE,
    #         LPAREN, RPAREN,
    #         LBRACK, RBRACK,
    #     }

    precedence = (
        ('left', PLUS, MINUS),
        ('left', IMPMUL),
        ('right', TIMES, DIVIDE),
        ('right', UMINUS),
        ('right', POW),
    )

    # Grammar rules and actions
    @_('expr PLUS expr')
    def expr(self, p):
        return tree.Add(p.expr0, p.expr1)

    @_('expr MINUS expr')
    def expr(self, p):
        return tree.Sub(p.expr0, p.expr1)

    @_('expr TIMES expr')
    def expr(self, p):
        return tree.Mul(p.expr0, p.expr1)

    @_('expr DIVIDE expr')
    def expr(self, p):
        return tree.Div(p.expr0, p.expr1)

    @_('expr POW expr')
    def expr(self, p):
        return tree.Pow(p.expr0, p.expr1)

    @_('MINUS expr %prec UMINUS')
    def expr(self, p):
        return tree.Sub(p.expr0, p.expr1)

    @_('PLUS expr %prec UMINUS')
    def expr(self, p):
        return f"{p.expr}"

    @_('number')
    def expr(self, p):
        return p.number

    @_('ident')
    def expr(self, p):
        return p.ident

    @_('wildcard')
    def expr(self, p):
        return p.wildcard

    @_('NUMBER')
    def number(self, p):
        return tree.Value(float(p.NUMBER))

    @_('IDENT')
    def ident(self, p):
        return tree.Variable(p.IDENT)

    @_('LBRACK RBRACK')
    def wildcard(self, _p):
        return tree.Wildcard()

    @_('number expr %prec IMPMUL')
    def expr(self, p):
        return tree.Mul(p.number, p.expr)

    @_('LPAREN expr RPAREN')
    def expr(self, p):
        return p.expr
