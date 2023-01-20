from parser import Lexer, Parser

if __name__ == '__main__':
    lexer = Lexer()
    parser = Parser()

    while True:
        try:
            text = input('sbm% ')
            result = parser.parse(lexer.tokenize(text))
            print(result)
        except EOFError:
            break
