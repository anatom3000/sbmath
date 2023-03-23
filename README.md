# sbmath

Symbolic math library. 

For now this can only do basic pattern matching on expression, but in the long run it should become a ~~scuffed~~ simpler version of `sympy`.

The pattern matching is still very buggy and wip, though it should be a bit more stable once the low-level API design is done. 

### Quick start

Run `main.py` to have an all-in-one REPL: 

```shell
$ python main.py
```

Alternatively you can modify `main.py` to run various specialized REPLs and/or tests.

### Installation

```shell
$ pip install -r requirements.txt
```

### Todo

- [x] Expression parsing from string
- [x] Evaluating expressions
- [ ] Somewhat stable pattern matching
- [x] Functions (take AST as argument for advanced transformations e.g. derivative)
- [x] Advanced wildcards (match only variables, values, etc.)
- [ ] Variables constraints/domain (type, range...)
- [ ] Add test suite
- [ ] Add documentation

### API Progress

|  Protocols | `Node.reduce` | `Node.matches` | `Node.contains` | `Node.replace` |
|-----------:|:-------------:|:--------------:|:---------------:|:--------------:|
| `AdvBinOp` |      Yes      |      Yes       |       Yes       |      Yes       |
|      `Pow` |      Yes      |      Yes       |       Yes       |      Yes       |
|    `Value` |      Yes      |      Yes       |       Yes       |      Yes       |
| `Variable` |      Yes      |      Yes       |       Yes       |      Yes       |
| `Wildcard` |      Yes      |      Yes       |       Yes       |      Yes       |


### License

This project is under the GPLv3 license.
