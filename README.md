# sbmath

Symbolic math parser/matcher/transformer/manipulator.

### Quick start

```rust
todo!();
```

### Todo

- [x] Expression parsing from string
- [x] Evaluating expressions
- [x] Match sums and differences together (ex: `(2+x-1)` matches `(-1-(-2)+x)`)
- [ ] Functions (take AST as argument for advanced transformations e.g. derivative)
- [x] Advanced wildcards (match only variables, values, etc.)
- [ ] Variables constraints (type, range...)

### API Progress

| Protocols  | `Node.reduce` | `Node.matches` | `Node.contains` | `Node.replace` |
| ----------:|:-------------:|:--------------:|:---------------:|:--------------:|
| `AdvBinOp` | Yes           | [FIXME] Yes    | Yes             | Yes            |
| `Pow`      | Yes           | TODO           | Yes             | Yes            |
| `Value`    | Yes           | Yes            | Yes             | Yes            |
| `Variable` | Yes           | Yes            | Yes             | Yes            |
| `Wildcard` | Yes           | [FIXME?] Yes   | Yes             | Yes            |