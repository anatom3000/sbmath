# sbmath

Symbolic math parser/matcher/transformer/manipulator.

### Quick start

```rust
todo!();
```

### Todo

- [x] Expression parsing from string
- [x] Evaluating expressions
- [ ] Somewhat stable pattern matching
- [ ] Functions (take AST as argument for advanced transformations e.g. derivative)
- [x] Advanced wildcards (match only variables, values, etc.)
- [ ] Variables constraints (type, range...)

### API Progress

|  Protocols | `Node.reduce` | `Node.matches` | `Node.contains` | `Node.replace` |
|-----------:|:-------------:|:--------------:|:---------------:|:--------------:|
| `AdvBinOp` |      Yes      |      Yes       |       Yes       |      Yes       |
|      `Pow` |      Yes      |      Yes       |       Yes       |      Yes       |
|    `Value` |      Yes      |      Yes       |       Yes       |      Yes       |
| `Variable` |      Yes      |      Yes       |       Yes       |      Yes       |
| `Wildcard` |      Yes      |      Yes       |       Yes       |      Yes       |