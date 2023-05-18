from __future__ import annotations

import copy
from functools import cached_property
from itertools import zip_longest

from sbmath.expression import Expression, Value
from sbmath.ops.polynomial.monomial import exponents_lexicographic_max, VariableExponents, Monomial, term_div, \
    exponents_mul

zero = Value(0)


class Polynomial:
    def __init__(self, terms: dict[VariableExponents, Expression], variables: list[Expression]):
        # terms: [variable_powers: term_coefficient]
        # Ex: ax²y+by+2 => [(2, 1): a, (0, 1): b, (0, 0): 2]
        self.terms = terms
        self.variables = variables

    @property
    def leading_exponents(self) -> Optional[VariableExponents]:
        if self.is_zero():
            return None
        # lexicographic ordering
        leading_exponents = (0,) * len(self.variables)
        for variable_exponents in self.terms.keys():
            leading_exponents = exponents_lexicographic_max(leading_exponents, variable_exponents)

        return leading_exponents

    @property
    def leading_coefficient(self) -> Optional[Expression]:
        if self.is_zero():
            return None

        return self.terms[self.leading_exponents]

    @property
    def leading_term(self) -> Optional[Monomial]:
        if self.is_zero():
            return None

        le = self.leading_exponents
        return le, self.terms[le]

    @classmethod
    def constant(cls, value: Expression, variables: list[Expression]) -> Polynomial:
        if value == zero:
            return cls.zero(variables)

        return cls({(0,) * len(variables): value}, variables)

    @classmethod
    def zero(cls, variables: list[Expression]) -> Polynomial:
        return cls({}, variables)

    def is_zero(self) -> bool:
        return len(self.terms) == 0

    def to_monic(self) -> Polynomial:
        if len(self.terms.keys()) == 0:
            return self

        return Polynomial({k: (v / self.leading_coefficient).reduce() for k, v in self.terms.items()}, self.variables)

    def remainder(self, system: list[Polynomial]):

        if isinstance(system, Polynomial):
            system = [system]

        if any(p.is_zero() for p in system):
            raise ZeroDivisionError("polynomial division")

        remainder = self.zero(self.variables)
        leading_term = self.leading_term
        quotient: Polynomial = copy.deepcopy(self)

        while len(quotient.terms.keys()) != 0:
            for poly in system:
                divided = term_div(leading_term, poly.leading_term)
                if divided is not None:
                    divided_exponents, divided_coefficient = divided
                    for term_exponents, term_coefficient in poly.terms.items():
                        exponent_product = exponents_mul(term_exponents, divided_exponents)
                        coefficient_product = (quotient.terms.get(exponent_product,
                                                                  zero) - divided_coefficient * term_coefficient).reduce()
                        if coefficient_product == zero:
                            try:
                                del quotient.terms[exponent_product]
                            except KeyError: pass
                        else:
                            quotient.terms[exponent_product] = coefficient_product
                    leading_exponents = quotient.leading_exponents
                    if leading_exponents is not None:
                        leading_term = leading_exponents, quotient.terms[leading_exponents]
                    break
            else:
                leading_exponents, leading_coefficient = leading_term
                if leading_exponents in remainder.terms.keys():
                    remainder.terms[leading_exponents] += leading_coefficient
                else:
                    remainder.terms[leading_exponents] = leading_coefficient

                del quotient.terms[leading_exponents]

                leading_exponents = quotient.leading_exponents
                if leading_exponents is not None:
                    leading_term = leading_exponents, quotient.terms[leading_exponents]

        return remainder

    def __mod__(self, other: list[Polynomial]):
        return self.remainder(other)

    def __add__(self, other: Polynomial):
        if not isinstance(other, Polynomial):
            return NotImplemented

        if self.variables != other.variables:
            # TODO: handle case when variables don't match
            return NotImplemented

        terms = {}
        for exponents in set(*self.terms.keys(), *other.terms.keys()):
            self_coef = self.terms.get(exponents, default=zero)
            other_coef = other.terms.get(exponents, default=zero)

            new_coef = (self_coef + other_coef).reduce()
            if new_coef != Value(0):
                terms[exponents] = new_coef

        return Polynomial(terms, self.variables)

    def __sub__(self, other: Polynomial):
        if not isinstance(other, Polynomial):
            return NotImplemented

        if self.variables != other.variables:
            # TODO: handle case when variables don't match
            return NotImplemented

        terms = {}
        for exponents in {*self.terms.keys(), *other.terms.keys()}:
            self_coef = self.terms.get(exponents, zero)
            other_coef = other.terms.get(exponents, zero)

            new_coef = (self_coef - other_coef).reduce()
            if new_coef != Value(0):
                terms[exponents] = new_coef

        return Polynomial(terms, self.variables)

    def __eq__(self, other):
        return isinstance(other, Polynomial) and (self.terms, self.variables) == (other.terms, other.variables)

    def __hash__(self):
        return hash((frozenset(self.terms.items()), tuple(self.variables)))

    def __str__(self):
        return f"({self.terms}; {self.variables})"

    __repr__ = __str__

    def add_variable(self, *new_variables: Expression) -> Polynomial:
        return Polynomial({(*k, *(0,)*len(new_variables)): v for k, v in self.terms.items()}, self.variables + list(new_variables))
