from __future__ import annotations

import copy
from collections.abc import Iterable

from sbmath.ops.polynomial.core import Polynomial
from sbmath.ops.polynomial.monomial import VariableExponents, polynomial_monomial_product, exponents_mul, exponents_div, exponents_lcm


def select_leading_term(system: list[Polynomial], critical_pair_indices: set[tuple[int, int]]) -> VariableExponents:
    return min(
        critical_pair_indices,
        key=lambda pair: exponents_lcm(system[pair[0]].leading_exponents,
                                       system[pair[1]].leading_exponents)
    )


def normal_form(system: list[Polynomial],
                polynomial: Polynomial,
                poly_to_index: dict[Polynomial, int],
                possible_base_indices: Iterable[int]) \
        -> Optional[list[Polynomial], tuple[dict[Polynomial, int], int]]:
    remainder = polynomial.remainder([system[index] for index in possible_base_indices])

    if remainder.is_zero():
        return None

    remainder = remainder.to_monic()

    if remainder not in poly_to_index.keys():
        poly_to_index[remainder] = len(system)
        system.append(remainder)

    return system, poly_to_index, poly_to_index[remainder]


def update(system: list[Polynomial], possible_base_indices: set[int], critical_pairs_indices: set[tuple[int, int]],
           selected_polynomial_index: int) \
        -> tuple[set[int], set[int]]:
    # update G using the set of critical pairs B and selected_polynomial
    # [BW] page 230
    selected_polynomial = system[selected_polynomial_index]
    selected_leading_exponents = selected_polynomial.leading_exponents

    # filter new pairs (selected_polynomial, g), g in G
    bases_indices_copy = copy.deepcopy(possible_base_indices)
    D = set()

    while bases_indices_copy:
        # select a pair (selected_polynomial, g) by popping an element from bases_indices_copy
        index = bases_indices_copy.pop()
        g = system[index]
        mg = g.leading_exponents
        lcm = exponents_lcm(selected_leading_exponents, mg)

        def lcm_divides(ip):
            # LCM(LM(selected_polynomial), LM(p)) divides LCM(LM(selected_polynomial), LM(g))
            return exponents_div(lcm,
                                exponents_lcm(selected_leading_exponents, system[ip].leading_exponents))

        # HT(selected_polynomial) and HT(g) disjoint: selected_leading_exponents*mg == LCMhg
        if exponents_mul(selected_leading_exponents, mg) == lcm or (
                not any(lcm_divides(ipx) for ipx in bases_indices_copy) and
                not any(lcm_divides(pr[1]) for pr in D)):
            D.add((selected_polynomial_index, index))

    E = set()

    while D:
        # select selected_polynomial, g from D (selected_polynomial the same as above)
        selected_polynomial_index, index = D.pop()
        mg = system[index].leading_exponents
        lcm = exponents_lcm(selected_leading_exponents, mg)

        if exponents_mul(selected_leading_exponents, mg) != lcm:
            E.add((selected_polynomial_index, index))

    # filter old pairs
    new_critical_pairs_indices = set()

    while critical_pairs_indices:
        # select g1, g2 from B (-> CP)
        ig1, ig2 = critical_pairs_indices.pop()
        mg1 = system[ig1].leading_exponents
        mg2 = system[ig2].leading_exponents
        lcm = exponents_lcm(mg1, mg2)

        # if HT(selected_polynomial) does not divide lcm(HT(g1), HT(g2))
        if not exponents_div(lcm, selected_leading_exponents) \
                or exponents_lcm(mg1, selected_leading_exponents) == lcm \
                or exponents_lcm(mg2, selected_leading_exponents) == lcm:
            new_critical_pairs_indices.add((ig1, ig2))

    new_critical_pairs_indices |= E

    # filter polynomials
    new_possible_base_indices = set()

    while possible_base_indices:
        index = possible_base_indices.pop()
        mg = system[index].leading_exponents

        if not exponents_div(mg, selected_leading_exponents):
            new_possible_base_indices.add(index)

    new_possible_base_indices.add(selected_polynomial_index)

    # noinspection PyTypeChecker
    return new_possible_base_indices, new_critical_pairs_indices


def spolynomial(polynomial1: Polynomial, polynomial2: Polynomial) -> Polynomial:
    leading_exponents1 = polynomial1.leading_exponents
    leading_exponents2 = polynomial2.leading_exponents

    # print(f"{polynomial2 = }")
    # print(f"{leading_exponents2 = }")

    lcm = exponents_lcm(leading_exponents1, leading_exponents2)
    # print(f"{lcm = }")
    #
    # print(f"{polynomial2 = }")
    # print(f"{leading_exponents2 = }")
    #
    # print(f"{leading_exponents1 = }, {exponents_div(lcm, leading_exponents1) = }")
    # print(f"{leading_exponents2 = }, {exponents_div(lcm, leading_exponents2) = }")

    return polynomial_monomial_product(polynomial1, exponents_div(lcm, leading_exponents1)) \
        - polynomial_monomial_product(polynomial2, exponents_div(lcm, leading_exponents2))


def grobner_bases(system: list[Polynomial]) -> list[Polynomial]:
    # the overall logic of this function comes from sympy's implementation of the buchberger's algorithm
    # I only restructured the code, added type annotations and made the algorithm symbolic
    # while the original implementation could only use
    # https://github.com/sympy/sympy/blob/26f7bdbe3f860e7b4492e102edec2d6b429b5aaf/sympy/polys/groebnertools.py#L50
    if len(system) == 0:
        return []

    tmp_system = system[:]
    while True:
        system: list[Polynomial] = tmp_system[:]
        tmp_system: list[Polynomial] = []
        for index, polynomial in enumerate(system):
            remainder = polynomial.remainder(system[:index])
            if not remainder.is_zero():
                tmp_system.append(remainder.to_monic())

        if system == tmp_system:
            del tmp_system
            break

    poly_to_index: dict[Polynomial, int] = {polynomial: index for index, polynomial in enumerate(system)}
    indices: set[int] = set(range(len(system)))
    possible_base_indices: set[int] = set()
    critical_pair_indices: set[tuple[int, int]] = set()

    while len(indices) != 0:
        smallest_polynomial = min(indices, key=lambda i: system[i].leading_exponents)
        indices.remove(smallest_polynomial)

        possible_base_indices, critical_pair_indices = update(
            system,
            possible_base_indices,
            critical_pair_indices,
            smallest_polynomial
        )

    while len(critical_pair_indices) != 0:
        poly1_index, poly2_index = select_leading_term(system, critical_pair_indices)
        critical_pair_indices.remove((poly1_index, poly2_index))

        spoly = spolynomial(system[poly1_index], system[poly2_index])

        result = normal_form(
            system,
            spoly,
            poly_to_index,
            sorted(possible_base_indices, key=lambda i: system[i].leading_exponents)
        )

        if result is None:
            continue

        system, poly_to_index, normal_index = result

        possible_base_indices, critical_pair_indices = update(
            system, possible_base_indices, critical_pair_indices, normal_index
        )

    reduced_base_indices: set[int] = set()

    for index in possible_base_indices:
        result = normal_form(system, system[index], poly_to_index, possible_base_indices - {index})
        if result is not None:
            system, poly_to_index, normal_index = result
            reduced_base_indices.add(normal_index)

    return sorted([system[index] for index in reduced_base_indices], key=lambda p: p.leading_exponents, reverse=True)


if __name__ == '__main__':
    from sbmath.tree import Variable, Value

    x = Variable('x')
    y = Variable('y')

    s = [
        Polynomial({(5, 2): Value(+1), (0, 1): Value(+2), (0, 0): Value(-4)}, [x, y]),
        Polynomial({(1, 1): Value(+7), (0, 1): Value(-8), (0, 0): Value(+0)}, [x, y]),
    ]

    b = grobner_bases(s)
    for p in b:
        print(p)

    # a = Polynomial({(2, 1): Value(1), (1, 0): Value(-3), (0, 0): Value(2)}, [x, y])
    # b = Polynomial({(1, 0): Value(1), (0, 0): Value(-1)}, [x, y])

    # print(a.remainder([b]))

    # print(exponents_lexicographic_max((1, 0), (1, 0)))
