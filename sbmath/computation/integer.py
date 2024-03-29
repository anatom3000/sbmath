from __future__ import annotations

from math import sqrt
from collections import defaultdict


def prime_factorize(n: int) -> dict[int, int]:
    factors: dict[int, int] = defaultdict(int)

    for i in range(2, n+1):
        while n % i == 0:
            n //= i
            factors[i] += 1

    return dict(factors)


def decompose_sqrt(n: int) -> tuple[int, int]:
    outside = 1
    inside = 1

    factorized = prime_factorize(n)
    for p, k in factorized.items():
        outside *= p ** (k // 2)
        if k % 2 != 0:
            inside *= p

    return outside, inside  # we usually write a√b (a is outside, b is inside), hence the order

__all__ = ['prime_factorize', 'decompose_sqrt']
