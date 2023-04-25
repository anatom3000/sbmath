from __future__ import annotations

from math import sqrt
from collections import defaultdict


def prime_factorize(n: int) -> dict[int, int]:
    factors: dict[int, int] = defaultdict(int)
    for i in range(2, int(sqrt(n))):
        while n % i == 0:
            n //= i
            factors[i] += 1

    return dict(factors)
