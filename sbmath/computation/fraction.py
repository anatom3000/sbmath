from __future__ import annotations

from math import gcd
from fractions import Fraction


def simplify_fraction(numerator: int, denominator: int) -> tuple[int, int]:
    common_divisor = gcd(numerator, denominator)
    return numerator // common_divisor, denominator // common_divisor


def float_to_fraction(x: float) -> tuple[int, int]:
    return Fraction(x).limit_denominator().as_integer_ratio()

__all__ = ['simplify_fraction', 'float_to_fraction']
