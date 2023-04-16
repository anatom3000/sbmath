from __future__ import annotations

from abc import ABC, abstractmethod


class Set(ABC):

    def union(self, other: Set) -> Set:
        return Union(self, other)

    def intersection(self, other: Set) -> Set:
        return Intersection(self, other)

    def issuperset(self, other: Set) -> bool:
        return other.issubset(self)

    def __contains__(self, item) -> bool:
        if isinstance(item, Set):
            return self.issuperset(item)

        return self.contains(item)

    def difference(self, other: Set) -> Set:
        return Difference(self, other)

    @abstractmethod
    def issubset(self, other: Set) -> bool:
        pass

    @abstractmethod
    def contains(self, other: Node) -> bool:
        pass

    @abstractmethod
    def __eq__(self, other: Node):
        pass


class Union(Set):
    def __eq__(self, other: Node):
        if not isinstance(other, Union):
            return False

        return all(any(sm == om for om in other.members) for sm in self.members)

    def issubset(self, other: Set) -> bool:
        pass

    def contains(self, other: Node) -> bool:
        pass

    def __init__(self, *members: Set):
        self.members = members


class Intersection(Set):
    def __eq__(self, other):
        if not isinstance(other, Intersection):
            return False

        return all(any(sm == om for om in other.members) for sm in self.members)

    def issubset(self, other: Set) -> bool:
        return all(m.issubset(other) for m in self.members)

    def contains(self, other: Node) -> bool:
        pass

    def __init__(self, *members: Set):
        self.members = members


class Difference(Set):
    def issubset(self, other: Set) -> bool:
        pass

    def contains(self, other: Node) -> bool:
        pass

    def __init__(self, base: Set, removed: Set):
        self.base = base
        self.removed = removed


class Interval(Set):
    def __init__(self, start: sbmath.tree.Node, end: sbmath.tree.Node, start_open: bool, end_open: bool):
        assert start.is_evaluable(), "bounds of inverval must be evaluable"
        assert end.is_evaluable(), "bounds of inverval must be evaluable"

        start_appr = start.approximate()
        end_appr = end.approximate()

        if start_appr > end_appr:
            start, end = end, start
            start_appr, end_appr = end_appr, start_appr
            start_open, end_open = end_open, start_open

        self.start = start
        self.end = end

        self._start_appr = start_appr
        self._end_appr = end_appr

    def union(self, other: Set) -> Set:
        if not isinstance(other, Interval):
            return super().union(other)

        if self == other:
            return self

        if self._start_appr < self._end_appr:
            return

    def intersection(self, other: Set) -> Set:
        pass

    def difference(self, other: Set) -> Set:
        pass

    def issubset(self, other: Set) -> bool:
        pass

    def contains(self, other: Node) -> bool:
        pass
