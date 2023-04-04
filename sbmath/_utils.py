import sys
from collections import defaultdict
from collections.abc import Hashable, Iterable
from typing import TypeVar


DEBUG = True
DEBUG_FLAGS = (
    # "repl",
    # "match",
    # "match_adv_wc",
    # "match_wc",
    # "reduce",
    "reduce_func",
    # "diff",
    # "misc"
)
DEBUG_INDENT = 0


def inc_indent():
    global DEBUG_INDENT
    DEBUG_INDENT += 1


def dec_indent():
    global DEBUG_INDENT
    DEBUG_INDENT -= 1


def debug(data, /, flag=''):
    if DEBUG and flag in DEBUG_FLAGS:
        print("| " * DEBUG_INDENT, data, file=sys.stderr, sep='')


K = TypeVar("K", bound=Hashable)
V = TypeVar("V", bound=Hashable)


class BiMultiDict:
    """
    A data structure similar to a dict but you can retrieve the keys holding a specific value and vice versa.
    If we associate:
        - A with Y (my_bmd.add('A', 'Y'))
        - B with Z (my_bmd.add('B', 'Z'))
        - C with X (my_bmd.add('C', 'X'))
        - C with Y (my_bmd.add('C', 'Y'))
    We can, using the mapping, know that:
        - the key A is associated with Y
        - the key C is associated with both X and Y
        - the value Z is associated with B
        - the value Y is associated with both A and C

    The concepts of keys and values are symmetrical.
    All objects (keys and values) must be hashable to be stored.
    When using bracket syntax:
        - `my_bmd[key]` gets values associated with `key`
        - `my_bmd[key] = value` associates `key` with `value` (does not overwrite previous relationships with `key`)
        - `del my_two_way[key]` will remove all relationships with `key`
    """

    def __str__(self):
        k2v = {k: self.get_from_key(k) for k in self.keys()}
        v2k = {v: self.get_from_value(v) for v in self.values()}

        return f"{k2v} <=> {v2k}"

    def __repr__(self):
        return str(self)

    def __init__(self, source: dict[K, V] = None):
        self._data: dict[K, V] = {}
        self._keys_to_values: defaultdict = defaultdict(list)
        self._values_to_keys: defaultdict = defaultdict(list)

        if source is not None:
            for k, v in source.items():
                self.add(k, v)

    def add(self, key, value) -> None:
        key_hash = hash(key)
        value_hash = hash(value)

        self._data[key_hash] = key
        self._data[value_hash] = value

        self._keys_to_values[key_hash].append(value_hash)
        self._values_to_keys[value_hash].append(key_hash)

    def get_from_key(self, key) -> list[V]:
        h = hash(key)
        if h not in self._keys_to_values.keys():
            return []

        return [self._data[h] for h in self._keys_to_values[h]]

    def get_from_value(self, value) -> list[K]:
        h = hash(value)
        if h not in self._values_to_keys.keys():
            return []

        return [self._data[h] for h in self._values_to_keys[h]]

    def try_remove_key(self, key) -> bool:
        h = hash(key)
        if h not in self._keys_to_values.keys():
            return False

        del self._data[h]
        del self._keys_to_values[h]
        for v in self._values_to_keys.keys():
            if h in self._values_to_keys[v]:
                self._values_to_keys[v].remove(h)

        return True

    def remove_key(self, key) -> None:
        h = hash(key)
        if h not in self._keys_to_values.keys():
            raise KeyError(f"unknown key {key}")

        del self._data[h]
        del self._keys_to_values[h]
        for v in self._values_to_keys.keys():
            if h in self._values_to_keys[v]:
                self._values_to_keys[v].remove(h)

    def try_remove_value(self, value) -> bool:
        h = hash(value)
        if h not in self._values_to_keys.keys():
            return False

        del self._data[h]
        del self._values_to_keys[h]
        for v in self._keys_to_values.keys():
            if h in self._keys_to_values[v]:
                self._keys_to_values[v].remove(h)

        return True

    def remove_value(self, value) -> None:
        h = hash(value)
        if h not in self._values_to_keys.keys():
            raise KeyError(f"unknown value {value}")

        del self._data[h]
        del self._values_to_keys[h]
        for v in self._keys_to_values.keys():
            if h in self._keys_to_values[v]:
                self._keys_to_values[v].remove(h)

    def has_relation(self, key, value) -> bool:
        hk = hash(key)
        hv = hash(value)

        if hk not in self._keys_to_values.keys():
            raise KeyError(f"unknown key {key}")
        if hv not in self._values_to_keys.keys():
            raise KeyError(f"unknown value {value}")

        return hv in self._keys_to_values[hk] and hk in self._values_to_keys[hv]

    def remove_relationship(self, key, value) -> None:
        hk = hash(key)
        hv = hash(value)

        if hk not in self._keys_to_values.keys():
            raise KeyError(f"unknown key {key}")
        if hv not in self._values_to_keys.keys():
            raise KeyError(f"unknown value {value}")

        if not (hv in self._keys_to_values[hk] and hk in self._values_to_keys[hv]):
            raise KeyError(f"{key} has no relation with {value}")

        self._keys_to_values[hk].remove(hv)
        self._values_to_keys[hv].remove(hk)

    def __getitem__(self, item) -> list[V]:
        return self.get_from_key(item)

    def __setitem__(self, key, value):
        return self.add(key, value)

    def __delitem__(self, key):
        return self.remove_key(key)

    def keys(self) -> Iterable[K]:
        return (self._data[h] for h in self._keys_to_values.keys())

    def values(self) -> Iterable[V]:
        return (self._data[h] for h in self._values_to_keys.keys())


if __name__ == "__main__":
    a = BiMultiDict()
    a["a"] = "y"
    a["b"] = "z"
    a["c"] = "x"
    a["c"] = "y"

    # from pprint import pp
    # pp(a.__dict__, indent=2)

    print("A", a.get_from_key("a"))
    print("B", a.get_from_key("b"))
    print("C", a.get_from_key("c"))
    print("X", a.get_from_value("x"))
    print("Y", a.get_from_value("y"))
    print("Z", a.get_from_value("z"))
    print(list(a.keys()), list(a.values()))

    print("---")
    a.remove_relationship("c", "x")

    # pp(a.__dict__, indent=2)
    print("A", a.get_from_key("a"))
    print("B", a.get_from_key("b"))
    print("C", a.get_from_key("c"))
    print("X", a.get_from_value("x"))
    print("Y", a.get_from_value("y"))
    print("Z", a.get_from_value("z"))
    print(list(a.keys()), list(a.values()))
