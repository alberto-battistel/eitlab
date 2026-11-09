"""Current patterns and protocols, independent of NGSolve and reconstruction."""
from collections.abc import Iterable
from dataclasses import dataclass, field
from types import MappingProxyType
import math


@dataclass(frozen=True)
class CurrentPattern:
    currents: dict[str, float]
    name: str = ""
    tolerance: float = 1e-12

    def __post_init__(self):
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("Current tolerance must be finite and nonnegative")
        values = {}
        for name, current in self.currents.items():
            value = complex(current)
            if not math.isfinite(value.real) or not math.isfinite(value.imag):
                raise ValueError("Currents must be finite")
            values[name] = value if value.imag else float(value.real)
        total = sum(values.values())
        if abs(total) > self.tolerance:
            raise ValueError(f"Net injected current must be zero, got {total}")
        object.__setattr__(self, "currents", MappingProxyType(values))

    def current_on(self, electrode_name):
        return self.currents.get(electrode_name, 0.0)

    __getitem__ = current_on

    def as_dict(self):
        return dict(self.currents)

    def __str__(self):
        terms = ", ".join(f"{name}: {value:+g}" for name, value in self.currents.items())
        return f"{self.name or 'CurrentPattern'} ({terms})"

    __repr__ = __str__


@dataclass
class CurrentProtocol:
    patterns: list[CurrentPattern] | CurrentPattern = field(default_factory=list)
    name: str = "Protocol"

    def __post_init__(self):
        self.patterns = [self.patterns] if isinstance(self.patterns, CurrentPattern) else list(self.patterns)
        if any(not isinstance(p, CurrentPattern) for p in self.patterns):
            raise TypeError("Protocols contain CurrentPattern instances")

    def append(self, pattern):
        if not isinstance(pattern, CurrentPattern):
            raise TypeError("Expected CurrentPattern")
        self.patterns.append(pattern)

    add = append

    def extend(self, patterns):
        values = list(patterns)
        if any(not isinstance(p, CurrentPattern) for p in values):
            raise TypeError("Expected CurrentPattern instances")
        self.patterns.extend(values)

    def pop(self, index=-1):
        return self.patterns.pop(index)

    def remove(self, *patterns):
        for pattern in patterns:
            if pattern in self.patterns:
                self.patterns.remove(pattern)

    def __iadd__(self, other):
        if isinstance(other, CurrentPattern):
            self.append(other)
        elif isinstance(other, Iterable):
            self.extend(other)
        else:
            return NotImplemented
        self.name = "Mixed"
        return self

    def __isub__(self, other):
        if isinstance(other, CurrentPattern):
            self.remove(other)
        elif isinstance(other, Iterable):
            self.remove(*other)
        else:
            return NotImplemented
        self.name = "Mixed"
        return self

    def __sub__(self, other):
        result = CurrentProtocol(self.patterns, self.name)
        result -= other
        return result

    def __iter__(self):
        return iter(self.patterns)

    def __getitem__(self, index):
        return self.patterns[index]

    def __len__(self):
        return len(self.patterns)

    def __str__(self):
        return f"{self.name} [{len(self)} patterns]"

    __repr__ = __str__


class ProtocolFactory:
    @staticmethod
    def adjacent(electrode_names, amplitude=1.0):
        names = tuple(electrode_names)
        if len(names) < 2 or len(set(names)) != len(names):
            raise ValueError("Need at least two unique electrodes")
        return CurrentProtocol([
            CurrentPattern({name: amplitude, names[(i + 1) % len(names)]: -amplitude})
            for i, name in enumerate(names)], name="adjacent")

    @staticmethod
    def jump_and_skip(electrode_names, first_electrode, amplitude=1.0, jump=1, skip=0):
        """Visit source indices with positive stride jump; sink is 1+skip ahead."""
        names = tuple(electrode_names)
        if len(names) < 2 or len(set(names)) != len(names):
            raise ValueError("Need at least two unique electrodes")
        if not isinstance(jump, int) or jump < 1:
            raise ValueError("jump must be a positive source-index stride")
        if not isinstance(skip, int) or not 0 <= skip < len(names) - 1:
            raise ValueError("skip must select a different sink electrode")
        first = names.index(first_electrode)
        return CurrentProtocol([
            CurrentPattern({names[(i + first) % len(names)]: amplitude,
                            names[(i + first + 1 + skip) % len(names)]: -amplitude})
            for i in range(0, len(names), jump)], name="jump_and_skip")
