from dataclasses import dataclass
from typing import List
from my_math import heaviside, delta, float_equal

@dataclass(frozen=True)
class Inflow:
    """inflow class:

    : param z[m]: float - position
    : param q[m^3 / s]: float - inflow
    """
    z: float  # m
    q: float  # m^3 / s

    def __call__(self, _z):
        return self.q if float_equal(_z, self.z) else 0

    def ref(self, _z):
        return self.q if _z <= self.z else 0


@dataclass(frozen=True)
class Inflows:
    Q: List[Inflow]

    def __call__(self, _z):
        return sum([self.Q[i](_z) for i in range(len(self.Q))])

    def ref(self, _z):
        return sum([self.Q[i].ref(_z) for i in range(len(self.Q))])

    def __len__(self):
        return len(self.Q)

