from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Pressure():
    """Programm is bar = 1e5 Pa"""

    @staticmethod
    def SI(x: NDArray):
        return x * 1e5

    @staticmethod
    def Programm(x: NDArray):
        return x / 1e5


dimPressure = Pressure()


@dataclass(frozen=True)
class Debit():
    """Programm is m^3 / day = (1 / 24 / 2600) m^3 / sec"""

    @staticmethod
    def SI(x: NDArray):
        return x / 24 / 3600

    @staticmethod
    def Programm(x: NDArray):
        return x * 24 * 3600


dimDebit = Debit()


@dataclass(frozen=True)
class Velocity():
    """Programm is m / hour = (1 / 3600) m / sec"""

    @staticmethod
    def SI(x: NDArray):
        return x / 3600

    @staticmethod
    def Programm(x: NDArray):
        return x * 3600


dimVelocity = Velocity()


@dataclass(frozen=True)
class Dims:
    d: Debit = dimDebit
    p: Pressure = dimPressure
    v: Velocity = dimVelocity


dims = Dims()

@dataclass(frozen=True)
class WellParams:
    """wellborn parametrs:

    : param r[sm]: float - radius
    : param rho[kg / m^3]: float - density
    : param pressure_bar[Pa]: float - BHP
    : param A[m]: float - begin of well
    : param B[m]: float - end of well
    """
    r: float  # sm
    rho: float  # kg / m^3
    pressure_bar: float  # bar
    A: float
    B: float
    g: float = 0  # acceleration free fall

    @property
    def S(self):
        return (self.r / 100) ** 2 * np.pi

    @property
    def BHP(self):
        return self.pressure_bar

    @property
    def G_Bottom(self):
        return 0
