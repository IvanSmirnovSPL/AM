from inflow import Inflows
from wellborn import WellParams
from typing import List
from scipy.interpolate import CubicSpline
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from wellborn import dims


def calc_mass_flow(inflows: Inflows, z: float) -> float:
    return inflows.ref(z)


def calc_velocity(inflows: Inflows, z: float, params: WellParams) -> float:
    return dims.v.Programm(0.25 + (dims.d.SI(calc_mass_flow(inflows, z))) / params.S)


def make_velocity_interpolation(_z: List[float],
                                inflows: Inflows,
                                params: WellParams) -> CubicSpline:
    return CubicSpline(_z,
                       [calc_velocity(inflows,
                                      _z[i],
                                      params) for i in range(len(_z))])


@dataclass(frozen=True)
class V:
    spl: CubicSpline

    def __call__(self, _z):
        return self.spl(_z)


class V_:
    def __call__(self, _z):
        return -0.04 * _z + 0.6 - (_z - 10)*_z * 0.01


def plot_mass_flow_velocity(a: float, b: float, inflows: Inflows, params: WellParams):
    fig, ax = plt.subplots(figsize=(12, 4), nrows=1, ncols=2)

    depths = np.linspace(a, b, num=1000)
    for i in range(len(inflows)):
        ax[0].plot([0, inflows.Q[i].q], inflows.Q[i].z * np.ones(2), lw=5, c='orange', label='inflow')
        # ax[1].plot([0, inflows.Q[i].q], inflows.Q[i].z * np.ones(2), lw=5, c='orange', label='inflow')
    ax[0].plot([calc_mass_flow(inflows, d) for d in depths], depths, label='mass flow')
    ax[1].plot([calc_velocity(inflows, d, params) for d in depths], depths, label='velocity')

    ax[0].grid(True)
    ax[1].grid(True)
    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    ax[0].legend()
    ax[1].legend()
    ax[0].set_ylabel('depth, m')
    ax[0].set_xlabel('mass flow, kg / s')
    ax[1].set_xlabel('velocity, m / s')
    plt.show()
