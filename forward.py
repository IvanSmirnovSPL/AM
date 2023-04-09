import matplotlib.pyplot as plt
from inflow import Inflows, Inflow
from mass_flow import calc_mass_flow, calc_velocity, V, V_, make_velocity_interpolation, plot_mass_flow_velocity
from wellborn import WellParams, dims
from typing import List
import numpy as np
from scipy.optimize import curve_fit
from numpy.typing import NDArray
# from vizual import makeVizualData
from my_math import interpolate_coefs, interpolate
from my_math import Transformation
from my_math import calc_weights_matrix, calc_forward_problem_matrix, calc_change_dim_matrix


def sol_forward(q: List[float], layers: NDArray, well_params: WellParams, J: int, v: V):
    h = (well_params.B - well_params.A) / (J - 1)
    assert len(layers) == len(q)
    inflows = Inflows([Inflow(layers[i], q[i] / h) for i in range(len(layers))])

    P_sol = np.zeros(J)
    P_sol[-1] = well_params.BHP

    G_sol = np.zeros(J)
    G_sol[-1] = 0

    for j in reversed(range(0, J - 1)):
        a = inflows(h * (j + 1 / 2))
        G_sol[j] = G_sol[j + 1] + h * inflows(h * (j + 1 / 2))
        P_sol[j] = dims.p.Programm(dims.p.SI(P_sol[j + 1]) - (1 / well_params.S) * \
                                   (dims.d.SI(G_sol[j + 1]) * dims.v.SI(v(h * (j + 1))) - dims.d.SI(
                                       G_sol[j]) * dims.v.SI(v(
                                       h * j))) - h * well_params.rho * well_params.g)

    return G_sol, P_sol


def calc_analytic_P(well_params: WellParams, z, inflows: Inflows):
    tmp1 = well_params.rho * well_params.g * z
    tmp2 = dims.p.SI(well_params.BHP) - well_params.rho * well_params.g * well_params.B
    tmp3 = (1 / well_params.S) * dims.d.SI(calc_mass_flow(inflows, z)) * dims.v.SI(
        calc_velocity(inflows, z, well_params))
    return dims.p.Programm(tmp1 + tmp2 + tmp3)


def calc_analytic_G(z, inflows: Inflows):
    return calc_mass_flow(inflows, z)


def pre_proc(
        well_params: WellParams,
        q_0: NDArray,
        q_ref: NDArray,
        J: int,
        layers, ) -> (
        float,
        List[float],
        NDArray,
        NDArray,
        List[float],
        List[float],
        List[float],
        List[float]):
    """
    Do preprocessing of forward solution.

    :param well_params: wellborn parameters.
    :param q_ref:  list of inflows values.
    :param J: nodes number.
    :return:
    """
    h = (well_params.B - well_params.A) / (J - 1)
    grid = np.linspace(well_params.A, well_params.B, num=J)
    if q_ref is not None:
        q_ref = np.array(q_ref)
        inflows_ref = Inflows([Inflow(layers[i], q_ref[i]) for i in range(len(layers))])
        _z = np.linspace(well_params.A, well_params.B, num=10000)
        v = V(make_velocity_interpolation(list(_z), inflows_ref, well_params))
        grid_ref = np.linspace(well_params.A, well_params.B, num=10001)
        P_ref = [calc_analytic_P(well_params, _z, inflows_ref) for _z in grid_ref]
        G_ref = [calc_analytic_G(_z, inflows_ref) for _z in grid_ref]
    else:
        q_ref = None
        inflows_ref = None
        _z = None
        v = None
        grid_ref = None
        P_ref = None
        G_ref = None

    return h, np.array(q_0), h, grid, inflows_ref, v, grid_ref, P_ref, G_ref


def calculate_forward_problem(well_params: WellParams,
                              J: int,
                              layers: NDArray,
                              q_0: NDArray,
                              v: V,
                              q_ref: NDArray,
                              transformation: Transformation,
                              ):
    q_0 = transformation.toPhysical(q_0)
    if q_ref is not None:
        q_ref = transformation.toPhysical(q_ref)
        h, q_sol, h, grid, inflows_ref, v, grid_ref, P_ref, G_ref = pre_proc(
            well_params,
            q_0,
            q_ref,
            J, layers)
        # G_sol, P_sol = sol_forward(q_sol, layers, well_params, J, v)
        return P_ref, G_ref, grid_ref, v, inflows_ref, grid
    else:
        h, q_sol, h, grid, *_ = pre_proc(
            well_params,
            q_0,
            None,
            J, layers)
        G_sol, P_sol = sol_forward(q_sol, layers, well_params, J, v)
        return P_sol, G_sol, grid, None


def debit_residual(G_ref, G_sol):
    k_ref = int((np.array(G_ref).size - 1) / (np.array(G_sol).size - 1))
    return G_ref[::k_ref] - G_sol


def pressure_residual(P_ref, P_sol):
    k_ref = int((np.array(P_ref).size - 1) / (np.array(P_sol).size - 1))
    return P_ref[::k_ref] - P_sol


def residual(P_ref, G_ref, P_sol, G_sol, weight):
    debit = debit_residual(G_ref, G_sol)
    pressure = pressure_residual(P_ref, P_sol)
    debit = dims.d.SI(debit)
    pressure = dims.p.SI(pressure)
    B = calc_change_dim_matrix(debit.size)
    W = calc_weights_matrix(weight)
    u = np.array([*debit, *pressure])
    return 0.5 * (B @ u).T @ W @ B @ u


if __name__ == "__main__":

    well_params = WellParams(r=10, rho=1000, pressure_bar=1, A=0, B=10)
    q_ref = np.array([60, 40]) * 10  # m^3 / day
    layers = np.array([3.5, 6.5])
    q_0 = np.array([65, 35]) * 10
    weight = np.array([1e-4, 1e13])

    P_ref, G_ref, grid_ref, v, inflows_ref, *_ = calculate_forward_problem(well_params, 91, layers, q_ref, None, q_ref)

    for J in [11, 31, 91]:
        P_sol, G_sol, grid, *_ = calculate_forward_problem(well_params, J, layers, q_0, v, None)
        debit, pressure = residual(P_ref, G_ref, P_sol, G_sol, J, grid_ref.size)
        print(debit * weight[0] + pressure * weight[1], debit, debit * weight[0], pressure, pressure * weight[1])

        makeVizualData(inflows_ref, well_params, grid_ref, grid, G_sol, G_ref, P_sol, P_ref, v, J)
