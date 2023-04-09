from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from wellborn import WellParams, dims
from numpy.typing import NDArray
from mass_flow import V
from inflow import Inflows, Inflow
from forward import calculate_forward_problem, residual, debit_residual, pressure_residual
from my_math import float_equal
from my_math import Transformation, f_forward
from my_math import calc_weights_matrix, calc_forward_problem_matrix, calc_change_dim_matrix
import shutil
import scipy
from enum import Enum
from mass_flow import calc_mass_flow, calc_velocity


class GradMethods(Enum):
    adjoint = 0
    usual = 1


def grad_calc(
        q: List[float],
        well_params: WellParams,
        F_0: float,
        J: int,
        v: V,
        P_ref: List[float],
        G_ref: List[float],
        weight: NDArray,
        grid_ref: NDArray,
        layers: List[float],
        transformation: Transformation,
        delta=1e-5,
) -> NDArray:
    """
    Calculate functional gradient.

    :param q:
    :param well_params:
    :param F_0:
    :param J:
    :param v:
    :param P_ref:
    :param G_ref:
    :param weight:
    :param grid:
    :param grid_ref:
    :param layers:
    :param delta:
    :return:
    """
    q = np.array(q)
    points = []
    functional = []
    for i in range(q.size):
        u = np.zeros(q.size)
        u[i] = delta
        points.append(q + u)
        points.append(q - u)
    for i in range(len(points)):
        P_sol, G_sol, grid, *_ = calculate_forward_problem(well_params, J, layers, points[i], v, None, transformation)
        F = residual(P_ref, G_ref, P_sol, G_sol, weight)
        functional.append(F)
    grad = np.zeros(q.size)
    for i in range(grad.size):
        grad[i] = (functional[i * 2] - functional[i * 2 + 1]) / (2 * delta)
    return np.array(grad)


class Residual:
    debit: list = []
    pressure: list = []
    h: list = []

    def show(self):
        fig, ax = plt.subplots(3, 1)
        ax1 = ax[0]
        ax2 = ax[1]
        ax3 = ax[2]

        def f(x, a, b):
            return a * x + b

        popt = curve_fit(f, np.log(np.array(self.h)), np.log(np.array(self.debit)))[0]
        ax1.scatter(np.log(np.array(self.h)), np.log(np.array(self.debit)), color='b',
                    label='debit')
        ax1.plot(np.log(np.array(self.h)), f(np.log(np.array(self.h)), *popt), color='r',
                 label=f' $y = {"%.1f" % popt[0]} \cdot x {"+" if popt[1] > 0 else "-"} {"%.1e" % abs(popt[1])}$')
        ax1.grid()
        ax1.legend()
        popt = curve_fit(f, np.log(np.array(self.h)), np.log(np.array(self.pressure)))[0]
        ax2.scatter(np.log(np.array(self.h)), np.log(np.array(self.pressure)), color='b',
                    label='pressure')
        ax2.plot(np.log(np.array(self.h)), f(np.log(np.array(self.h)), *popt), color='r',
                 label=f' $y = {"%.1f" % popt[0]} \cdot x {"+" if popt[1] > 0 else "-"} {"%.1e" % abs(popt[1])}$')
        ax2.grid()
        ax2.legend()
        self.debit = np.array(self.debit) / max(self.debit)
        self.pressure = np.array(self.pressure) / max(self.pressure)
        popt = curve_fit(f, np.log(np.array(self.h)), np.log(np.array(self.debit) + np.array(self.pressure)))[0]
        ax3.scatter(np.log(np.array(self.h)), np.log(np.array(self.debit) + np.array(self.pressure)), color='b',
                    label='debit + pressure')
        ax3.plot(np.log(np.array(self.h)), f(np.log(np.array(self.h)), *popt), color='r',
                 label=f' $y = {"%.1f" % popt[0]} \cdot x {"+" if popt[1] > 0 else "-"} {"%.1e" % abs(popt[1])}$')
        ax3.grid()
        ax3.legend()
        plt.savefig(r'images\residual.png', dpi=500)


def linear_interpolation(x_1, y_1, x_0, y_0, x):
    return y_0 + (x - x_0) * (y_1 - y_0) / (x_1 - x_0)


def from_integer_to_fractional(grid, data):
    h = (grid[1] - grid[0]) / 2
    grid = np.array(grid)
    rez = np.zeros(grid.size - 1)
    for i in range(grid.size - 1):
        rez[i] = linear_interpolation(grid[i], data[i], grid[i + 1], data[i + 1], grid[i] + h)
    return rez


def derivative_f_along_q_i_(layers, J, well_params, i, grid=None):
    h = (well_params.B - well_params.A) / (J - 1)
    if grid is None:
        grid = np.linspace(well_params.A, well_params.B, num=J)
    derivative = np.zeros(J * 2)
    z_i = layers[i]
    for i in range(len(grid)):
        z = grid[i] + h / 2
        if np.isclose(z, z_i):
            derivative[i] = 1
    return derivative


def derivative_f_along_q_i(layers, J, well_params, i, grid=None):
    h = (well_params.B - well_params.A) / (J - 1)
    if grid is None:
        grid = np.linspace(well_params.A, well_params.B, num=J)
    derivative = np.zeros(J * 2)
    z_i = layers[i]
    for i in range(len(grid)):
        z = grid[i] + h / 2
        if np.isclose(z, z_i):
            derivative[i] = h
    return derivative


def interpolate_2_half_grid(f):
    f_d = f[:int(f.size / 2)].copy()
    f_p = f[int(f.size / 2):].copy()
    f_d_res = np.zeros_like(f_d)
    f_p_res = np.zeros_like(f_p)
    f_d_res[0] = f_d[0]
    f_d_res[-1] = f_d[-1]
    f_p_res[0] = f_p[0]
    f_p_res[-1] = f_p[-1]
    for i in range(1, f_d_res.size - 1):
        f_d_res[i] = (f_d[i - 1] + f_d[i]) / 2
        f_p_res[i] = (f_p[i - 1] + f_p[i]) / 2
    return np.array([*list(f_d_res), *list(f_p_res)])


def calc_adjoint_solution(J, well_params, v, weight, G_ref, G_sol, P_ref, P_sol, K):
    A = calc_forward_problem_matrix(J, well_params, v)
    p = dims.p.SI(pressure_residual(P_ref, P_sol))
    d = dims.d.SI(debit_residual(G_ref, G_sol))
    u = np.array([*d, *p])
    B = calc_change_dim_matrix(int(u.size / 2), K)
    W = calc_weights_matrix(weight, K)
    f = B.T @ W @ B @ u / np.sqrt(2)  # деление на корень - подгон
    adjoint_sol = scipy.linalg.solve(A.T, f)

    # f_f = f_forward(well_params, J, np.array([3.5, 5.5]), np.array([500, 400]))
    # forward_solution = scipy.linalg.solve(A, f_f)
    # pres = dims.p.Programm(forward_solution[J:])
    # deb = dims.d.Programm(forward_solution[:J])
    # half_adjoint = interpolate_2_half_grid(adjoint_sol)
    # layers = np.array([3.5, 5.5])
    # q_ref = np.array([500, 400])
    # inflows_ref = Inflows([Inflow(layers[i], q_ref[i]) for i in range(len(layers))])
    # def calc_analytic_P(well_params: WellParams, z, inflows: Inflows):
    #     tmp1 = well_params.rho * well_params.g * z
    #     tmp2 = dims.p.SI(well_params.BHP) - well_params.rho * well_params.g * well_params.B
    #     tmp3 = (1 / well_params.S) * dims.d.SI(calc_mass_flow(inflows, z)) * dims.v.SI(
    #         calc_velocity(inflows, z, well_params))
    #     return dims.p.Programm(tmp1 + tmp2 + tmp3)
    # h = (well_params.B - well_params.A) / (J - 1)
    # P_analit = np.zeros(np.array(P_sol).size)
    # for i in range(P_analit.size):
    #     P_analit[i] = calc_analytic_P(well_params, h * i, inflows_ref)
    #
    #
    # plt.plot(G_sol, label='my', lw=5)
    # plt.plot(deb, label='matrix')
    # plt.legend()
    # plt.show()
    # plt.plot(P_sol, label='my', lw=5)
    # plt.plot(pres, label='matrix')
    # plt.plot(P_analit, label='analit')
    # plt.legend()
    # plt.show()
    return adjoint_sol


def calculate_i_gradient(layers, J, well_params, i, v, grid=None):
    df_dq = derivative_f_along_q_i(layers=layers, J=J, well_params=well_params, i=i, grid=grid)
    return df_dq @ v


def grad_calc_adjoint(J, well_params, v, weight, G_ref, G_sol, P_ref, P_sol, layers, K):
    adjoint_sol = calc_adjoint_solution(J=J, well_params=well_params, v=v, weight=weight, G_ref=G_ref.copy(),
                                        G_sol=G_sol.copy(),
                                        P_ref=P_ref.copy(), P_sol=P_sol.copy(), K=K)
    layers = np.array(layers)
    g = np.zeros(layers.size)
    for i in range(layers.size):
        g[i] = calculate_i_gradient(layers, J, well_params, i, v=adjoint_sol.copy(), grid=None)
    return g


if __name__ == "__main__":
    well_params = WellParams(r=10, rho=1000, pressure_bar=1, A=0, B=10)
    J = 91
    weight = np.array([1e-6, 1e1])
    q_ref = np.array([60, 40]) * 10  # m^3 / day
    layers = np.array([3.5, 6.5])
    q_0 = np.array([80, 50]) * 10
    transformation = Transformation(q_0)
    q_ref = transformation.toNormalized(q_ref)
    P_ref, G_ref, grid_ref, v, inflows_ref, grid = calculate_forward_problem(well_params, 91, layers, q_ref, None,
                                                                             q_ref, transformation)
    P_sol, G_sol, grid, *_ = calculate_forward_problem(well_params, J, layers, q_0, v, None, transformation)

    adjoint_sol = calc_adjoint_solution(J=91, well_params=well_params, v=v, weight=weight, G_ref=G_ref, G_sol=G_sol,
                                        P_ref=P_ref, P_sol=P_sol, grid_ref_size=grid_ref.size)
    g_0 = calculate_i_gradient(layers, J, well_params, 0, v=adjoint_sol, grid=None)
    g_1 = calculate_i_gradient(layers, J, well_params, 1, v=adjoint_sol, grid=None)
    print(np.array([g_0, g_1]))
