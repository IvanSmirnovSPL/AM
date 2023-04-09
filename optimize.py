import os

from forward import calculate_forward_problem, residual
from wellborn import WellParams
from mass_flow import V
from numpy.typing import NDArray
from typing import List, Tuple
import numpy as np
from vizual import makeVizualData, show_gradients
import matplotlib.pyplot as plt
from my_math import Transformation
from wellborn import dims
from support import grad_calc_adjoint, grad_calc, GradMethods
import shutil
from pathlib import Path
from enum import Enum
from vizual import vizual_information

from vizual import makeVizualData

GRAD_METHOD = GradMethods.adjoint

def minimize(
        q_cur: NDArray,
        q_ref: NDArray,
        well_params: WellParams,
        J: int,
        weight: NDArray,
        layers: NDArray,
        transformation: Transformation, res_path: Path, K: int) -> Tuple:
    """
    Function to solve inverse problem.

    :param h: grid step
    :param layers: positions of inflows.
    :param v: velocity.
    :param weight: weights.
    :param grid: solver grid.
    :param grid_ref: reference data grid.
    :param P_ref: wellborn pressure reference.
    :param G_ref: debit reference.
    :param q_sol: initial guess.
    :return:
    """
    q = []
    g_norm = []
    q_adm = [q_0]
    alpha = []
    g = []
    functional = []
    G_grad = [0.]

    P_sol, G_sol, grid, *_ = calculate_forward_problem(well_params, J, layers, q_0, v, None, transformation)
    F = residual(P_ref, G_ref, P_sol, G_sol, weight)
    P_init, G_init = P_sol, G_sol
    F_0 = F
    print(f'F_0: {F_0}')
    q.append(q_cur)
    functional.append(F / F_0)

    k = 0
    while functional[-1] >= 1e-3 and k < 200:
        k += 1

        if GRAD_METHOD == GradMethods.adjoint:
            grad = grad_calc_adjoint(J, well_params, v, weight, G_ref, G_sol, P_ref, P_sol,
                                     layers, K)
        elif GRAD_METHOD == GradMethods.usual:
            grad = grad_calc(q_cur, well_params, F_0, J, v, P_ref, G_ref, weight, grid_ref, layers, transformation)
        else:
            print('ERROR GRAD METHOD')
            exit()

        g_norm.append(np.linalg.norm(grad))
        g.append(grad / np.linalg.norm(grad))

        # RMSProp
        gamma = 0.5
        epsilon = 2
        G_grad.append(gamma * G_grad[-1] + (1 - gamma) * g[-1].T @ g[-1])
        alpha.append(1e-3 / (np.sqrt(G_grad[-1] + epsilon)))

        q_cur = q_cur - g[-1] * alpha[-1]

        P_sol, G_sol, grid, *_ = calculate_forward_problem(well_params, J, layers, q_cur, v, None, transformation)
        F = residual(P_ref, G_ref, P_sol, G_sol, weight)
        q.append(q_cur)
        functional.append(F / F_0)
    title = ''
    for j in range(q_0.size):
        title += f'{transformation.toPhysical(q_0[j])}_'
    plt.clf()
    if len(g) > 0:
        fig = plt.figure(figsize=(10, 5))
        for j in range(g[0].size):
            plt.plot(np.array(g)[:, j], '-o', label=f'$g_{j}$')
        # plt.plot(np.array(g)[:, 1], '-o', c='b', label='$g_1$')
        plt.grid(True)
        plt.legend()
        # plt.show()
        plt.title(f'{title}')
        plt.savefig(Path(res_path, f'grad_{title}.png'),
                    dpi=250)
    if len(g_norm) > 0:
        fig = plt.figure(figsize=(10, 5))
        print(g_norm[:2])
        plt.plot(np.array(np.log(g_norm)), '-o', c='r', label=f'$g_norm$')
        # plt.plot(np.array(g)[:, 1], '-o', c='b', label='$g_1$')
        plt.grid(True)
        plt.legend()
        # plt.show()
        plt.title(f'{title}')
        plt.savefig(Path(res_path, f'grad_norm_{title}.png'),
                    dpi=250)

    plt.clf()
    plt.plot(alpha)
    plt.grid(True)
    plt.savefig(Path(res_path, f'alpha_{title}.png'),
                dpi=250)
    return q_cur, functional, q, G_sol, P_sol, G_init, P_init


if __name__ == "__main__":
    RES_DIRECTORY = ''
    if GRAD_METHOD == GradMethods.adjoint:
        RES_DIRECTORY = 'optimization_adjoint'
    elif GRAD_METHOD == GradMethods.usual:
        RES_DIRECTORY = 'optimization_usual'
    else:
        print('ERROR GRAD METHOD')
        exit()

    res_path = Path(Path.cwd(), RES_DIRECTORY)
    shutil.rmtree(res_path, ignore_errors=True)
    os.mkdir(res_path)
    well_params = WellParams(r=10, rho=1000, pressure_bar=1, A=0, B=10)
    J = 91
    K = 10
    weight = np.array([1e5] + [1e2] * K)
    q_ref_main = np.array([30, 50, 60]) * 10  # m^3 / day
    layers = np.array([4.5, 6.5, 8.5])
    transformation = Transformation(q_ref_main)

    for q1 in np.linspace(25, 35, num=2):
        for q2 in np.linspace(45, 55, num=2):
            q_0 = np.array([q1, q2, 65]) * 10
            q_ref = transformation.toNormalized(q_ref_main)
            q_0 = transformation.toNormalized(q_0)

            P_ref, G_ref, grid_ref, v, inflows_ref, grid = calculate_forward_problem(well_params, 91, layers, q_ref,
                                                                                     None,
                                                                                     q_ref, transformation)

            # show_gradients(P_ref, G_ref, grid_ref, v, J, layers, well_params, weight, [np.linspace(700, 500, num=11), np.linspace(500, 300, num=11)])
            # exit()

            q_cur, functional, q, G_sol, P_sol, G_init, P_init = minimize(q_0, None, well_params, J, weight,
                                                                          layers,
                                                                          transformation, res_path, K=K)
            vizual_information(transformation, q_0, q_ref, P_ref, G_ref, grid_ref, grid, q_cur, functional, q, G_sol,
                               P_sol, G_init, P_init, res_path)
