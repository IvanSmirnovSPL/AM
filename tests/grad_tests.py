from support import grad_calc_adjoint, grad_calc
from wellborn import WellParams
from my_math import Transformation
from forward import calculate_forward_problem, residual
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

well_params = WellParams(r=10, rho=1000, pressure_bar=1, A=0, B=10)
J = 91
K = 10
weight = np.array([1e5] + [1e2] * K)
# weight = np.array([1] + [1] * K)
q_ref_main = np.array([60, 50, 55, 45]) * 10  # m^3 / day
layers = np.array([3.5, 5.5, 7.5, 8.5])

transformation = Transformation(q_ref_main)
P_ref, G_ref, grid_ref, v, inflows_ref, grid = calculate_forward_problem(well_params, J, layers,
                                                                         transformation.toNormalized(q_ref_main), None,
                                                                         transformation.toNormalized(q_ref_main),
                                                                         transformation)

grad_adjoint_method = []
grad_usual_method = []

q = []
func = []
for q1 in np.linspace(50, 70, num=10):
    for q2 in np.linspace(40, 60, num=10):
        q_0 = np.array([q1, q2, 50, 50]) * 10
        q.append(q_0)
        # transformation = Transformation(q_0)

        q_ref = transformation.toNormalized(q_ref_main)
        q_0 = transformation.toNormalized(q_0)


        P_sol_cur, G_sol_cur, grid, *_ = calculate_forward_problem(
            well_params,
            J,
            layers,
            q_0,
            v,
            None,
            transformation
        )
        grad_adjoint_method.append(
            grad_calc_adjoint(
                J=J,
                well_params=well_params,
                v=v, weight=weight,
                G_ref=G_ref,
                G_sol=G_sol_cur,
                P_ref=P_ref,
                P_sol=P_sol_cur,
                layers=layers,
                K=K)
        )
        grad_adjoint_method[-1] /= np.linalg.norm(grad_adjoint_method[-1])
        grad_usual_method.append(
            grad_calc(
                q_0,
                well_params,
                1,
                J,
                v,
                P_ref,
                G_ref,
                weight,
                grid_ref,
                list(layers),
                transformation)
        )
        func.append(residual(P_ref, G_ref, P_sol_cur, G_sol_cur, weight))
        grad_usual_method[-1] /= np.linalg.norm(grad_usual_method[-1])
        # g_1 = np.array([grad_adjoint_method[-1][0] * ( q_0[0] / q_0[1]), grad_adjoint_method[-1][1]])
        # g_1 = g_1 / np.linalg.norm(g_1)
        # g_2 = np.array([grad_adjoint_method[-1][0] / (q_0[0] / q_0[1]), grad_adjoint_method[-1][1]])
        # g_2 = g_2 / np.linalg.norm(g_2)
        # g_3 = np.array([grad_adjoint_method[-1][0] * (layers[0] / layers[1]), grad_adjoint_method[-1][1]])
        # g_3 = g_3 / np.linalg.norm(g_3)
        # g_4 = np.array([grad_adjoint_method[-1][0] / (layers[0] / layers[1]), grad_adjoint_method[-1][1]])
        # g_4 = g_4 / np.linalg.norm(g_4)
        # print(
        #     grad_adjoint_method[-1][0] / grad_adjoint_method[-1][1],
        #     grad_usual_method[-1][0] / grad_usual_method[-1][1],
        #     g_1[0] / g_1[1],
        #     g_2[0] / g_2[1],
        #     g_3[0] / g_3[1],
        #     g_4[0] / g_4[1],
        # )


for idx, q_ in enumerate(q):
    print(idx, q_, func[idx])

plt.plot(func, '-o')
plt.show()

plt.cla()
plt.clf()
fig = plt.figure(figsize=(10, 5))


def draw_raw(adjoint: NDArray, usual: NDArray, num: int, ax: fig.subplots):
    ax1 = ax[num][0]
    ax2 = ax[num][1]
    ax3 = ax[num][2]
    ax4 = ax[num][3]
    ax5 = ax[num][4]
    ax1.plot(adjoint, '--o', c='r', label=f'$adm_{num}$')
    ax2.plot(usual, '-o', c='r', label=f'$g_{num}$')
    ax3.plot(adjoint, '-', c='k', label=f'$adm_{num}$', lw=5)
    ax3.plot(usual, '-', c='r', label=f'$g_{num}$')
    delta = np.zeros(usual.size)
    delta_rel = np.zeros(usual.size)
    for i in range(usual.size):
        delta[i] = abs(adjoint[i] - usual[i])
        delta_rel[i] = delta[i] / abs(usual[i])
        if delta[i] > 0.1:
            if abs(usual[i]) < 1e-6 or usual[i] * adjoint[i] < 0:
                delta[i] = 0
            else:
                print('ATTENTION: ', adjoint[i], usual[i], i, delta[i], num)
        if delta_rel[i] > 0.1:
            if abs(usual[i]) < 1e-6 or usual[i] * adjoint[i] < 0:
                delta_rel[i] = 0
            else:
                print('ATTENTION RELATIVE: ', adjoint[i], usual[i], i, delta_rel[i], num)
    ax4.plot(delta, '-', c='g', label='error')
    ax5.plot(delta_rel, '-', c='g', label='relative error')
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    ax5.grid(True)
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()


ax = fig.subplots(layers.size, 5)
for i in range(layers.size):
    draw_raw(np.array(grad_adjoint_method)[:, i], np.array(grad_usual_method)[:, i], i, ax)

plt.legend()
plt.show()
