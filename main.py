import matplotlib.pyplot as plt
from inflow import Inflows, Inflow
from mass_flow import calc_mass_flow, calc_velocity, V, make_velocity_interpolation, plot_mass_flow_velocity
from wellborn import WellParams
from typing import List
import numpy as np

J = 101
well_params = WellParams(r=10, rho=1000, pressure_bar=20, A=0, B=10)
h = (well_params.B - well_params.A) / (J - 1)
q_ref = [1 / 3600, 3 / 3600, 2 / 3600]
inflows_ref = Inflows([Inflow(9 - h / 2, q_ref[0]), Inflow(5 - h / 2, q_ref[1]), Inflow(3 - h / 2, q_ref[2])])
_z = np.linspace(well_params.A, well_params.B, num=100)
v = V(make_velocity_interpolation(list(_z), inflows_ref, well_params))
G_ref = calc_mass_flow(inflows_ref, well_params.A)
grid = np.linspace(well_params.A, well_params.B, num=J)
idx1 = np.where(grid >= 4)[0][0]
idx2 = np.where(grid >= 7)[0][0]
WEIGHT = 200


def delta(p1, p2, idx):
    return (p1[idx] - p2[idx]) ** 2


def P_func(P):
    return (delta(P, P_sol_ref, 0) + delta(P, P_sol_ref, idx1) + delta(P, P_sol_ref, idx2))


def G_func(G):
    return (G_ref - G) ** 2


def Func(P, G):
    return WEIGHT ** 2 * P_func(P) + G_func(G)


def sol_forward(q: List[float]):
    inflows = Inflows([Inflow(9 - h / 2, q[0]), Inflow(5 - h / 2, q[1]), Inflow(3 - h / 2, q[2])])
    P_sol = np.zeros(J)
    P_sol[-1] = well_params.BHP

    G_sol = np.zeros(J)
    G_sol[-1] = 0

    for j in reversed(range(0, J - 1)):
        G_sol[j] = G_sol[j + 1] + inflows(h * j)
        P_sol[j] = P_sol[j + 1] + (1 / well_params.S) * \
                   (G_sol[j + 1] * v(h * (j + 1)) - G_sol[j] * v(h * j)) - h * well_params.rho * well_params.g

    return G_sol, P_sol
    # print(f'F: {abs(G_sol[0] - G_ref)}, G_ref: {G_ref}, G_sol: {G_sol[0]}')


G_sol_reg, P_sol_ref = sol_forward(q_ref)


def grad_calc(q, delta=1e-3):
    q = np.array(q)
    points = []
    functional = []
    for i in range(q.shape[0]):
        u = np.zeros(len(q))
        u[i] = delta
        points.append(q + u)
        points.append(q - u)
    for i in range(len(points)):
        functional.append(10 * abs(sol_forward(points[i])[0][0] - G_ref))
    grad = np.zeros(q.shape)
    for i in range(grad.shape[0]):
        grad[i] = (functional[i * 2] - functional[i * 2 + 1]) / (2 * delta)
    return grad


def minimize(q):
    q_ = []
    F_ = []
    G_sol, P_sol = sol_forward(q)
    F_0 = Func(P_sol, G_sol[0])
    F = F_0
    F__ = F_0
    q_.append(q)
    F_.append(F / F__)
    k = 0
    while F / F__ >= 1e-5 and k < 100:
        k += 1
        grad = grad_calc(q, delta=1e-5)
        q -= (1 / 3600) * grad * 0.001
        q_.append(q)
        G_sol, P_sol = sol_forward(q)
        F_0 = F
        F = Func(P_sol, G_sol[0])
        F_.append(F / F__)
    print(WEIGHT ** 2 * P_func(P_sol) / F__, G_func(G_sol[0]) / F__)
    return q, G_sol, P_sol, q_, F_


# # plt.scatter(G_sol, grid, label='G_sol', lw = 7)
# # plt.scatter([calc_mass_flow(inflows, t) for t in grid], grid, label='G_ref', lw=2)
# # plt.scatter(P_sol, grid)
# # plt.grid()
# # plt.gca().invert_yaxis()
# # plt.legend()
# # plt.show()
# # print(G_sol * 10)
# # print(G_ref)
#
q_0 = [1.25 / 3600, 3.5 / 3600, 2.5 / 3600]
G_, P_ = sol_forward(q_0)
q, G_sol, P_sol, q_, F_ = minimize(q_0)
plt.scatter(G_sol, grid, label='G_sol', lw=9)
plt.scatter(G_, grid, label='G_init', lw=7)
plt.scatter([calc_mass_flow(inflows_ref, t) for t in grid], grid, label='G_ref', lw=2)
plt.grid()
plt.gca().invert_yaxis()
plt.legend()
plt.show()

plt.scatter(list(range(len(q))), q, label='sol', lw=9)
plt.scatter(list(range(len(q))), q_0, label='init', lw=7)
plt.scatter(list(range(len(q))), q_ref, label='ref', lw=2)
plt.grid()
plt.gca().invert_yaxis()
plt.legend()
plt.show()

for i in range(len(q_[0])):
    plt.plot(range(len(q_)), [q_[j][i] for j in range(len(q_))], '-o', label=f'$q_{i}$')
    plt.plot(range(len(q_)), q_ref[i] * np.ones(len(q_)), label=f'q_ref[{i}]')

    # plt.plot(range(len(q_)), q_ref[i], label=f'q_ref[{i}]')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(range(len(F_)), np.log(F_), '-o')
plt.legend()
plt.grid(True)
plt.show()

# plot_mass_flow_velocity(well_params.A, well_params.B, inflows_ref, well_params)


plt.plot(P_sol, grid, '-o', label='solution', lw=5)
plt.plot(P_sol_ref, grid, '-o', label='reference', lw=1)
plt.gca().invert_yaxis()
plt.legend()
plt.grid(True)
plt.show()

# print(P_sol[idx1], P_sol_ref[idx1])
# print(np.sqrt(delta(P_sol, P_sol_ref, 0) + delta(P_sol, P_sol_ref, idx1) + delta(P_sol, P_sol_ref, idx2)))
