import matplotlib.pyplot as plt
import numpy as np

from mass_flow import calc_velocity
from forward import calculate_forward_problem, residual
from support import grad_calc, grad_calc_adjoint, GradMethods
from pathlib import Path
from wellborn import WellParams, dims
from my_math import Transformation


def vizual_information(transformation, q_0, q_ref, P_ref, G_ref, grid_ref, grid, q_cur, functional, q, G_sol,
                       P_sol, G_init, P_init, res_path):
    print(functional)

    # makeVizualData(inflows_ref, well_params, grid_ref, grid, G_sol, G_ref, P_sol, P_ref, v, J, path='images_opt')

    plt.clf()
    fig = plt.figure(figsize=(15, 15))
    ax = fig.subplots(2, 2)
    ax1 = ax[0][0]
    ax2 = ax[0][1]
    ax3 = ax[1][0]
    ax4 = ax[1][1]

    ax1.plot(G_sol, grid, label='G_sol', lw=9)
    ax1.plot(G_ref, grid_ref, label='G_ref', lw=5)
    ax1.plot(G_init, grid, label='G_init', lw=2)
    ax1.set_title('Mass flow')
    ax1.grid()
    ax1.invert_yaxis()
    ax1.legend()

    ax2.plot(P_sol, grid, label='P_sol', lw=9)
    ax2.plot(P_ref, grid_ref, label='P_ref', lw=5)
    ax2.plot(P_init, grid, label='P_init', lw=2)
    ax2.set_title('Wellborn pressure')
    ax2.grid()
    ax2.invert_yaxis()
    ax2.legend()

    for i in range(len(q[0])):
        q_i = [q[j][i] for j in range(len(q))]
        # q_adm_i = [q_adm[j][i] for j in range(len(q))]
        ax3.plot(q_i, '-o', label=f'$q_{i}$', lw=2)
        # ax3.plot(q_adm_i, '-o', label=f'$q_{i} adm$', lw=2)
        ax3.plot(np.ones(len(q)) * q_ref[i], label=f'$q_{i} ref$', lw=2)
    ax3.set_title('q')
    ax3.grid()
    ax3.legend()

    ax4.plot(np.log10(functional), '-o', label='functional', lw=2)
    ax4.set_title('Wellbohr functional')
    ax4.grid()
    ax4.legend()
    title = ''
    for j in range(q_0.size):
        title += f'{transformation.toPhysical(q_0[j])}_'
    plt.title(f'{title}')
    # plt.show()
    plt.savefig(
        Path(res_path, f'sol_{title}.png'),
        dpi=250)

    print(f'q_cur: {q_cur}', f'q_ref: {q_ref}', sep='\n')
    print(f'q_cur: {transformation.toPhysical(q_cur)}', f'q_ref: {transformation.toPhysical(q_ref)}',
          sep='\n')
    print('functional', functional)

def makeVizualData(inflows_ref, well_params, grid_ref, grid, G_sol, G_ref, P_sol, P_ref, v, J, path='images'):
    fig, ax = plt.subplots(1, 3)
    ax1 = ax[0]
    ax2 = ax[1]
    ax3 = ax[2]
    k = int((grid_ref.size - 1) / (grid.size - 1))
    ax1.scatter(np.array(G_sol) / 24, grid, label='$G_{sol}$', color='b', lw=4)
    ax1.plot(np.array(G_ref) / 24, grid_ref, label='$G_{ref}$', color='r', lw=2)
    foo = np.linspace(min(np.array(G_sol) / 24), max(np.array(G_sol) / 24), num=5)
    ax1.set_xticks(foo, labels=list(map(lambda t: "%.1f" % t, list(foo[:-1]))) + [r"$\frac{m^3}{hour}$"])
    ax1.set_title('Mass flow')
    ax1.grid()
    ax1.invert_yaxis()
    ax1.text(0, 0.5, r"глубина, $m$", rotation=90, transform=ax1.transAxes)
    ax1.legend()

    ax2.scatter([(p - 1) * 1e7 for p in P_sol], grid, label='$P_{sol} - 1 bar$', color='b', lw=4)
    ax2.plot([(p - 1) * 1e7 for p in P_ref], grid_ref, label='$P_{ref} - 1 bar$', color='r', lw=2)
    foo = np.linspace(min([(p - 1) * 1e7 for p in P_sol]), max([(p - 1) * 1e7 for p in P_sol]), num=4)
    ax2.set_xticks(foo, labels=list(map(lambda t: "%.1f" % t, list(foo[:-1]))) + [r"$10^{-7} bar$"])
    ax2.text(0, 0.5, r"глубина, $m$", rotation=90, transform=ax2.transAxes)
    ax2.set_title('Wellbohr pressure')
    ax2.grid()
    ax2.invert_yaxis()
    ax2.legend()

    def analytic_velocity(z):
        return calc_velocity(inflows_ref, z, well_params)

    a = list(map(analytic_velocity, grid_ref))
    ax3.plot(list(map(v, grid_ref)), grid_ref, color='b',
             label='spline ' + f'$ \Delta v{"%.2e" % abs(max(a) - min(a))}$', lw=4)

    ax3.plot(list(map(analytic_velocity, grid_ref)), grid_ref, color='r', label='analytic', lw=2)
    foo = np.linspace(min(a), max(a), num=4)
    ax3.set_xticks(foo, labels=list(map(lambda t: "%.1f" % t, list(foo[:-1]))) + [r"$\frac{m}{sec}$"])
    ax3.text(0, 0.5, r"глубина, $m$", rotation=90, transform=ax3.transAxes)
    ax3.set_title('Velocity')
    ax3.grid()
    ax3.invert_yaxis()
    ax3.legend()

    plt.savefig(f'{path}\sol{J}.png', dpi=1000)


def show_gradients(P_ref, G_ref, grid_ref, v, J, layers, well_params, weights, space, mean, transformation, type: GradMethods, F_0 = 1, delta = 3e-3):
    q = [[] for i in range(len(space))]
    tmp = mean
    for idx, space_ in enumerate(space):
        bar = tmp.copy()
        for space__ in space_:
            bar[idx] = space__
            q[idx].append(bar.copy())

    fig, ax = plt.subplots(1, len(q))
    # производная по q_i
    for idx, q_ in enumerate(q):
        func = []
        q_cur = []
        for q__ in q_:
            P_sol, G_sol, grid, *_ = calculate_forward_problem(well_params, J, layers, transformation.toNormalized(q__), v, None, transformation)
            F = residual(P_ref, G_ref, P_sol, G_sol, weights)
            func.append(F / F_0)
            q_cur.append(q__)
        ax[idx].plot(np.array(q_cur)[:, idx], func, '-o', c='b')
        ax[idx].grid(True)
        ax[idx].set_title(f'Номер параметра: {idx}')
        ax[idx].set_xlabel('Значение параметра')
        ax[idx].set_ylabel('Значение функционала')
        for q___, r in zip(q_cur, func):

            if type is GradMethods.usual:
                grad = grad_calc(transformation.toNormalized(q___), well_params, F_0, J, v, P_ref, G_ref, weights, grid_ref, layers, transformation)
            elif type is GradMethods.adjoint:
                P_sol, G_sol, grid, *_ = calculate_forward_problem(well_params, J, layers,
                                                                   transformation.toNormalized(q___), v, None,
                                                                   transformation)
                grad = grad_calc_adjoint(J, well_params, v, weights, G_ref, G_sol, P_ref, P_sol,
                                         layers, K)
            grad = grad / np.linalg.norm(grad)  # нормированный градиент
            point = np.array([q___[idx], r])
            point_norm = np.linalg.norm(point)
            x_l = (1 - delta) * q___[idx]
            x_r = (1 + delta) * q___[idx]
            y_l = r - grad[idx] * 1e-2 * (q___[idx] - x_l)
            y_r = r - grad[idx] * 1e-2 * (q___[idx] - x_r)



            ax[idx].plot([x_l, x_r], [y_l, y_r], c='r')
    plt.show()


if __name__=="__main__":
    well_params = WellParams(r=10, rho=1000, pressure_bar=1, A=0, B=10)
    J = 91
    K = 10
    weight = np.array([1e5] + [1e2] * K)
    q_ref_main = np.array([60, 40, 50]) * 10  # m^3 / day
    layers = np.array([4.5, 6.5, 8.5])
    transformation = Transformation(q_ref_main)
    P_ref, G_ref, grid_ref, v, inflows_ref, grid = calculate_forward_problem(well_params, 91, layers, transformation.toNormalized(q_ref_main),
                                                                             None,
                                                                             transformation.toNormalized(q_ref_main), transformation)
    show_gradients(np.array(P_ref), np.array(G_ref), grid_ref, v, J, layers, well_params, weight,
                   [np.linspace(565, 567, 11), np.linspace(470, 480, 11), np.linspace(400, 600, 11)],
                   np.array([600, 477.77, 450]),
                   transformation, GradMethods.usual)