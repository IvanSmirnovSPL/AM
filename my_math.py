import numpy as np
from numpy.typing import NDArray
from wellborn import dims
from wellborn import WellParams


def f_forward(well_params: WellParams, J, layers, q):
    h = (well_params.B - well_params.A) / (J - 1)
    half_h = h / 2
    f = np.zeros(J * 2)
    for j in range(J - 1):
        for i in range(np.array(layers).size):
            if np.isclose((j * h + half_h), layers[i]):
                f[j] = - dims.d.SI(q[i]) / h
        f[j + j] = well_params.rho * well_params.g * (j * h + half_h)
    f[J - 1] = 0
    f[-1] = dims.p.SI(well_params.BHP)
    return f

def calc_forward_problem_matrix(J, well_params, v):
    h = (well_params.B - well_params.A) / (J - 1)
    A = np.zeros((J * 2, J * 2))
    for j in range(J - 1):
        A[j][j] = -1.
        A[j][j + 1] = 1.
    A[J - 1][J - 1] = h
    for j in range(J, J * 2 - 1):
        A[j][j - J] = dims.v.SI(v((j - J) * h)) / well_params.S
        A[j][j + 1 - J] = - dims.v.SI(v((j + 1 - J) * h)) / well_params.S
        A[j][j] = -1.
        A[j][j + 1] = 1.
    A[J * 2 - 1][J * 2 - 1] = h
    return A / h


def calc_weights_matrix(weight, K=10):
    W = np.zeros((K + 1, K + 1))
    W[0][0] = 2 * weight[0]
    for m in range(K):
        W[m + 1][m + 1] = 2 * weight[m + 1]
    return W


def calc_change_dim_matrix(M, K=10):
    B = np.zeros((1 + K, 2 * M))
    B[0][0] = 1
    k = int(M / K)
    for m in range(K):
        B[1 + m][M + k * m] = 1
    return B


class Transformation:
    indent: float = 0.5

    def __init__(self, p: NDArray):
        self._min = min(p) * (1 - self.indent)
        self._max = max(p) * (1 + self.indent)
        self._amplitude = self._max - self._min

    def toNormalized(self, p):
        return (p - np.ones_like(p) * self._min) / self._amplitude

    def toPhysical(self, x):
        return x * self._amplitude + np.ones_like(x) * self._min


def F(x1, x2):
    return np.linalg.norm(np.array(x1) - np.array(x2), ord=2)


def heaviside(x: float):
    if x >= 0:
        return 1
    else:
        return 0


def delta(x, err):
    if abs(x) <= err:
        return 1
    else:
        return 0


def float_equal(a: float, b: float, accuracy: float = 1e-10):
    return abs(a - b) <= accuracy


def interpolate_coefs(fd_depth, md_depth):
    """Calculate coefficients for linear interpolation FROM md_depth TO fd_depth depth grid.

    Parameters
    ----------
    fd_depth : numpy.ndarray, shape(m1,)
        Array of depth grid we make interpolation TO.
    md_time : numpy.ndarray, shape(m2,)
        Array of depth grid we make interpolation FROM.

    Returns
    ----------
    coeffs : list of point_coeff, where
        point_coeff : list[ind1, ind2, k1, k2]
            Coefficients for each point d in fd_depth: ind1 and ind2 are the indexes of points in md_depth around d, k1 and k2 are linear interpolation coefficients.

    """
    coeffs = []
    if len(fd_depth) == 1:
        return coeffs

    for d in fd_depth:
        for i in range(len(md_depth) - 1):
            if (md_depth[i] <= d < md_depth[i + 1]) or (md_depth[i] < d <= md_depth[i + 1]):
                ind1 = i
                ind2 = ind1 + 1
                k2 = (d - md_depth[ind1]) / (md_depth[ind2] - md_depth[ind1])
                k1 = 1 - k2
                coeffs.append([ind1, ind2, k1, k2])
                break
    return coeffs


def interpolate(md, coeffs_t):
    """Make linear interpolation for md (2D data array) using coeffs_t (interpolation along time grid) and coeffs_d (interpolation along depth grid).

    Parameters
    ----------
    md : numpy.ndarray, shape(n, m)
        2D array of data values before interpolation.
    coeffs_t : list, length is n1
        Coefficients for interpolation. See interpolateTime.
    coeffs_d : list, length is m1
        Coefficients for interpolation. See interpolateDepth.

    Returns
    ----------
    data : numpy.ndarray, shape(n1, m1)
        2D array of data values after interpolation.

    """
    coeffs_t = np.array(coeffs_t)

    t_i1 = coeffs_t.T[0].astype('int32')
    t_i2 = coeffs_t.T[1].astype('int32')
    t_k1 = coeffs_t.T[2]
    t_k2 = coeffs_t.T[3]

    data = md[t_i1] + np.multiply(t_k2.reshape(t_k2.shape[0], 1), (md[t_i2] - md[t_i1]))

    return data
