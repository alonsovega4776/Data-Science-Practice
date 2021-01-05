import matplotlib.pyplot as plt
import numpy as np
import math
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as rot
plt.rcParams['figure.figsize'] = [16, 8]


def noisy_data(μ, Σ, θ, n):
    R = rot.from_euler('Z', θ, True)
    R = R.as_matrix()[:2, :2]
    return R @ Σ @ np.random.randn(2, n) + np.diag(μ) @ np.ones((2, n))


def pca(X, n):
    X_avg = np.mean(X, axis=1)
    B = X - np.tile(X_avg, (n, 1)).T
    (U, Σ_svd, V_trans) = np.linalg.svd(B / np.sqrt(n))
    return U, Σ_svd, V_trans, X_avg


Μ = np.array([[0, 0],
              [1, 2],
              [2, 3],
              [5, 6]])                              # translation from origin
Σ_cov = np.concatenate((np.diag([1.5, 1.5]),
                        np.diag([1.7, 0.8]),
                        np.diag([0.2, 2.0]),
                        np.diag([2.0, 1.4])), axis=0)    # compress and elongate data (variance)
Ω = [0.0, 10.0, 20.0, 65.0]
n_points = 1000

Ξ = np.zeros((2 * 4, n_points))
for i_1 in range(4):
    X = noisy_data(Μ[i_1, :], Σ_cov[2*i_1:2*i_1+2, :], Ω[i_1], n_points)
    Ξ[2 * i_1:2 * i_1 + 2, :] = X

fig = plt.figure()
θ = 2 * np.pi * np.arange(0, 1, 0.01)
for i_2 in range(4):
    #fig = plt.figure()
    ax_1 = fig.add_subplot(4, 2, 2*i_2 + 1)
    ax_1.plot(Ξ[2*i_2, :], Ξ[2*i_2+1, :], '.', Color='k')
    ax_1.grid()
    plt.xlim(-5, 10)
    plt.ylim(-5, 12)
    plt.title('Noisy Data  μ = ' + str(Μ[i_2, :]) + '   Σ = ' + str(Σ_cov[2*i_2:2*i_2+2, :])
              + '  Rotated ' + str(Ω[i_2]) + ' deg.', color='blue', fontsize=10.5)

    ax_2 = fig.add_subplot(4, 2, 2*i_2+2)
    ax_2.plot(Ξ[2*i_2, :], Ξ[2*i_2+1, :], '.', Color='k')
    ax_2.grid()
    plt.xlim((-5, 10))
    plt.ylim((-5, 12))
    plt.title('Principal Components')

    (U, Σ_svd, V_trans, X_avg) = pca(Ξ[2*i_2:2*i_2 + 2, :], n_points)

    X_std = U @ np.diag(Σ_svd) @ np.array([np.cos(θ), np.sin(θ)])

    ell_1, = ax_2.plot(X_avg[0] + X_std[0, :], X_avg[1] + X_std[1, :], '-', color='r', LineWidth=1.5)
    ell_2, = ax_2.plot(X_avg[0] + 2 * X_std[0, :], X_avg[1] + 2 * X_std[1, :], '-', color='r', LineWidth=1.5)
    ell_3, = ax_2.plot(X_avg[0] + 3 * X_std[0, :], X_avg[1] + 3 * X_std[1, :], '-', color='r', LineWidth=1.5)

    vec_1, = ax_2.plot(np.array([X_avg[0], X_avg[0] + U[0, 0] * Σ_svd[0]]),
              np.array([X_avg[1], X_avg[1] + U[1, 0] * Σ_svd[0]]), '-', color='cyan', LineWidth=1.5)
    vec_2, = ax_2.plot(np.array([X_avg[0], X_avg[0] + U[0, 1] * Σ_svd[1]]),
              np.array([X_avg[1], X_avg[1] + U[1, 1] * Σ_svd[1]]), '-', color='cyan', LineWidth=1.5)
    ax_2.legend([ell_1, vec_1], ['standard deviation ellipsoids', 'left singular vectors'])

fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
plt.show()



'''
# ______________________________________________________________________________________________________________________
data_obs = np.loadtxt(os.path.join('.', 'ovariancancer_obs.csv'), delimiter=',')

f = open(os.path.join('.', 'ovariancancer_grp.csv'), 'r')
grp = f.read().split('\n')

(U_ov, Σ_ov, V_trans_ov) = np.linalg.svd(data_obs, full_matrices=False)

fig_ov = plt.figure()
ax_ov = fig_ov.add_subplot(111, projection='3d')

for j in range(data_obs.shape[0]):
    x = V_trans_ov[0, :] @ data_obs[j, :].T
    y = V_trans_ov[1, :] @ data_obs[j, :].T
    z = V_trans_ov[2, :] @ data_obs[j, :].T

    if grp[j] == 'Cancer':
        ax_ov.scatter(x, y, z, marker='x', color='r', s=50)
    else:
        ax_ov.scatter(x, y, z, marker='o', color='b', s=50)

ax_ov.view_init(25, 20)
plt.show()
'''

