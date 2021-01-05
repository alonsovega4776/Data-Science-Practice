import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'font.size': 18})
plt.rcParams['figure.figsize'] = [12, 12]


# Circle Map
def circle(x_k, Ω, K):
    return x_k + Ω - (K/2*np.pi)*np.sin(2*np.pi*x_k)


# Logistic Map
def logistic(x_k, β):
    return β*x_k*(1 - x_k)


ζ_i = 0.0
ζ_f = 1.0
mesh = np.array([[], []])
n_iter = 500
n_plot = n_iter - 100

for β in np.arange(ζ_i, ζ_f, 0.00025):
    x = 0.2
    for i_1 in range(n_iter):
        x = circle(x, β, 0.5)
        if i_1 == n_plot:
            x_ss = x
        if i_1 > n_plot:
            mesh = np.append(mesh, np.array([[β], [x]]), axis=1)
            if np.abs(x - x_ss) < 0.001:
                break

plt.plot(mesh[1, :], mesh[0, :], '.', ms=0.1, color='k')
plt.xlim(0, 1)
plt.ylim(ζ_i, ζ_f)
plt.gca().invert_yaxis()

plt.show()

plt.plot(mesh[1, :], mesh[0, :], '.', ms=0.1, color='k')
plt.xlim(0, 1)
plt.ylim(3.45, 4)
plt.gca().invert_yaxis()
