import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})

# Domain
dt = 0.001
L = np.pi
t = L*np.arange(-1 + dt, 1 + dt, dt)
n = len(t)
n_quad = int(np.floor(n / 4))

# Hat Function
f = np.zeros_like(t)
f[n_quad: 2*n_quad] = (4 / n) * np.arange(1, n_quad + 1)
f[2*n_quad: 3*n_quad] = np.ones(n_quad) - (4/n)*np.arange(0, n_quad)

(fig, axis) = plt.subplots()
axis.plot(t, f, '-', color='k', LineWidth=2.0)

# Fourier Series
name = 'Accent'
c_map = get_cmap('tab10')
colors = c_map.colors
axis.set_prop_cycle(color=colors)

A_0 = np.sum(f) * dt
f_hat = A_0 / 2

a_coeff = np.zeros(20)
b_coeff = np.zeros(20)
for i_1 in range(20):
    # basis
    g_a = np.cos(np.pi * (i_1 + 1) * t / L)
    g_b = np.sin(np.pi * (i_1 + 1) * t / L)

    # projections
    a_coeff[i_1] = np.sum(f * g_a) * dt
    b_coeff[i_1] = np.sum(f * g_b) * dt

    # linear combination
    f_hat = f_hat + a_coeff[i_1] * g_a \
                  + b_coeff[i_1] * g_b
    axis.plot(t, f_hat, '--')

plt.show()
