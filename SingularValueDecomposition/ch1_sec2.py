import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})

x = 3                                        # true slope
a = np.arange(-2, 2, 0.25)                   # domain
a = a.reshape((-1, 1))                       # col. vector

(μ, σ) = 0, 0.5                              # normal dist.
ε = np.random.normal(μ, σ, a.shape)          # distrurbance

b = x*a + ε

plt.plot(a, x*a, Color='k', LineWidth=2, label='True Line')
plt.plot(a, b, 'x', Color='r', MarkerSize=10, label='Noisy Data')

(U, Σ, V_trans) = np.linalg.svd(a, full_matrices=False)
x_tilde = V_trans.T @ np.linalg.inv(np.diag(Σ)) @ U.T @ b           # least square fit

plt.plot(a, x_tilde * a, '--', Color='b', LineWidth=4, label='Regression Line')

plt.xlabel('a')
plt.ylabel('b')

plt.grid(linestyle='--')
plt.legend()
plt.show()

# ______________________________________________________________________________________________________________________

# load data
A = np.loadtxt(os.path.join('hald_ingredients.csv'), delimiter=',')
b = np.loadtxt(os.path.join('hald_heat.csv'), delimiter=',')

(U_c, Σ_c, Vtrans_c) = np.linalg.svd(A, full_matrices=False)
x = Vtrans_c.T @ np.linalg.inv(np.diag(Σ_c)) @ U_c.T @ b

plt.plot(b, Color='k', LineWidth=2, label='Heat Data')
plt.plot(A @ x, '-o', Color='r', LineWidth=1.5, MarkerSize=6, label='Regression')
plt.legend()
plt.xlabel('Element-wise')
plt.show()

# ______________________________________________________________________________________________________________________

