import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.io
import matplotlib.patches as patches
plt.rcParams['figure.figsize'] = [16, 8]
plt.rcParams.update({'font.size': 18})

t = np.arange(-3, 3, 0.01)

# True
U_1 = np.cos(17*t) * np.exp(-t**2)
U_2 = np.sin(11*t)
U_true = np.array([U_1, U_2]).T

Σ_1 = [2, 0]
Σ_2 = [0, 0.5]
Σ_true = np.array([Σ_1, Σ_2])

V_1 = np.sin(5*t) * np.exp(-t**2)
V_2 = np.cos(13*t)
V_true = np.array([V_1, V_2]).T

X_true = U_true @ Σ_true @ V_true.T

plt.imshow(X_true)
plt.set_cmap('gray')
plt.axis('off')

plt.show()

# Noise
γ = 1.0
X_noise = np.random.randn(*X_true.shape)
X = X_true + γ*X_noise

plt.imshow(X)
plt.set_cmap('gray')
plt.axis('off')

plt.show()

# Optimal Truncated SVD
(U, Σ, V_trans) = np.linalg.svd(X, full_matrices=False)
N = X.shape[0]

τ = (4/np.sqrt(3)) * np.sqrt(N) * γ
r = np.max(np.where(Σ > τ))

X_hat = U[:, :r+1] @ np.diag(Σ[:r+1]) @ V_trans[:r+1, :]
plt.imshow(X_hat)
plt.set_cmap('gray')
plt.axis('off')

plt.show()

# 90 % of Cumulative Energy Truncation
energy = np.cumsum(Σ) / np.sum(Σ)
r_90pph = np.min(np.where(energy > 0.90))

X_hat_90energy = U[:, :r_90pph+1] @ np.diag(Σ[:r_90pph+1]) @ V_trans[:r_90pph+1, :]
plt.imshow(X_hat_90energy)
plt.set_cmap('gray')
plt.axis('off')

# Singular Values
fig1, ax1 = plt.subplots(1)

ax1.semilogy(Σ, '-o', color='k', LineWidth=2)
ax1.semilogy(np.diag(Σ[:r+1]), 'o', color='r', LineWidth=2)
ax1.plot(np.array([-20, N+20]), np.array([τ, τ]), '--', color='r', LineWidth=2)
rect = patches.Rectangle((-5, 20), 100, 200, LineWidth=2, LineStyle='--', FaceColor='none', EdgeColor='k')
ax1.add_patch(rect)
plt.xlim((-10, 610))
plt.ylim((0.003, 300))
ax1.grid()
plt.show()

fig2, ax2 = plt.subplots(1)

ax2.semilogy(Σ, '-o', color='k', LineWidth=2)
ax2.semilogy(np.diag(Σ[:(r+1)]), 'o', color='r', LineWidth=2)
ax2.plot(np.array([-20, N+20]), np.array([τ, τ]), '--', color='r', LineWidth=2)
plt.xlim((-5, 100))
plt.ylim((20, 200))
ax2.grid()
plt.show()

fig3, ax3 = plt.subplots(1)
ax3.plot(energy, '-o', color='k', LineWidth=2)
ax3.plot(energy[:(r_90pph+1)], 'o', color='b', LineWidth=2)
ax3.plot(energy[:(r+1)], 'o',color='r', LineWidth=2)
plt.xticks(np.array([0, 300, r_90pph, 600]))
plt.yticks(np.array([0, 0.5, 0.9, 1]))
plt.xlim((-10, 610))
ax3.plot(np.array([r_90pph, r_90pph, -10]),np.array([0, 0.9, 0.9]),'--',color='b',LineWidth=2)

ax3.grid()
plt.show()


