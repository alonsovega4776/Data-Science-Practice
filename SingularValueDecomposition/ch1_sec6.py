import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import os

plt.rcParams['figure.figsize'] = [16, 6]
plt.rcParams.update({'font.size': 18})


# Randomized SVD
def rand_svd(X, newRank, overSample, iterPower):
    # Step 1: sample col. space of X [n x m]
    col_X = X.shape[1]
    P = np.random.randn(col_X, newRank + overSample)                 # over-sample by p
    Z = X @ P

    for i_1 in range(iterPower):                            # q power iterations
        Z = X @ (X.T @ Z)                           # Z approx. col.s of X with high prob.

    (Q, R) = np.linalg.qr(Z, mode='reduced')        # QR-fact. of Z, get low rank Q

    # Step 2: Project X onto smaller dim.
    Y = Q.T @ X                                     # Project X into Q
    (U_Y, Σ, V_trans) = np.linalg.svd(Y,
                             full_matrices=False)   # SVD of Y

    # Step 3: Reconstruct LSV(Left Singular Vectors) of X
    U_X = Q @ U_Y

    return U_X, Σ, V_trans
# ______________________________________________________________________________________________________________________


A = imread(os.path.join('.', 'jupiter.jpg'))
X = np.mean(A, axis=2)                                      # Convert to gray scale

(U, Σ, V_trans) = np.linalg.svd(X, full_matrices=False)     # Ordinary SVD

r = 400
q = 1
p = 5
(U_rand, Σ_rand, V_rand_trans) = rand_svd(X, newRank=r, overSample=p, iterPower=q)      # Random SVD

# Reconstruction

X_hat = U[:, :r + 1] @ np.diag(Σ[:r + 1]) @ V_trans[:r + 1, :]
ε = np.linalg.norm(X - X_hat)

X_hat_rand = U_rand[:, :r + 1] @ np.diag(Σ_rand[:r + 1]) @ V_rand_trans[:r + 1, :]
ε_rand = np.linalg.norm(X - X_hat_rand, ord=2)

(fig, axis) = plt.subplots(1, 3)

plt.set_cmap('gray')
axis[0].imshow(X)
axis[0].axis('off')
axis[1].imshow(X_hat)
axis[1].axis('off')
axis[2].imshow(X_hat_rand)
axis[2].axis('off')

plt.show()

# ______________________________________________________________________________________________________________________

# Power Iteration
X = np.random.randn(1000, 100)

(U, Σ, V_trans) = np.linalg.svd(X, full_matrices=False)     # Ordinary SVD
Σ = np.arange(1, 0, -0.01)
X_hat = U @ np.diag(Σ) @ V_trans

color_list = np.array([[0, 0, 2/3],  # Define color map
                 [0, 0, 1],
                 [0, 1/3, 1],
                 [0, 2/3, 1],
                 [0, 1, 1],
                 [1/3, 1, 2/3],
                 [2/3, 1, 1/3],
                 [1, 1, 0],
                 [1, 2/3, 0],
                 [1, 1/3, 0],
                 [1, 0, 0],
                 [2/3, 0, 0]])

plt.plot(Σ, 'o-', color='k', LineWidth=2, label='SVD')

Υ = X
for q in range(1, 6):
    Y = X.T @ Y
    Y = X @ Y
    (U_q, Σ_q, V_q_trans) = np.linalg.svd(Y, full_matrices=0)

plt.legend()
plt.show()

