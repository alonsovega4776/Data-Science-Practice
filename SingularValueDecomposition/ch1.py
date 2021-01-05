from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os


X = np.random.rand(5, 3)                            # [n x m] = [5 x 3]
(U, S, V) = np.linalg.svd(X, full_matrices=True)

econ = X.shape[0] >= X.shape[1]
if econ:
    (U_hat, Σ_hat, V_hat) = np.linalg.svd(X, full_matrices=False)
else:
    (U, Σ, V) = np.linalg.svd(X, full_matrices=True)

# ___________________________________________________________________________________________________
plt.rcParams['figure.figsize'] = [16, 8]

A = imread(os.path.join('dog.jpg'))
A_gray = np.mean(A, -1)

img = plt.imshow(A_gray)
img.set_cmap('gray')
plt.axis('off')
plt.show()

econ = A_gray.shape[0] >= A_gray.shape[1]
if econ:
    (U, Σ, V_t) = np.linalg.svd(A_gray, full_matrices=False)
else:
    (U, Σ, V_t) = np.linalg.svd(A_gray, full_matrices=True)
Σ = np.diag(Σ)

j = 0
rank = (5, 20, 100)
for r in rank:
    A_approx = U[:, :r] @ Σ[:r, :r] @ V_t[:r, :]
    plt.figure(j+1)
    j += 1
    img = plt.imshow(A_approx)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('r = ' + str(r))

plt.figure(1)
plt.semilogy(np.diag(Σ))
plt.title('Singular Values')
plt.show()

plt.figure(2)
plt.plot(np.cumsum(np.diag(Σ))/np.sum(np.diag(Σ)))
plt.title('Cumulative energy')
plt.show()

