import matplotlib.pyplot as plt
import numpy as np
import math
import os
import scipy.io
from scipy.spatial.transform import Rotation as rot
plt.rcParams['figure.figsize'] = [16, 8]
plt.rcParams.update({'font.size': 18})

data = scipy.io.loadmat(os.path.join('.', 'allFaces.mat'))
faces = data['faces']
m = int(data['m'])
n = int(data['n'])
n_faces = np.ndarray.flatten(data['nfaces'])

allPersons = np.zeros((n*6, m*6))
count = 0

for i_1 in range(6):
    for i_2 in range(6):                                            # only use the 1st 36 people
        allPersons[i_1*n: (i_1+1)*n, i_2*m: (i_2+1)*m] = \
            np.reshape(faces[:, np.sum(n_faces[:count])], (m, n)).T
        count += 1

img = plt.imshow(allPersons)
img.set_cmap('gray')
plt.axis('off')
plt.show()

for person in range(len(n_faces)):
    subset = faces[:, sum(n_faces[: person]): sum(n_faces[: person+1])]
    allFaces = np.zeros((n*8, m*8))

    count = 0

    for i_1 in range(8):
        for i_2 in range(8):
            if count < n_faces[person]:
                allFaces[i_1*n: (i_1+1)*n, i_2*m: (i_2+1)*m] = np.reshape(subset[:, count], (m, n)).T
                count += 1

    img = plt.imshow(allFaces)
    img.set_cmap('gray')
    plt.axis('off')
#   plt.show()

# ______________________________________________________________________________________________________________________

training_faces = faces[:, :np.sum(n_faces[:36])]
avg_face = np.mean(training_faces, axis=1)

X = training_faces - np.tile(avg_face, (training_faces.shape[1], 1)).T
(U, Σ, V_trans) = np.linalg.svd(X, full_matrices=False)

fig_1 = plt.figure()
ax_1 = fig_1.add_subplot(121)
img_avg = ax_1.imshow(np.reshape(avg_face, (m, n)).T)
img_avg.set_cmap('gray')
plt.axis('off')

ax_2 = fig_1.add_subplot(122)
img_u1 = ax_2.imshow(np.reshape(U[:, 0], (m, n)).T)
img_u1.set_cmap('gray')
plt.axis('off')

plt.show()

test_face = faces[:, np.sum(n_faces[:36])]
plt.imshow(np.reshape(test_face, (m, n)).T)
plt.set_cmap('gray')
plt.title('Original Image')
plt.axis('off')
plt.show()

test_faceMS = test_face - avg_face
r_list = [25, 50, 100, 200, 400, 800, 1600]

for r in r_list:
    face_hat = avg_face + U[:, :r] @ (U[:, :r].T @ test_faceMS)
    img = plt.imshow(np.reshape(face_hat, (m, n)).T)
    img.set_cmap('gray')
    plt.title('r = ' + str(r))
    plt.axis('off')

    plt.show()