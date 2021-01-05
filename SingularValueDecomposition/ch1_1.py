from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.spatial.transform import Rotation as rot
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = [16, 8]
plt.rcParams.update({'font.size': 18})

# Parameters
θ_rad = np.array([np.pi/15, -np.pi/9, np.pi/20])
θ = [12.0, -20.0, -9.0]
σ = [3.0, 1.0, 1.0/2.0]

# Rotations
R_x = rot.from_euler(seq='X', angles=θ[0], degrees=True)
R_y = rot.from_euler(seq='Y', angles=θ[1], degrees=True)
R_z = rot.from_euler(seq='Z', angles=θ[2], degrees=True)

R_zyx = R_z.as_matrix() @ R_y.as_matrix() @ R_x.as_matrix()

# Scaling
Σ = np.diag(σ)


X = R_zyx @ Σ       # V is the identity

# Hypersphere Sphere
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
u = np.linspace(-np.pi, np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

# Plot Manifold
surf_1 = ax1.plot_surface(x, y, z, cmap='jet', alpha=0.6,
                          facecolors=plt.cm.jet(z), linewidth=0.5, rcount=30, ccount=30)
surf_1.set_edgecolor('k')
ax1.set_xlim3d(-2, 2)
ax1.set_ylim3d(-2, 2)
ax1.set_zlim3d(-2, 2)

x_rot = np.zeros_like(x)
y_rot = np.zeros_like(y)
z_rot = np.zeros_like(z)

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        vec = [x[i, j], y[i, j], z[i, j]]
        vec_rot = X @ vec
        x_rot[i, j] = vec_rot[0]
        y_rot[i, j] = vec_rot[1]
        z_rot[i, j] = vec_rot[2]

ax2 = fig.add_subplot(122, projection='3d')
surf_2 = ax2.plot_surface(x_rot, y_rot, z_rot, cmap='jet', alpha=0.6,
                          facecolors=plt.cm.jet(z), linewidth=0.5, rcount=30, ccount=30)
surf_2.set_edgecolor('k')
ax2.set_xlim3d(-2, 2)
ax2.set_ylim3d(-2, 2)
ax2.set_zlim3d(-2, 2)

plt.show()


