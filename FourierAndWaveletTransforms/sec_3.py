import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import axes3d
plt.rcParams['figure.figsize'] = [12, 12]
plt.rcParams.update({'font.size': 18})

c = 2
L = 20
N = 1000

dx = L/N
x = np.arange(-L/2, L/2, dx)

κ = 2*np.pi*np.fft.fftfreq(N, d=dx)

u_0 = 1/np.cosh(x)
u_0_hat = np.fft.fft(u_0)

u_0_hat_ri = np.concatenate((u_0_hat.real, u_0_hat.imag))

dt = 0.025
t = np.arange(0, 100*dt, dt)


def rhs_wave(u_hat_ri, t, κ, c):
    u_hat = u_hat_ri[:N] + (1j)*u_hat_ri[N:]
    d_u_hat = -c*(1j)*κ*u_hat
    d_u_hat_ri = np.concatenate((d_u_hat.real, d_u_hat.imag)).astype('float64')
    return d_u_hat_ri


u_hat_ri = odeint(rhs_wave, u_0_hat_ri, t, args=(κ, c))
u_hat = u_hat_ri[:, :N] + (1j) * u_hat_ri[:, N:]





