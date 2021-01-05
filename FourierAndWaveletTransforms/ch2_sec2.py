import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 12]
plt.rcParams.update({'font.size': 18})

dt = 0.001
t = np.arange(0, 1, dt)

f = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t)
f_ε = f + 2.5*np.random.randn(len(t))


n = len(t)
f_hat = np.fft.fft(f_ε, n)

pow_spec = f_hat * np.conj(f_hat)/n
freq = (1/(dt*n)) * np.arange(n)
L = np.arange(1, np.floor(n/2), dtype='int')

ind = pow_spec > 100
pow_spec_filt = pow_spec * ind

f_hat = ind * f_hat
f_filt = np.fft.ifft(f_hat)

'''
fig, axis = plt.subplots(3, 1)

plt.sca(axis[0])
plt.plot(t, f_ε, color='r', LineWidth=1.5, label='Noisy')
plt.plot(t, f, color='k', LineWidth=2, label='Clean')
plt.xlim(t[0], t[-1])
plt.legend()

plt.sca(axis[1])
plt.plot(t, f, color='k', LineWidth=1.5, label='Clean')
plt.plot(t, f_filt, color='b', LineWidth=2, label='Filtered')
plt.xlim(t[0], t[-1])
plt.legend()

plt.sca(axis[2])
plt.plot(freq[L], pow_spec[L], color='r', LineWidth=2, label='Noisy')
plt.plot(freq[L], pow_spec_filt[L], color='b', LineWidth=1.5, label='Filtered')
plt.xlim(freq[L[0]], freq[L[-1]])
plt.legend()
'''

n = 128
L = 30

dx = L/n
x = np.arange(-L/2, L/2, dx, dtype='complex')
f = np.cos(x) * np.exp(-np.power(x, 2)/25)
dfdx = -(np.sin(x) * np.exp(-np.power(x, 2)/25) + (2/25)*x*f)

dfdx_fdiff = np.zeros(len(dfdx), dtype='complex_')
for κ in range(len(dfdx) - 1):
    dfdx_fdiff[κ] = (f[κ+1] - f[κ])/dx

dfdx_fdiff[-1] = dfdx_fdiff[-2]


f_hat = np.fft.fft(f)
κ = (2*np.pi/L)*np.arange(-n/2, n/2)
κ = np.fft.fftshift(κ)
dfdx_hat = κ * f_hat * (1j)
dfdx_FFT = np.real(np.fft.ifft(dfdx_hat))

plt.plot(x, dfdx.real,color='k',LineWidth=2,label='True Derivative')
plt.plot(x, dfdx_fdiff.real,'--',color='b',LineWidth=1.5,label='Finite Diff.')
plt.plot(x, dfdx_FFT.real,'--',color='r',LineWidth=1.5,label='FFT Derivative')
plt.legend()
plt.show()



