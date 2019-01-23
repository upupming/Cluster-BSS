import numpy as np
import matplotlib.pyplot as plt

t = np.arange(256)
sp = np.fft.fft(np.sin(t))
freq = np.fft.fftfreq(t.shape[-1], 0.1)

# sp = np.fft.fftshift(sp)
# freq = np.fft.fftshift(freq)

print(sp)
print(freq)

plt.plot(freq, sp.real, freq, sp.imag)

plt.show()