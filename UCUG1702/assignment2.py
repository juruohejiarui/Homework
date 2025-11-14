import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(1e-5, 1, 300, dtype=float)

def E_a(x : float | np.ndarray) -> float | np.ndarray:
	return -1.436 / x

def E_r(x : float | np.ndarray) -> float | np.ndarray:
	return 7.32 * (10 ** -6) / np.pow(x, 8)

def E_n(x : float | np.ndarray) -> float | np.ndarray:
	return E_a(x) + E_r(x)

y_a = E_a(x)
y_r = E_r(x)
y_n = E_n(x)

plt.plot(x, y_a, label='E_a', color='blue')
plt.plot(x, y_r, label='E_r', color='black')
plt.plot(x, y_n, label='E_n', color='green')


plt.ylim(-30, 30)
plt.xlim(0, 1)


plt.title('Potential Energy vs Distance')

plt.xlabel('Distance (nm)')
plt.ylabel('Potential Energy (eV)')

plt.legend()
plt.grid()
plt.show()