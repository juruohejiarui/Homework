import numpy as np
import matplotlib.pyplot as plt

def f(x : np.ndarray) -> np.ndarray :
	return (
		(x ** 2 - 4) ** 2 + 5 * np.sin(2 * np.pi * x)
		  + 3 * np.cos(4 * np.pi * x) + 2 * np.sin(6 * np.pi * x)
		  - 1 * np.cos(8 * np.pi * x) + 2 * np.sin(10 * np.pi * x)
		  + np.exp(np.sin(5 * np.pi * x))
		  + 20 * np.exp((-5 * (x - 1) ** 2)))


def metropolis(y_old, y_new, T) :
	if y_new < y_old :
		return y_new
	elif np.random.random() < np.exp((y_old - y_old) / T) :
		return y_new
	else :
		return y_old


def SA(T_0, T_f, n_thread) :
	x0 = (np.pi + 0.2) * np.random.randn()

	
	
	pass

if __name__ == "__main__" :
	x = np.arange(-np.pi, np.pi, 0.001)
	y = f(x)

	minX, maxX = min(x), max(x)
	minY, maxY = min(y), max(y)

	plt.title("test")
	plt.xlabel("x", loc="right")
	plt.ylabel("y", loc='top')
	plt.plot(x, y, "-")
	plt.hlines(y = minY, xmin = minX, xmax=maxX, color='r', linestyles=':')
	plt.hlines(y = maxY, xmin = minX, xmax=maxX, color='r', linestyles=':')

	plt.show()