import numpy as np
import matplotlib.pyplot as plt

rgl, rgr = -2 * np.pi, 2 * np.pi


def f(x : np.ndarray) -> np.ndarray :
	return (
		np.log((x ** 2 - 4) ** 2) + 5 * np.sin(2 * np.pi * x)
		  + 3 * np.cos(4 * np.pi * x) + 2 * np.sin(6 * np.pi * x)
		  - 1 * np.cos(8 * np.pi * x) + 2 * np.sin(10 * np.pi * x)
		  + np.exp(np.sin(5 * np.pi * x))
		  + 20 * np.exp((-5 * (x - 1) ** 2)))

def genNew(x, T) :
	while True :
		x_new = x + T * (2 * np.random.random() - 1)
		if x_new < rgl or x_new > rgr : continue
		return x_new

def metropolis(x_old, x_new, T) :
	y_new, y_old = f(x_new), f(x_old)
	if y_new < y_old :
		return x_new
	elif np.random.random() < np.exp((y_old - y_new) / T) :
		return x_new
	else :
		return x_old


def SA(T_0, T_f, n_thread = 1, alpha = 0.99) :
	x0 = rgl + (rgr - rgl) * np.random.random_sample(n_thread)

	T = T_0
	while T > T_f :
		for i in range(n_thread) :
			x_new = genNew(x[i], T)
			x0[i] = metropolis(x0[i], x_new, T)
		T = T * alpha
	
	# print(x0, f(x0))
	return np.array([x0, f(x0)])

if __name__ == "__main__" :
	x = np.arange(rgl, rgr, 0.001)
	y = f(x)

	minX, maxX = min(x), max(x)
	minY, maxY = min(y), max(y)

	plt.title("test")
	plt.xlabel("x", loc="right")
	plt.ylabel("y", loc='top')
	plt.plot(x, y, "-", linewidth=1)
	# plt.hlines(y = minY, xmin = minX, xmax=maxX, color='r', linestyles=':')
	plt.vlines(x = [x[np.argmin(y)]], ymin=minY, ymax=maxY, color='r', linestyles=':', linewidth=3)
	# plt.hlines(y = maxY, xmin = minX, xmax=maxX, color='r', linestyles=':')

	
	x, y = SA(5000, 0.001, 500, alpha=0.99)

	plt.vlines(x = x, ymin = y, ymax=maxY, colors='g', linestyles='-', linewidth=1)

	plt.show()
