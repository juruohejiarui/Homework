import numpy as np
import matplotlib.pyplot as plt
import lib

rgl, rgr = lib.f_range()

def genNew(x, T) :
	l, r = max(rgl, x - T), min(rgr, x + T)
	return np.random.random_sample() * (r - l) + l


def SA(T_0, T_f, n_thread = 1, alpha = 0.99) :
	x0 = rgr + (rgr - rgl) * np.random.random_sample(n_thread)
	T = T_0
	while T > T_f :
		for i in range(n_thread) :
			x_new = genNew(x0[i], T)
			x0[i] = lib.metropolis(x0[i], x_new, T, lib.f)
		T = T * alpha
	
	# print(x0, f(x0))
	return np.array([x0, lib.f(x0)])

if __name__ == "__main__" :
	x = np.arange(rgl, rgr, 5e-6)
	y = lib.f(x)

	minX, maxX = min(x), max(x)
	minY, maxY = min(y), max(y)

	plt.title("test")
	plt.xlabel("x", loc="right")
	plt.ylabel("y", loc='top')
	plt.plot(x, y, "-", linewidth=1)
	# plt.hlines(y = minY, xmin = minX, xmax=maxX, color='r', linestyles=':')
	plt.vlines(x = [x[np.argmin(y)]], ymin=minY, ymax=maxY, color='r', linestyles=':', linewidth=3)
	# plt.hlines(y = maxY, xmin = minX, xmax=maxX, color='r', linestyles=':')

	
	x, y = SA(rgr - rgl, 0.001, 500, alpha=0.99)

	plt.vlines(x = x, ymin = y, ymax=maxY, colors='g', linestyles='-', linewidth=1)

	plt.show()
