import numpy as np

def f_range() -> tuple[float, float] :
	return -1, np.pi

def f(x : np.ndarray) -> np.ndarray :
	return (
	    ((x ** 2 - 4) ** 2) + 5 * np.sin(2 * np.pi * x)
		  + 3 * np.cos(4 * np.pi * x) + 2 * np.sin(6 * np.pi * x)
		  - 1 * np.cos(8 * np.pi * x) + 2 * np.sin(10 * np.pi * x)
		  + np.exp(np.sin(5 * np.pi * x))
		  + 20 * np.exp((-5 * (x - 1) ** 2)))

def metropolis(x_old, x_new, T, rgl, rgr, f) :
	y_new, y_old = f(x_new), f(x_old)
	if y_new < y_old :
		return x_new
	elif np.random.random() < np.exp((y_old - y_new) / (T * (rgr - rgl))) :
		return x_new
	else :
		return x_old
	
def genNew(x, T, rgl, rgr) :
	l, r = max(rgl, x - T), min(rgr, x + T)
	return np.random.random_sample() * (r - l) + l