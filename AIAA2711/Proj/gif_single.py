import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gif
import lib
import sys
import numpy as np

rgl, rgr = lib.f_range()


T_0, T_f = rgr - rgl, 0.0015
alpha = 0.85

x = np.arange(rgl, rgr, 5e-6)
y = lib.f(x)

y_min, y_max = min(y), max(y)

@gif.frame
def gen_frame(x, y, x_cur, x_new, T_cur) :
    # draw the line
    plt.plot(x, y, "-", linewidth=1)
    plt.scatter(x_cur, lib.f(x_cur), c="red", marker="o")
    # plt.scatter(x_new, lib.f(x_new), c="blue", marker="o")

    plt.title(f"Temperature: {T_cur:.3f}")

    plt.xlim(rgl, rgr)
    plt.ylim(y_min - 1, y_max + 1)
    plt.xlabel("x", loc="right")
    plt.ylabel("y", loc='top')

frames = []
x0 = rgl + (rgr - rgl) * np.random.random_sample(1).item()
T = T_0
while T > T_f :
    x_new = lib.genNew(x0, T, rgl, rgr)
    x0 = lib.metropolis(x0, x_new, T, rgl, rgr, lib.f)
    frames.append(gen_frame(x, y, x0, x_new, T))
    T *= alpha

file_path = "test.gif"
if len(sys.argv) > 1 :
    file_path = sys.argv[1]

gif.save(frames, file_path, duration=80)