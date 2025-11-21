# 生成并显示蠕变试验中三种材料的应变-时间曲线图（单图）
import numpy as np
import matplotlib.pyplot as plt

# 参数
sigma0 = 1.0
E_elastic = 1.0  # 理想弹性体的弹性模量，最终应变 = sigma0 / E_elastic = 1.0
eta_viscous = 1.0  # 理想黏性体的黏度，斜率 = sigma0/eta = 1.0
# 标准线性固体 (Zener) 参数（粘弹性聚合物示例）
E1 = 2.0   # 串联弹簧（瞬时弹性）
E2 = 5.0   # 并联弹簧（迟滞弹性）
eta = 2.0  # 黏性元件粘度
tau = eta / E2

t = np.linspace(0, 5, 500)
# 理想弹性固体：瞬时跳变到常数（在数值图中用极小时间模拟瞬时跳变）
t_elastic = np.concatenate(([0.0, 1e-6], t[1:]))
gamma_elastic = np.concatenate(([0.0, sigma0 / E_elastic], np.full(t.size-1, sigma0 / E_elastic)))

# 理想牛顿流体：线性增长
gamma_viscous = (sigma0 / eta_viscous) * t

# 标准线性固体 (Zener)：瞬时弹性 + 指数趋近的迟滞部分
gamma_visco = (sigma0 / E1) + (sigma0 / E2) * (1 - np.exp(-t / tau))

# 绘图（单个子图）
plt.figure(figsize=(8,5))
plt.plot(t_elastic, gamma_elastic, label='理想胡克弹性固体（弹性）', linewidth=2)
plt.plot(t, gamma_viscous, label='理想牛顿黏性流体（黏性）', linewidth=2)
plt.plot(t, gamma_visco, label='粘弹性聚合物（标准线性固体）', linewidth=2)

plt.xlabel('时间 t')
plt.ylabel('应变 γ')
plt.title('蠕变试验：应变 γ vs 时间 t（在 t=0 施加常应力 σ0）')
plt.legend()
plt.grid(True)
plt.xlim(0, 5)
plt.ylim(0, 1.2)
plt.annotate('σ0/E (弹性体)', xy=(0.5, sigma0/E_elastic), xytext=(1.0, 1.02),
             arrowprops=dict(arrowstyle='->', linewidth=0.8))
plt.tight_layout()
plt.show()
