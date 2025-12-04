import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

#data_path = 'example_data.lammpstrj'
data_path = 'F:/university/UCUG1702/water 1/dump-surface.lammpstrj'
#data_path = 'F:/university/UCUG1702/water 1/dump-surface-280.lammpstrj'
#data_path = 'F:/university/UCUG1702/water 1/dump-surface-320.lammpstrj'

# 沿 z 轴方向将体系分层（用于分析不同 z 高度的水分子取向）
layer_num = 1000

# 存储盒子边界的最大/最小坐标（后续从轨迹文件中读取）
max_pos = None  # np.ndarray shape=(3,)
min_pos = None  # np.ndarray shape=(3,)


class H2O:
    """水分子类，用于存储单个水分子的氧原子和两个氢原子坐标，并处理周期性边界问题"""

    def __init__(self, 
                 pO: np.ndarray, 
                 pH1: np.ndarray, 
                 pH2: np.ndarray,
                 box_min: np.ndarray,
                 box_max: np.ndarray) -> None:
        """
        初始化水分子坐标

        参数:
            pO: 氧原子坐标 (x,y,z)
            pH1: 第一个氢原子坐标 (x,y,z)
            pH2: 第二个氢原子坐标 (x,y,z)
            box_min: 本帧盒子下界 (xlo, ylo, zlo)
            box_max: 本帧盒子上界 (xhi, yhi, zhi)
        """
        # 保存本帧盒子边界
        self.box_min = box_min.copy()
        self.box_max = box_max.copy()

        # 原子坐标
        self.pO = pO.copy()
        self.pH1 = pH1.copy()
        self.pH2 = pH2.copy()

        # 在 box 信息已知时做 PBC 修正
        self.pH1 = self._fix_pH(self.pH1)
        self.pH2 = self._fix_pH(self.pH2)

    def _fix_pH(self, pH: np.ndarray) -> np.ndarray:
        """
        修正氢原子坐标以符合周期性边界条件

        思路：如果氢原子与氧原子的差向量在某一维超过 box_length / 2，
        就通过 ±box_length 把 H 拉回到与 O 最近的镜像（只平移一次）。

        参数:
            pH: 氢原子当前坐标

        返回:
            修正后的氢原子坐标
        """
        p = pH.copy()
        for i in range(3):
            box = self.box_max[i] - self.box_min[i]
            # 防御：box 非法时跳过
            if box <= 0:
                continue

            delta = p[i] - self.pO[i]

            # 最近镜像映射（等价于 v2 里的逻辑，但没有 while 死循环）
            if delta > box / 2:
                p[i] -= box
            elif delta < -box / 2:
                p[i] += box

        return p



def calc_1_frame(data_raw: list[str], 
                 h2o_num: int, 
                 data_start_line: int) -> tuple[list[H2O], np.ndarray, np.ndarray]:
    """
    解析单个时间帧的数据，将原子坐标组装成水分子对象列表

    参数:
        data_raw: 当前帧的所有行（header + 原子行）
        h2o_num: 水分子数量（每个水分子包括3个原子）
        data_start_line: 当前帧在 data_raw 中的起始行号（一般为0）

    返回:
        (当前帧所有水分子的列表, 本帧盒子下界 box_min, 本帧盒子上界 box_max)
    """
    global min_pos, max_pos

    # TIMESTEP 开头算起的第 5, 6, 7 行是 box 边界
    box_bounds = data_raw[data_start_line + 5: data_start_line + 8]
    # 每行格式：lo hi [可选标签]，我们只取前两个数
    xlo, xhi = map(float, box_bounds[0].split()[:2])
    ylo, yhi = map(float, box_bounds[1].split()[:2])
    zlo, zhi = map(float, box_bounds[2].split()[:2])

    # 本帧盒子
    frame_box_min = np.array([xlo, ylo, zlo])
    frame_box_max = np.array([xhi, yhi, zhi])

    # 如果你还想保留「整个轨迹的全局 min/max」，可以继续这样更新：
    if min_pos is None:
        min_pos = frame_box_min.copy()
    else:
        min_pos = np.minimum(min_pos, frame_box_min)

    if max_pos is None:
        max_pos = frame_box_max.copy()
    else:
        max_pos = np.maximum(max_pos, frame_box_max)

    # 原子坐标起始行：头 9 行之后
    line_id = data_start_line + 9
    data: list[H2O] = []

    def get_pos(line: str, expected_type: int) -> np.ndarray:
        parts = line.strip().split()
        atom_type = int(parts[1])
        assert atom_type == expected_type, f"Expected type {expected_type}, got {atom_type}"
        return np.array([float(parts[2]), float(parts[3]), float(parts[4])])

    for _ in range(h2o_num):
        pO = get_pos(data_raw[line_id], 1)          # 氧原子（类型1）
        pH1 = get_pos(data_raw[line_id + 1], 2)     # 氢原子1（类型2）
        pH2 = get_pos(data_raw[line_id + 2], 2)     # 氢原子2（类型2）
        data.append(H2O(pO, pH1, pH2, frame_box_min, frame_box_max))
        line_id += 3                                # 下一个水分子

    return data, frame_box_min, frame_box_max


# ================== 读取所有时间帧 ==================


frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
frame_boxes: list[tuple[np.ndarray, np.ndarray]] = []  # 新增：记录每一帧的 (box_min, box_max)

with open(data_path, "r") as f:
    total_line_count = sum(1 for _ in f)

process = tqdm(total=total_line_count, desc="Reading trajectory")

with open(data_path, "r") as f:
    while True:
        # 读取一帧的头信息（9 行）
        head = []
        for _ in range(9):
            line = f.readline()
            process.update()
            if not line:
                head = []
                break
            head.append(line)

        if not head:
            process.close()
            break

        assert "ITEM: TIMESTEP" in head[0]

        natoms = int(head[3])     # 这一帧原子总数
        h2o_num = natoms // 3     # 水分子数

        # 精确逐行读取 natoms 行
        atom_lines = []
        for _ in range(natoms):
            line = f.readline()
            if not line:
                break
            atom_lines.append(line)
            process.update()

        if len(atom_lines) < natoms:
            # 文件被截断或 test 文件不完整，直接结束
            process.close()
            break

        # 解析本帧：得到 H2O 列表 + 本帧盒子
        frame_data, box_min_frame, box_max_frame = calc_1_frame(head + atom_lines, h2o_num, 0)

        # 保存本帧所有水分子坐标
        frames.append((
            np.array([h2o.pO for h2o in frame_data]),
            np.array([h2o.pH1 for h2o in frame_data]),
            np.array([h2o.pH2 for h2o in frame_data]),
        ))
        # 保存本帧盒子
        frame_boxes.append((box_min_frame, box_max_frame))

process.close()

# 确认所有帧的水分子数一致
assert max(f[0].shape[0] for f in frames) == min(f[0].shape[0] for f in frames)

print(f"Total frames parsed: {len(frames)} with {frames[0][0].shape[0]} H2O each.")
# 例如打印第 0 帧的盒子
print("frame 0 box bounds:", frame_boxes[0][0], frame_boxes[0][1])

# 确认所有帧的水分子数一致
assert max(f[0].shape[0] for f in frames) == min(f[0].shape[0] for f in frames)

print(f"Total frames parsed: {len(frames)} with {frames[0][0].shape[0]} H2O each.")
print("box bounds:", min_pos, max_pos)

# ================== 按帧分层（基于水分子 z 范围，而不是 box） ==================

layers: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = [[] for _ in range(layer_num)]

# 记录“所有帧的水分子 z_min / z_max”（后面画图时可以用）
global_z_min = np.inf
global_z_max = -np.inf

for frame_idx, (pO, pH1, pH2) in enumerate(frames):
    if layer_num <= 0:
        continue

    # 以氧原子为代表，取这一帧水分子的 z 最小值 / 最大值
    frame_z_lo = float(pO[:, 2].min())
    frame_z_hi = float(pO[:, 2].max())

    # 更新全局水分子 z 范围（可选，用于画图的 z 标注）
    global_z_min = min(global_z_min, frame_z_lo)
    global_z_max = max(global_z_max, frame_z_hi)

    dz = (frame_z_hi - frame_z_lo) / layer_num

    # 若这一帧所有 O 刚好 z 一样（极端情况），就把整帧都放到中间层
    if dz <= 0:
        mid_layer = layer_num // 2
        for layer_idx in range(layer_num):
            if layer_idx == mid_layer:
                layers[layer_idx].append((pO.copy(), pH1.copy(), pH2.copy()))
            else:
                layers[layer_idx].append(
                    (np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)))
                )
        continue

    # 正常情况：按当前帧水分子 z 范围均匀分 layer_num 层
    for layer_idx in range(layer_num):
        low = frame_z_lo + layer_idx * dz
        if layer_idx < layer_num - 1:
            high = frame_z_lo + (layer_idx + 1) * dz
            mask = (pO[:, 2] >= low) & (pO[:, 2] < high)
        else:
            # 最后一层包含 z_hi
            high = frame_z_hi
            mask = (pO[:, 2] >= low) & (pO[:, 2] <= high)

        layers[layer_idx].append((pO[mask], pH1[mask], pH2[mask]))


def compute_dipole_angles_for_layer(pO_layer: np.ndarray, 
                                    pH1_layer: np.ndarray, 
                                    pH2_layer: np.ndarray) -> np.ndarray:
    """
    计算一层中所有水分子的偶极矩与z轴（平面法向）的夹角（单位：度）
    
    偶极矩定义：从两个氢原子电荷中心指向氧原子的平均位置，向量为 (2O - (H1 + H2))
    夹角范围：0~180度（区分偶极矩向上/向下）
    
    参数:
        pO_layer: 该层所有氧原子坐标，shape=(N,3)
        pH1_layer: 该层所有氢原子1坐标，shape=(N,3)
        pH2_layer: 该层所有氢原子2坐标，shape=(N,3)
    
    返回:
        夹角数组，shape=(N,)
    """
    if pO_layer.shape[0] == 0:  # 若该层无水分子，返回空数组
        return np.array([])

    # 偶极矩方向向量（这里不关心绝对大小，只要方向）
    dipole = 2 * pO_layer - (pH1_layer + pH2_layer)  # shape=(N,3)
    
    # z轴方向（平面法向）
    z_axis = np.array([0, 0, 1])
    
    # 计算每个偶极矩与z轴的夹角
    z = dipole[:, 2]  # 偶极矩在z方向上的分量
    len_dipole = np.linalg.norm(dipole, axis=1)  # 偶极矩长度
    cos_angle = z / len_dipole  # 夹角的余弦
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 数值稳定性处理，防止超出[-1,1]
    
    angle = np.arccos(cos_angle)  # 得到弧度
    return angle / np.pi * 180  # 转换成度


def vec_angle(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    计算两向量间夹角，返回角度（单位：deg）
    
    参数:
        vec1, vec2: 输入的三个维坐标向量
    
    返回:
        两向量之间夹角（0~180度）
    """
    d_prod = (vec1 * vec2).sum()  # 点积
    angle = np.arccos(d_prod / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))  # 夹角弧度
    return angle / np.pi * 180  # 转为度


def compute_tilt_angles_for_layer(pO_layer: np.ndarray, 
                                  pH1_layer: np.ndarray, 
                                  pH2_layer: np.ndarray) -> np.ndarray:
    """
    计算一层中所有水分子的倾斜角（tilt angle），及其与xy平面的夹角
    
    定义：
        - 偶极矩d是从两个氢原子电荷中心指向氧原子的位置向量：d = r_O - (r_H1 + r_H2)/2
        - 倾斜角θ：偶极矩与z轴的夹角

    参数:
        pO_layer: 氧原子坐标，shape=(N,3)
        pH1_layer: 氢原子1坐标，shape=(N,3)
        pH2_layer: 氢原子2坐标，shape=(N,3)
    
    返回:
        倾斜角数组（单位：度），shape=(N,)
    """
    N = pO_layer.shape[0]
    if N == 0:  # 如果当前层没有水分子
        return np.array([])

    # 电荷中心（氢原子的平均位置）
    h_center = (pH1_layer + pH2_layer) / 2.0  # shape=(N,3)
    # 偶极矩向量
    dipole = pO_layer - h_center  # shape=(N,3)

    # z方向单位向量
    z = np.array([0, 0, 1])

    # 每个偶极矩与z轴的夹角
    cos_theta = dipole @ z / np.linalg.norm(dipole, axis=1)  # shape=(N,)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 防止数值误差
    theta = np.arccos(cos_theta)  # 弧度

    return theta / np.pi * 180  # 角度


def compute_plane_angles_for_layer(pO_layer: np.ndarray, 
                                   pH1_layer: np.ndarray, 
                                   pH2_layer: np.ndarray) -> np.ndarray:
    """
    计算一层中每个水分子平面与z轴的夹角（单位：度）
    
    定义：
        - 水分子平面由 (O, H1, H2) 三点构成
        - 法向量 n = (H1 - O) x (H2 - O)
        - 平面法向量与z轴之间的夹角
    
    参数:
        pO_layer: 氧原子坐标，shape=(N,3)
        pH1_layer: 氢原子1坐标，shape=(N,3)
        pH2_layer: 氢原子2坐标，shape=(N,3)
    
    返回:
        夹角数组（单位：度），shape=(N,)
    """
    N = pO_layer.shape[0]
    if N == 0:
        return np.array([])

    # 构造两个向量：OH1 和 OH2
    OH1 = pH1_layer - pO_layer
    OH2 = pH2_layer - pO_layer

    # 每个水分子平面的法向量 n = OH1 x OH2
    normals = np.cross(OH1, OH2)  # shape=(N,3)

    z = np.array([0, 0, 1])  # z轴方向
    # 计算每个法向量与z轴的夹角
    # angle = arccos( n_z / |n| )
    norm_normals = np.linalg.norm(normals, axis=1)
    # 去除法向量长度为0的情况（理论上不应该发生）
    valid_mask = norm_normals > 0
    normals = normals[valid_mask]
    norm_normals = norm_normals[valid_mask]

    cos_angle = normals[:, 2] / norm_normals
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 数值稳定性
    angle = np.arccos(cos_angle)

    return angle / np.pi * 180  # 转为度


# 针对每一层计算所有帧的倾斜角分布
angle_dipole : list[list[np.ndarray]] = [[] for _ in range(layer_num)]  # 偶极矩角度
angle_tilt   : list[list[np.ndarray]] = [[] for _ in range(layer_num)]  # 倾斜角
angle_plane  : list[list[np.ndarray]] = [[] for _ in range(layer_num)]  # 平面法向角

# 遍历每一层、每一帧，计算各种角度
sum=0
for layer_idx in range(layer_num):
    s=0
    #for frame_idx in tqdm(range(len(frames)), desc=f"Processing layer {layer_idx}/{layer_num}"):
    for frame_idx in range(len(frames)):
        pO_layer, pH1_layer, pH2_layer = layers[layer_idx][frame_idx]
        s+=pO_layer.shape[0]
        # 如果该层该帧没有水分子，则跳过
        if pO_layer.shape[0] == 0:
            angle_dipole[layer_idx].append(np.array([]))
            angle_tilt[layer_idx].append(np.array([]))
            angle_plane[layer_idx].append(np.array([]))
            continue

        # 偶极矩与z轴的夹角
        angle_dipole[layer_idx].append(
            compute_dipole_angles_for_layer(pO_layer, pH1_layer, pH2_layer)
        )

        # 倾斜角
        angle_tilt[layer_idx].append(
            compute_tilt_angles_for_layer(pO_layer, pH1_layer, pH2_layer)
        )

        # 水分子平面法向与z轴的夹角
        angle_plane[layer_idx].append(
            compute_plane_angles_for_layer(pO_layer, pH1_layer, pH2_layer)
        )
    #print(s)
    sum+=s
# 统计并绘制每一层的各类角度分布（示例：偶极矩角度分布）
print(sum)
theta_bins = np.linspace(0, 180, 181)  # 0~180度按1度划分
theta_mid = (theta_bins[:-1] + theta_bins[1:]) / 2

# 对每层计算角度分布（直方图归一化为概率分布）
dipole_hist_per_layer = []
tilt_hist_per_layer   = []
plane_hist_per_layer  = []

for layer_idx in range(layer_num):
    # 将该层所有帧的角度数据拼接
    all_dipole_angles = np.concatenate(angle_dipole[layer_idx]) if len(angle_dipole[layer_idx]) else np.array([])
    all_tilt_angles   = np.concatenate(angle_tilt[layer_idx])   if len(angle_tilt[layer_idx]) else np.array([])
    all_plane_angles  = np.concatenate(angle_plane[layer_idx])  if len(angle_plane[layer_idx]) else np.array([])

    # 如果该层完全没有数据，则记录空
    if all_dipole_angles.size == 0:
        dipole_hist_per_layer.append(np.zeros_like(theta_mid))
    else:
        hist, _ = np.histogram(all_dipole_angles, bins=theta_bins, density=True)
        dipole_hist_per_layer.append(hist)

    if all_tilt_angles.size == 0:
        tilt_hist_per_layer.append(np.zeros_like(theta_mid))
    else:
        hist, _ = np.histogram(all_tilt_angles, bins=theta_bins, density=True)
        tilt_hist_per_layer.append(hist)

    if all_plane_angles.size == 0:
        plane_hist_per_layer.append(np.zeros_like(theta_mid))
    else:
        hist, _ = np.histogram(all_plane_angles, bins=theta_bins, density=True)
        plane_hist_per_layer.append(hist)

# ================== 统计角度分布 ==================
print(sum)
theta_bins = np.linspace(0, 180, 181)  # 0~180度按1度划分
theta_mid = (theta_bins[:-1] + theta_bins[1:]) / 2

# 对每层计算角度分布（直方图归一化为概率分布）
dipole_hist_per_layer = []
tilt_hist_per_layer   = []
plane_hist_per_layer  = []

for layer_idx in range(layer_num):
    # 将该层所有帧的角度数据拼接
    all_dipole_angles = np.concatenate(angle_dipole[layer_idx]) if len(angle_dipole[layer_idx]) else np.array([])
    all_tilt_angles   = np.concatenate(angle_tilt[layer_idx])   if len(angle_tilt[layer_idx]) else np.array([])
    all_plane_angles  = np.concatenate(angle_plane[layer_idx])  if len(angle_plane[layer_idx]) else np.array([])

    # 如果该层完全没有数据，则记录空
    if all_dipole_angles.size == 0:
        dipole_hist_per_layer.append(np.zeros_like(theta_mid))
    else:
        hist, _ = np.histogram(all_dipole_angles, bins=theta_bins, density=True)
        dipole_hist_per_layer.append(hist)

    if all_tilt_angles.size == 0:
        tilt_hist_per_layer.append(np.zeros_like(theta_mid))
    else:
        hist, _ = np.histogram(all_tilt_angles, bins=theta_bins, density=True)
        tilt_hist_per_layer.append(hist)

    if all_plane_angles.size == 0:
        plane_hist_per_layer.append(np.zeros_like(theta_mid))
    else:
        hist, _ = np.histogram(all_plane_angles, bins=theta_bins, density=True)
        plane_hist_per_layer.append(hist)


# ================== 一些通用的小工具函数 ==================

def choose_layers_for_overlay(num_layers: int, max_lines: int = 10) -> list[int]:
    """
    自动选择若干层用来 overlay 画曲线：
    - 如果层数少于 max_lines，则全部画
    - 否则均匀抽样
    """
    if num_layers <= max_lines:
        return list(range(num_layers))
    step = max(1, num_layers // max_lines)
    return list(range(0, num_layers, step))


'''# ================== 画图 1：偶极矩角度分布（多层 overlay） ==================

selected_layers = choose_layers_for_overlay(layer_num, max_lines=10)

plt.figure(figsize=(8, 6))
for layer_idx in selected_layers:
    prob = dipole_hist_per_layer[layer_idx]
    if np.all(prob == 0):
        continue
    plt.plot(theta_mid, prob, label=f'Layer {layer_idx}')

plt.xlabel(r'$\theta_{\mu-z}$ (deg)')
plt.ylabel('Probability')
plt.title('Dipole angle distribution (selected layers)')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("dipole_angle_overlay.png", dpi=300)
plt.close()


# ================== 画图 2：倾斜角分布（多层 overlay） ==================

plt.figure(figsize=(8, 6))
for layer_idx in selected_layers:
    prob = tilt_hist_per_layer[layer_idx]
    if np.all(prob == 0):
        continue
    plt.plot(theta_mid, prob, label=f'Layer {layer_idx}')

plt.xlabel(r'$\theta_{\mathrm{tilt}}$ (deg)')
plt.ylabel('Probability')
plt.title('Tilt angle distribution (selected layers)')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("tilt_angle_overlay.png", dpi=300)
plt.close()


# ================== 画图 3：平面法向角分布（多层 overlay） ==================

plt.figure(figsize=(8, 6))
for layer_idx in selected_layers:
    prob = plane_hist_per_layer[layer_idx]
    if np.all(prob == 0):
        continue
    plt.plot(theta_mid, prob, label=f'Layer {layer_idx}')

plt.xlabel(r'$\theta_{\mathrm{plane-z}}$ (deg)')
plt.ylabel('Probability')
plt.title('Plane normal angle distribution (selected layers)')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("plane_angle_overlay.png", dpi=300)
plt.close()'''


'''# ================== 画图 4：每层单独子图（偶极矩角度分布，可自适应 layer_num） ==================

# 若层数太多，可以视情况关掉这一块
ncols = 4
nrows = int(np.ceil(layer_num / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                         figsize=(3.2 * ncols, 2.4 * nrows),
                         sharex=True, sharey=False)
axes = np.array(axes).reshape(-1)  # 展平，方便索引

for layer_idx in range(layer_num):
    ax = axes[layer_idx]
    prob = dipole_hist_per_layer[layer_idx]
    if np.all(prob == 0):
        ax.set_visible(False)
        continue

    ax.plot(theta_mid, prob)
    ax.set_title(f'Layer {layer_idx}', fontsize=8)
    ax.set_xlim(0, 180)
    prob_max = np.max(prob)
    ax.set_ylim(0, prob_max * 1.1 if prob_max > 0 else 1.0)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.grid(True, alpha=0.3)

# 多出来的 axes（如果有）关掉
for i in range(layer_num, len(axes)):
    axes[i].set_visible(False)

fig.text(0.5, 0.04, r'$\theta_{\mu-z}$ (deg)', ha='center')
fig.text(0.04, 0.5, 'Probability', va='center', rotation='vertical')

plt.tight_layout(rect=[0.06, 0.06, 1, 1])
plt.savefig("dipole_angle_grid.png", dpi=300)
plt.close()'''


# ============================================================
# 新功能：计算每层的平均角度（偶极矩、倾斜角、平面角）
# ============================================================

avg_angle_dipole = np.zeros(layer_num)
avg_angle_tilt   = np.zeros(layer_num)
avg_angle_plane  = np.zeros(layer_num)

for layer_idx in range(layer_num):

    # 拼接该层所有帧的角度（忽略空帧）
    all_dipole = np.concatenate(angle_dipole[layer_idx]) if len(angle_dipole[layer_idx]) else np.array([])
    all_tilt   = np.concatenate(angle_tilt[layer_idx])   if len(angle_tilt[layer_idx]) else np.array([])
    all_plane  = np.concatenate(angle_plane[layer_idx])  if len(angle_plane[layer_idx]) else np.array([])

    # 避免空层导致 nan
    avg_angle_dipole[layer_idx] = all_dipole.mean() if all_dipole.size > 0 else np.nan
    avg_angle_tilt[layer_idx]   = all_tilt.mean()   if all_tilt.size > 0 else np.nan
    avg_angle_plane[layer_idx]  = all_plane.mean()  if all_plane.size > 0 else np.nan

'''# 打印结果
print("\n===== 每层平均角度统计结果（单位：度） =====")
for i in range(layer_num):
    print(f"Layer {i:2d}: Dipole={avg_angle_dipole[i]:7.2f}°, "
          f"Tilt={avg_angle_tilt[i]:7.2f}°, "
          f"Plane={avg_angle_plane[i]:7.2f}°")

# 保存为 txt 文件
np.savetxt("average_angles_per_layer.txt",
           np.vstack([avg_angle_dipole, avg_angle_tilt, avg_angle_plane]).T,
           header="Dipole_Avg  Tilt_Avg  Plane_Avg  (degrees)")

print("\n已保存：average_angles_per_layer.txt")'''

plt.figure(figsize=(8, 5))
plt.plot(
    range(layer_num),
    avg_angle_dipole,
    marker='o',
    linestyle='-',
    linewidth=1.8,
    markersize=5,
    label='Average dipole angle'
)

plt.xlabel("Layer index")
plt.ylabel("Average dipole–normal angle (deg)")
plt.title("Average dipole orientation vs layer")
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig("average_dipole_angle_vs_layer.png", dpi=300)
plt.show()
plt.close()



def plot_angle_distribution_for_layer(layer_idx, angle_list, angle_name="dipole"):
    """
    绘制当前层的角度分布直方图。
    
    参数：
        layer_idx : int       想绘制的层号
        angle_list: list[np.ndarray]  该层所有帧的角度数组
        angle_name: str       名称 ("dipole" / "tilt" / "plane")
    """

    # 将该层所有帧角度拼接
    if len(angle_list[layer_idx]) == 0:
        print(f"Layer {layer_idx} has no data.")
        return
    
    angles = np.concatenate(angle_list[layer_idx])
    if angles.size == 0:
        print(f"Layer {layer_idx} has no angles.")
        return

    # 自动生成 bin
    theta_bins = np.linspace(0, 180, 181)
    theta_mid = (theta_bins[:-1] + theta_bins[1:]) / 2
    hist, _ = np.histogram(angles, bins=theta_bins, density=True)

    # 绘图
    plt.figure(figsize=(7, 5))
    plt.bar(theta_mid, hist, width=1.0, alpha=0.7, edgecolor='k')
    plt.xlabel("Angle (deg)")
    plt.ylabel("Probability density")
    plt.title(f"{angle_name.capitalize()} angle distribution in layer {layer_idx}")
    plt.savefig(f"{angle_name.capitalize()} angle distribution in layer {layer_idx}.png")
    plt.tight_layout()
    plt.show()

g=[0,1,2,5,10,989,994,997,998,999]
for i in g:
    plot_angle_distribution_for_layer(i, angle_dipole, angle_name="dipole")
