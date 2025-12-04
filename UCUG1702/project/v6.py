import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ================== 盒子边界（由 calc_1_frame 使用） ==================
max_pos = None  # np.ndarray shape=(3,)
min_pos = None  # np.ndarray shape=(3,)


class H2O:
    """水分子类：存一个水分子的 O、H1、H2 坐标，并在初始化时做一次 PBC 修正。"""

    def __init__(
        self,
        pO: np.ndarray,
        pH1: np.ndarray,
        pH2: np.ndarray,
        box_min: np.ndarray,
        box_max: np.ndarray,
    ) -> None:
        # 保存本帧盒子边界
        self.box_min = box_min.copy()
        self.box_max = box_max.copy()

        # 原子坐标
        self.pO = pO.copy()
        self.pH1 = pH1.copy()
        self.pH2 = pH2.copy()

        # 在 box 信息已知时做 PBC 修正（只修 H，O 保持原坐标）
        self.pH1 = self._fix_pH(self.pH1)
        self.pH2 = self._fix_pH(self.pH2)

    def _fix_pH(self, pH: np.ndarray) -> np.ndarray:
        """
        修正氢原子坐标以符合周期性边界条件：
        如果 H 相对 O 在某一维度的差值超过 box_length/2，就 ±box_length 拉回最近镜像。
        """
        p = pH.copy()
        for i in range(3):
            box = self.box_max[i] - self.box_min[i]
            if box <= 0:
                continue

            delta = p[i] - self.pO[i]

            if delta > box / 2:
                p[i] -= box
            elif delta < -box / 2:
                p[i] += box

        return p


def calc_1_frame(
    data_raw: list[str], h2o_num: int, data_start_line: int
) -> tuple[list[H2O], np.ndarray, np.ndarray]:
    """
    解析单个时间帧的数据，将原子坐标组装成水分子对象列表

    参数:
        data_raw: 当前帧的所有行（header + 原子行）
        h2o_num: 水分子数量（每个水分子包括 3 个原子）
        data_start_line: 当前帧在 data_raw 中的起始行号（一般为 0）
    """
    global min_pos, max_pos

    # TIMESTEP 开头算起的第 5, 6, 7 行是 box 边界
    box_bounds = data_raw[data_start_line + 5 : data_start_line + 8]
    xlo, xhi = map(float, box_bounds[0].split()[:2])
    ylo, yhi = map(float, box_bounds[1].split()[:2])
    zlo, zhi = map(float, box_bounds[2].split()[:2])

    # 本帧盒子
    frame_box_min = np.array([xlo, ylo, zlo])
    frame_box_max = np.array([xhi, yhi, zhi])

    # 更新全局盒子边界（仅用于打印）
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
        pO = get_pos(data_raw[line_id], 1)  # 氧原子（类型 1）
        pH1 = get_pos(data_raw[line_id + 1], 2)  # 氢原子 1（类型 2）
        pH2 = get_pos(data_raw[line_id + 2], 2)  # 氢原子 2（类型 2）
        data.append(H2O(pO, pH1, pH2, frame_box_min, frame_box_max))
        line_id += 3

    return data, frame_box_min, frame_box_max


# ================== 角度计算函数 ==================


def compute_dipole_angles_for_layer(
    pO_layer: np.ndarray, pH1_layer: np.ndarray, pH2_layer: np.ndarray
) -> np.ndarray:
    """
    计算一批水分子的偶极矩与 z 轴（平面法向）的夹角（单位：度）

    偶极矩定义：从两个氢原子电荷中心指向氧原子的平均位置，向量为 (2O - (H1 + H2))
    夹角范围：0~180 度
    """
    if pO_layer.shape[0] == 0:
        return np.array([])

    dipole = 2 * pO_layer - (pH1_layer + pH2_layer)  # shape=(N,3)
    z_axis = np.array([0, 0, 1])

    z_comp = dipole[:, 2]
    len_dipole = np.linalg.norm(dipole, axis=1)
    cos_angle = z_comp / len_dipole
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle = np.arccos(cos_angle)
    return angle / np.pi * 180.0


# ================== 画图函数（原有三个 + PDF） ==================
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d


def plot_angle_vs_x_line(
    x_sorted: np.ndarray,
    theta_sorted: np.ndarray,
    n_bins: int = 1000,
    filename: str = "average_dipole_angle_vs_relative_z.png",
):
    """
    按横轴指标等间距分成 n_bins 个区间，
    每个区间取平均偶极矩角度，画成折线图。
    """
    x_min, x_max = float(x_sorted.min()), float(x_sorted.max())
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)

    # 为了方便统计，利用已排序好的 x_sorted
    indices = np.searchsorted(x_sorted, bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    mean_angles = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        start, end = indices[i], indices[i + 1]
        if end > start:
            seg = theta_sorted[start:end]
            mean_angles[i] = seg.mean()
            counts[i] = end - start

    # 去掉完全没有数据的 bin
    valid_mask = ~np.isnan(mean_angles)
    x_plot = bin_centers[valid_mask]
    y_plot = mean_angles[valid_mask]

    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, y_plot, linestyle="-", linewidth=1.2)
    plt.xlabel("Relative z (z - <z>)")
    plt.ylabel("Average dipole–z angle (deg)")
    plt.title("Average dipole orientation vs relative z")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")


def plot_cos_theta_vs_x_spline(
    x_sorted: np.ndarray,
    theta_sorted: np.ndarray,
    n_bins: int = 300,
    smooth_factor: float = 0.01,
    filename: str = "cos_theta_vs_relative_z_spline.png",
):
    """
    1. theta_sorted(度) -> cos(theta)
    2. 对 x_sorted 按等距 bin 取 <cos theta>
    3. 用平滑样条做拟合
    4. 画出光滑曲线（没有 mark 点）
    """
    # 1. 角度 -> cos(theta)
    theta_rad = theta_sorted / 180.0 * np.pi
    cos_theta = np.cos(theta_rad)

    # 2. 分 bin 计算 <cos theta>
    x_min, x_max = float(x_sorted.min()), float(x_sorted.max())
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    indices = np.searchsorted(x_sorted, bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    cos_mean = np.full(n_bins, np.nan)
    for i in range(n_bins):
        start, end = indices[i], indices[i + 1]
        if end > start:
            cos_mean[i] = cos_theta[start:end].mean()

    # 去掉没有数据的 bin
    mask = ~np.isnan(cos_mean)
    x_bin = bin_centers[mask]
    y_bin = cos_mean[mask]

    # 3. 平滑样条拟合
    spline = UnivariateSpline(x_bin, y_bin, s=smooth_factor * len(x_bin))

    x_fine = np.linspace(x_bin.min(), x_bin.max(), 2000)
    y_fine = spline(x_fine)

    y_fine = np.clip(y_fine, -1.0, 1.0)  # 把所有值限制在 [-1,1]

    # 4. 画图
    plt.figure(figsize=(8, 5))
    plt.plot(x_fine, y_fine, linestyle="-", linewidth=2.0)

    plt.xlabel("z (relative)")
    plt.ylabel(r"$\langle \cos\theta_{\mathrm{OH},z} \rangle$")
    plt.title(r"$\langle \cos\theta \rangle$ vs z (spline-smoothed)")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")


def plot_cos_theta_vs_x_gaussian(
    x_sorted: np.ndarray,
    theta_sorted: np.ndarray,
    n_bins: int = 400,
    sigma: float = 2.0,
    filename: str = "cos_theta_vs_relative_z_gaussian.png",
):
    # 角度 -> cosθ
    theta_rad = np.deg2rad(theta_sorted)
    cos_theta = np.cos(theta_rad)

    # 按 x 分 bin，求每个 bin 的 <cosθ>
    x_min, x_max = float(x_sorted.min()), float(x_sorted.max())
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    indices = np.searchsorted(x_sorted, bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    cos_mean = np.full(n_bins, np.nan)
    for i in range(n_bins):
        s, e = indices[i], indices[i + 1]
        if e > s:
            cos_mean[i] = cos_theta[s:e].mean()

    mask = ~np.isnan(cos_mean)
    x_bin = bin_centers[mask]
    y_bin = cos_mean[mask]

    # 高斯平滑（线性操作）
    y_smooth = gaussian_filter1d(y_bin, sigma=sigma)

    # 画图：无 marker 的平滑曲线
    plt.figure(figsize=(8, 5))
    plt.plot(x_bin, y_smooth, linewidth=2.0)
    plt.xlabel("z (relative)")
    plt.ylabel(r"$\langle \cos\theta_{\mathrm{OH},z} \rangle$")
    plt.title(r"$\langle \cos\theta \rangle$ vs z (gaussian-smoothed)")
    plt.grid(True, alpha=0.35)
    plt.ylim(-1.0, 1.0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print("Saved:", filename)


def plot_angle_pdf_in_percent_range(
    x_sorted: np.ndarray,
    theta_sorted: np.ndarray,
    p_low: float,
    p_high: float,
    angle_name: str = "dipole",
    filename_prefix: str = "dipole_angle_pdf",
):
    """
    在横轴指标范围内取 [p_low, p_high] 这一段（按“坐标范围百分比”），
    对其中的水分子偶极矩角度画概率密度直方图。
    """
    assert 0.0 <= p_low < p_high <= 1.0

    x_min, x_max = float(x_sorted.min()), float(x_sorted.max())
    x_lo = x_min + p_low * (x_max - x_min)
    x_hi = x_min + p_high * (x_max - x_min)

    mask = (x_sorted >= x_lo) & (x_sorted < x_hi)
    if not np.any(mask):
        print(
            f"No data in x range [{x_lo:.4f}, {x_hi:.4f}] "
            f"({p_low:.2f}–{p_high:.2f} of full range)."
        )
        return

    angles = theta_sorted[mask]

    theta_bins = np.linspace(0, 180, 181)
    theta_mid = (theta_bins[:-1] + theta_bins[1:]) / 2
    hist, _ = np.histogram(angles, bins=theta_bins, density=True)

    plt.figure(figsize=(7, 5))
    plt.bar(theta_mid, hist, width=1.0, alpha=0.7, edgecolor="k")
    plt.xlabel("Dipole–z angle (deg)")
    plt.ylabel("Probability density")
    plt.title(
        f"{angle_name.capitalize()} angle PDF for x in "
        f"[{p_low:.2f}, {p_high:.2f}] of range\n"
        f"(x ∈ [{x_lo:.3f}, {x_hi:.3f}])"
    )
    out_name = f"{filename_prefix}_{int(p_low * 100)}_{int(p_high * 100)}.png"
    plt.tight_layout()
    plt.savefig(out_name, dpi=300)
    plt.close()
    print(f"Saved: {out_name}")


# ================== 读取单个轨迹 + 计算 x_sorted, theta_sorted ==================


def load_and_compute(data_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    读取一个 LAMMPS 轨迹文件，返回拼接后排序好的
    x_sorted（相对 z）和 theta_sorted（偶极矩-法线夹角，度）
    """
    global min_pos, max_pos

    # 每个文件重置盒子边界
    min_pos = None
    max_pos = None

    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    frame_boxes: list[tuple[np.ndarray, np.ndarray]] = []

    # 先统计总行数用于 tqdm
    with open(data_path, "r") as f:
        total_line_count = sum(1 for _ in f)

    process = tqdm(total=total_line_count, desc=f"Reading trajectory {data_path}")

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

            natoms = int(head[3])  # 这一帧原子总数
            h2o_num = natoms // 3  # 水分子数

            # 精确逐行读取 natoms 行
            atom_lines = []
            for _ in range(natoms):
                line = f.readline()
                if not line:
                    break
                atom_lines.append(line)
                process.update()

            if len(atom_lines) < natoms:
                # 文件被截断或测试文件不完整
                process.close()
                break

            # 解析本帧：得到 H2O 列表 + 本帧盒子
            frame_data, box_min_frame, box_max_frame = calc_1_frame(
                head + atom_lines, h2o_num, 0
            )

            # 保存本帧所有水分子坐标（按 O/H1/H2 分开放）
            frames.append(
                (
                    np.array([h2o.pO for h2o in frame_data]),
                    np.array([h2o.pH1 for h2o in frame_data]),
                    np.array([h2o.pH2 for h2o in frame_data]),
                )
            )
            frame_boxes.append((box_min_frame, box_max_frame))

    process.close()

    # 确认所有帧的水分子数一致
    assert max(f[0].shape[0] for f in frames) == min(f[0].shape[0] for f in frames)

    print(f"Total frames parsed: {len(frames)} with {frames[0][0].shape[0]} H2O each.")
    print("Global box bounds:", min_pos, max_pos)
    print("Frame 0 box bounds:", frame_boxes[0][0], frame_boxes[0][1])

    # === 生成 x_sorted, theta_sorted ===
    all_x_list = []  # 存所有帧的“横轴指标”
    all_theta_list = []  # 存所有帧的偶极矩角度（度）

    for frame_idx, (pO, pH1, pH2) in enumerate(frames):
        # 以 O 的 z 作为水分子 z 坐标
        z = pO[:, 2]
        z0 = z.mean()  # 当前帧 z 均值 -> 零点
        x_frame = z - z0  # 当前帧的“横轴指标”

        # 当前帧全部水分子的偶极矩–z 夹角
        theta_frame = compute_dipole_angles_for_layer(pO, pH1, pH2)

        # 对当前帧按横轴指标排序（仅体现在保存顺序上）
        order = np.argsort(x_frame)
        x_frame_sorted = x_frame[order]
        theta_frame_sorted = theta_frame[order]

        all_x_list.append(x_frame_sorted)
        all_theta_list.append(theta_frame_sorted)

    # 综合所有时刻，整体拼接并按照横轴指标重新排序
    all_x = np.concatenate(all_x_list)
    all_theta = np.concatenate(all_theta_list)

    order_global = np.argsort(all_x)
    x_sorted = all_x[order_global]
    theta_sorted = all_theta[order_global]

    print(f"Total water molecules (all frames): {x_sorted.shape[0]}")
    print(f"x range: [{x_sorted.min():.4f}, {x_sorted.max():.4f}]")

    return x_sorted, theta_sorted


# ================== 三组数据对比图（并列子图） ==================


def plot_compare_angle_vs_x(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    n_bins: int = 400,
    filename: str = "compare_average_dipole_angle_vs_relative_z_3T.png",
):
    """
    三温度的：平均角度 vs 相对 z，三张图并列
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, (label, (x_sorted, theta_sorted)) in zip(axes, results.items()):
        x_min, x_max = float(x_sorted.min()), float(x_sorted.max())
        bin_edges = np.linspace(x_min, x_max, n_bins + 1)
        indices = np.searchsorted(x_sorted, bin_edges)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        mean_angles = np.full(n_bins, np.nan)
        for i in range(n_bins):
            s, e = indices[i], indices[i + 1]
            if e > s:
                mean_angles[i] = theta_sorted[s:e].mean()

        mask = ~np.isnan(mean_angles)
        ax.plot(bin_centers[mask], mean_angles[mask], linewidth=1.2)
        ax.set_title(label)
        ax.set_xlabel("Relative z (z - <z>)")
        ax.grid(True, alpha=0.35)

    axes[0].set_ylabel("Average dipole–z angle (deg)")
    fig.suptitle("Average dipole orientation vs relative z (3 temperatures)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print("Saved:", filename)


def plot_compare_cos_theta_gaussian(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    n_bins: int = 400,
    sigma: float = 2.0,
    filename: str = "compare_cos_theta_vs_relative_z_gaussian_3T.png",
):
    """
    三温度的：<cosθ> vs z (高斯平滑)，三张图并列
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, (label, (x_sorted, theta_sorted)) in zip(axes, results.items()):
        theta_rad = np.deg2rad(theta_sorted)
        cos_theta = np.cos(theta_rad)

        x_min, x_max = float(x_sorted.min()), float(x_sorted.max())
        bin_edges = np.linspace(x_min, x_max, n_bins + 1)
        indices = np.searchsorted(x_sorted, bin_edges)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        cos_mean = np.full(n_bins, np.nan)
        for i in range(n_bins):
            s, e = indices[i], indices[i + 1]
            if e > s:
                cos_mean[i] = cos_theta[s:e].mean()

        mask = ~np.isnan(cos_mean)
        x_bin = bin_centers[mask]
        y_bin = cos_mean[mask]

        y_smooth = gaussian_filter1d(y_bin, sigma=sigma)

        ax.plot(x_bin, y_smooth, linewidth=2.0)
        ax.set_title(label)
        ax.set_xlabel("z (relative)")
        ax.grid(True, alpha=0.35)
        ax.set_ylim(-1.0, 1.0)

    axes[0].set_ylabel(r"$\langle \cos\theta_{\mathrm{OH},z} \rangle$")
    fig.suptitle(r"$\langle \cos\theta \rangle$ vs z (gaussian-smoothed, 3 temperatures)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print("Saved:", filename)


# ================== 主程序：三组数据 ==================

if __name__ == "__main__":
    # 根据你的文件实际路径调整
    data_files = {
        "280K": "./data/dump-surface-280.lammpstrj",
        "300K": "./data/dump-surface.lammpstrj",
        "320K": "./data/dump-surface-320.lammpstrj",
    }

    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # 每个温度单独做“原有所有图”
    for label, path in data_files.items():
        print("=" * 80)
        print(f"Processing {label} from {path}")
        x_sorted, theta_sorted = load_and_compute(path)
        results[label] = (x_sorted, theta_sorted)

        # --- 原有的三类图（文件名加上温度后缀） ---
        plot_angle_vs_x_line(
            x_sorted,
            theta_sorted,
            n_bins=400,
            filename=f"average_dipole_angle_vs_relative_z_{label}.png",
        )

        plot_cos_theta_vs_x_spline(
            x_sorted,
            theta_sorted,
            n_bins=400,
            filename=f"cos_theta_vs_relative_z_spline_{label}.png",
        )

        plot_cos_theta_vs_x_gaussian(
            x_sorted,
            theta_sorted,
            n_bins=400,
            sigma=2.0,
            filename=f"cos_theta_vs_relative_z_gaussian_{label}.png",
        )

        # --- 三个区间的角度 PDF（左 10%、中 10%、右 10%） ---
        percent_windows = [
            (0.00, 0.10),
            (0.45, 0.55),
            (0.90, 1.00),
        ]
        for p_low, p_high in percent_windows:
            plot_angle_pdf_in_percent_range(
                x_sorted,
                theta_sorted,
                p_low,
                p_high,
                angle_name="dipole",
                filename_prefix=f"dipole_angle_pdf_{label}",
            )

    # ===== 三组数据的对比图（三张图并列） =====
    plot_compare_angle_vs_x(results, n_bins=400)
    plot_compare_cos_theta_gaussian(results, n_bins=400, sigma=2.0)
