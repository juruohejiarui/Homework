#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对 LAMMPS 水体系 (type=1: O, type=2: H) 分析 O–H 取向和悬挂氢 (dangling H)：

- 用 O–H 距离识别水分子（每个 O 找 2 个 H）
- 对每条 O–H 键：
    * 计算与 z 轴的夹角 theta_OH_z（度）和 cos(theta_OH_z)
    * 用几何判据判断该 H 是否作为 donor 参与氢键
      (O_donor–O_acceptor 距离 < ROO_CUT 且 O–H···O 角 <= ANGLE_CUT)
      如果对任一 acceptor 满足则认为“成键”，否则“悬挂”
- 按 z (用 O 的 z 坐标) 分层，对时间取平均：
    * 所有 H 的 <theta_OH_z>、<cos theta_OH_z>
    * 悬挂 H 的 <theta_OH_z>、<cos theta_OH_z>
    * 悬挂 H 占比 f_dangling(z)

输出：
  1) dangling_OH_orientation_z_timeavg.dat
     列：
       z_center[A]
       H_all_count_sum_over_frames
       avg_theta_all_deg
       avg_cos_all
       H_dangling_count_sum_over_frames
       avg_theta_dangling_deg
       avg_cos_dangling
       fraction_dangling
  2) 两张图：
       dangling_OH_theta_z.png
       dangling_OH_costheta_z.png
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt


# ========================= 可调参数 =========================

# O–H 共价键识别
R_OH_BOND_MAX = 1.2   # Å

# 氢键判据
ROO_CUT   = 3.5       # O_donor–O_acceptor 最大距离 (Å)
ANGLE_CUT = 30.0      # O–H···O 角阈值 (deg) —— 越小越接近线性

# z 方向分层厚度
DZ = 1.0              # Å

# 最多读取的帧数；None 或 0 表示所有帧
MAX_FRAMES = None


# ========================= 读取 LAMMPS 轨迹 =========================

def iter_lammps_frames(filename):
    with open(filename, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                raise ValueError("Expected 'ITEM: TIMESTEP', got: " + line)

            timestep = int(f.readline().strip())

            line = f.readline()
            if not line.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError("Missing 'ITEM: NUMBER OF ATOMS'")
            n_atoms = int(f.readline().strip())

            line = f.readline()
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Missing 'ITEM: BOX BOUNDS'")
            xlo, xhi = map(float, f.readline().split()[:2])
            ylo, yhi = map(float, f.readline().split()[:2])
            zlo, zhi = map(float, f.readline().split()[:2])
            box = np.array([[xlo, xhi],
                            [ylo, yhi],
                            [zlo, zhi]], dtype=float)

            line = f.readline()
            if not line.startswith("ITEM: ATOMS"):
                raise ValueError("Missing 'ITEM: ATOMS'")
            cols = line.split()[2:]
            col_index = {name: i for i, name in enumerate(cols)}

            required = {"id", "type", "x", "y", "z"}
            if not required.issubset(col_index.keys()):
                raise ValueError(f"Missing columns {required}, have {col_index.keys()}")

            data = []
            for _ in range(n_atoms):
                parts = f.readline().split()
                data.append(parts)
            data = np.array(data, dtype=float)

            ids = data[:, col_index["id"]].astype(int)
            types = data[:, col_index["type"]].astype(int)
            x = data[:, col_index["x"]]
            y = data[:, col_index["y"]]
            z = data[:, col_index["z"]]
            coords = np.vstack([x, y, z]).T

            yield timestep, box, ids, types, coords


# ========================= 工具函数 =========================

def minimum_image(delta, box_lengths):
    return delta - np.round(delta / box_lengths) * box_lengths


def build_water_molecules(types, coords, box_lengths):
    O_indices = np.where(types == 1)[0]
    H_indices = np.where(types == 2)[0]
    coords_O = coords[O_indices]
    coords_H = coords[H_indices]

    water_H_indices = []
    bad_O = 0

    for o_pos in coords_O:
        d = coords_H - o_pos
        d = minimum_image(d, box_lengths)
        r = np.linalg.norm(d, axis=1)
        close = np.where(r < R_OH_BOND_MAX)[0]
        if len(close) != 2:
            bad_O += 1
        water_H_indices.append(H_indices[close])

    if bad_O > 0:
        print(f"  [注意] 有 {bad_O} 个 O 没有正好 2 个 H。")

    return O_indices, water_H_indices


def compute_OH_orientation_and_dangling(O_indices, water_H_indices,
                                        coords, box_lengths,
                                        ROO_cut=ROO_CUT, angle_cut=ANGLE_CUT):
    """
    对每条 O–H：
      - 算 v_OH 与 z 轴的夹角 theta_OH_z、cos(theta_OH_z)
      - 判断该 H 是否作为 donor 形成至少一条氢键：
            对所有 j != i 的 acceptor O_j
             若 r_OO < ROO_cut 且 O–H···O 角 <= angle_cut
             则该 H 视为 "bonded"，否则 "dangling"

    返回：
      O_z_for_H     : (n_H_in_water,)  该 H 所属 O 的 z 坐标
      theta_deg     : (n_H_in_water,)  与 z 轴夹角（度）
      cos_theta     : (n_H_in_water,)  cos(theta)
      is_dangling   : (n_H_in_water,)  bool，是否悬挂
    """
    coords_O = coords[O_indices]
    n_water = len(O_indices)

    O_z_list = []
    theta_list = []
    cos_list = []
    dangling_list = []

    for i in range(n_water):
        O_pos = coords_O[i]
        O_z = O_pos[2]
        H_globals = water_H_indices[i]

        for h_idx in H_globals:
            H_pos = coords[h_idx]

            # O->H 向量
            v_OH = minimum_image(H_pos - O_pos, box_lengths)
            norm_OH = np.linalg.norm(v_OH)
            if norm_OH == 0.0:
                continue

            # 与 z 轴夹角
            cos_theta_z = v_OH[2] / norm_OH
            cos_theta_z = max(min(cos_theta_z, 1.0), -1.0)
            theta_z = math.degrees(math.acos(cos_theta_z))

            # 判断是否作为 donor 形成氢键
            dangling = True
            for j in range(n_water):
                if j == i:
                    continue
                Oj = coords_O[j]

                # O_donor -> O_acceptor
                d_OO_vec = minimum_image(Oj - O_pos, box_lengths)
                r_OO = np.linalg.norm(d_OO_vec)
                if r_OO > ROO_cut:
                    continue

                # H -> O_acceptor
                v_HOa = minimum_image(Oj - H_pos, box_lengths)
                norm_HOa = np.linalg.norm(v_HOa)
                if norm_HOa == 0.0:
                    continue

                cos_angle = np.dot(v_OH, v_HOa) / (norm_OH * norm_HOa)
                cos_angle = max(min(cos_angle, 1.0), -1.0)
                angle = math.degrees(math.acos(cos_angle))

                if angle <= angle_cut:
                    dangling = False
                    break  # 已经成键，无需再找别的 acceptor

            O_z_list.append(O_z)
            theta_list.append(theta_z)
            cos_list.append(cos_theta_z)
            dangling_list.append(dangling)

    O_z_for_H = np.array(O_z_list, dtype=float)
    theta_deg = np.array(theta_list, dtype=float)
    cos_theta = np.array(cos_list, dtype=float)
    is_dangling = np.array(dangling_list, dtype=bool)

    return O_z_for_H, theta_deg, cos_theta, is_dangling


def analyze_z_layers_OH(O_z_for_H, theta_deg, cos_theta, is_dangling, box, dz=DZ):
    """
    按 z 分层，分别统计：
      - 所有 H 的数量/平均 θ / 平均 cosθ
      - dangling H 的数量/平均 θ / 平均 cosθ

    返回：
      z_centers
      count_all, sum_theta_all, sum_cos_all
      count_dangling, sum_theta_dangling, sum_cos_dangling
    """
    zlo, zhi = box[2]
    Lz = zhi - zlo

    n_bins = int(math.ceil(Lz / dz))
    edges = np.linspace(zlo, zhi, n_bins + 1)
    z_centers = 0.5 * (edges[:-1] + edges[1:])

    count_all = np.zeros(n_bins, dtype=float)
    sum_theta_all = np.zeros(n_bins, dtype=float)
    sum_cos_all = np.zeros(n_bins, dtype=float)

    count_dang = np.zeros(n_bins, dtype=float)
    sum_theta_dang = np.zeros(n_bins, dtype=float)
    sum_cos_dang = np.zeros(n_bins, dtype=float)

    bin_indices = np.floor((O_z_for_H - zlo) / dz).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    for i, b in enumerate(bin_indices):
        count_all[b] += 1
        sum_theta_all[b] += theta_deg[i]
        sum_cos_all[b] += cos_theta[i]

        if is_dangling[i]:
            count_dang[b] += 1
            sum_theta_dang[b] += theta_deg[i]
            sum_cos_dang[b] += cos_theta[i]

    return (z_centers,
            count_all, sum_theta_all, sum_cos_all,
            count_dang, sum_theta_dang, sum_cos_dang)


# ========================= 主程序 =========================

def main():
    if len(sys.argv) < 2:
        print("用法: python analyze_dangling_OH_orientation.py dump_file.lammpstrj")
        sys.exit(1)

    filename = sys.argv[1]
    print(f"读取文件: {filename}")

    first_frame = True
    n_frames = 0

    # 时间平均累加器
    sum_count_all = None
    sum_theta_all = None
    sum_cos_all = None

    sum_count_dang = None
    sum_theta_dang = None
    sum_cos_dang = None

    # 全局统计（盒子整体）
    global_theta_all_sum = 0.0
    global_cos_all_sum = 0.0
    global_H_all = 0

    global_theta_dang_sum = 0.0
    global_cos_dang_sum = 0.0
    global_H_dang = 0

    for timestep, box, ids, types, coords in iter_lammps_frames(filename):
        n_frames += 1
        if MAX_FRAMES is not None and MAX_FRAMES > 0 and n_frames > MAX_FRAMES:
            break

        box_lengths = box[:, 1] - box[:, 0]

        print(f"帧 {n_frames}, timestep = {timestep}")

        O_indices, water_H_indices = build_water_molecules(types, coords, box_lengths)
        print(f"  水分子数: {len(O_indices)}")

        # 计算每条 O–H 的取向 + 是否悬挂
        O_z_for_H, theta_deg, cos_theta, is_dangling = compute_OH_orientation_and_dangling(
            O_indices, water_H_indices, coords, box_lengths,
            ROO_cut=ROO_CUT, angle_cut=ANGLE_CUT
        )

        n_H = len(theta_deg)
        n_dang = is_dangling.sum()
        print(f"  本帧 H 总数: {n_H}, 悬挂 H 数: {n_dang} ({n_dang / max(n_H,1):.3f})")

        # 全盒子平均
        global_theta_all_sum += theta_deg.sum()
        global_cos_all_sum += cos_theta.sum()
        global_H_all += n_H

        if n_dang > 0:
            global_theta_dang_sum += theta_deg[is_dangling].sum()
            global_cos_dang_sum += cos_theta[is_dangling].sum()
            global_H_dang += n_dang

        # 按 z 分层
        (z_centers,
         count_all, sum_theta_all_frame, sum_cos_all_frame,
         count_dang, sum_theta_dang_frame, sum_cos_dang_frame) = analyze_z_layers_OH(
            O_z_for_H, theta_deg, cos_theta, is_dangling, box, dz=DZ
        )

        if first_frame:
            Z = z_centers
            sum_count_all = count_all
            sum_theta_all = sum_theta_all_frame
            sum_cos_all = sum_cos_all_frame

            sum_count_dang = count_dang
            sum_theta_dang = sum_theta_dang_frame
            sum_cos_dang = sum_cos_dang_frame

            first_frame = False
        else:
            if len(z_centers) != len(Z) or not np.allclose(z_centers, Z, atol=1e-6):
                raise ValueError("不同帧的 z 分层不一致，请检查 DZ 或盒子变化。")

            sum_count_all += count_all
            sum_theta_all += sum_theta_all_frame
            sum_cos_all += sum_cos_all_frame

            sum_count_dang += count_dang
            sum_theta_dang += sum_theta_dang_frame
            sum_cos_dang += sum_cos_dang_frame

    if n_frames == 0 or global_H_all == 0:
        print("没有有效帧，退出。")
        sys.exit(1)

    print("====================================================")
    print(f"总帧数: {n_frames}")
    if MAX_FRAMES is not None and MAX_FRAMES > 0:
        print(f"(实际分析帧数不超过 MAX_FRAMES = {MAX_FRAMES})")
    print(f"整体 <theta_OH,z> (所有 H, deg): {global_theta_all_sum / global_H_all:.3f}")
    print(f"整体 <cos theta_OH,z> (所有 H):  {global_cos_all_sum / global_H_all:.4f}")
    if global_H_dang > 0:
        print(f"整体 <theta_OH,z> (dangling H, deg): {global_theta_dang_sum / global_H_dang:.3f}")
        print(f"整体 <cos theta_OH,z> (dangling H):  {global_cos_dang_sum / global_H_dang:.4f}")
        print(f"整体 dangling H 比例: {global_H_dang / global_H_all:.4f}")
    else:
        print("整体没有检测到 dangling H？（可能是判据过宽或体系特殊）")
    print("====================================================")

    # 时间平均 z 分布
    avg_theta_all = np.zeros_like(Z)
    avg_cos_all = np.zeros_like(Z)
    avg_theta_dang = np.zeros_like(Z)
    avg_cos_dang = np.zeros_like(Z)
    frac_dang = np.zeros_like(Z)

    mask_all = sum_count_all > 0
    avg_theta_all[mask_all] = sum_theta_all[mask_all] / sum_count_all[mask_all]
    avg_cos_all[mask_all] = sum_cos_all[mask_all] / sum_count_all[mask_all]

    mask_dang = sum_count_dang > 0
    avg_theta_dang[mask_dang] = sum_theta_dang[mask_dang] / sum_count_dang[mask_dang]
    avg_cos_dang[mask_dang] = sum_cos_dang[mask_dang] / sum_count_dang[mask_dang]

    frac_mask = sum_count_all > 0
    frac_dang[frac_mask] = sum_count_dang[frac_mask] / sum_count_all[frac_mask]

    # 写数据文件
    out_data = np.vstack([
        Z,
        sum_count_all,
        avg_theta_all,
        avg_cos_all,
        sum_count_dang,
        avg_theta_dang,
        avg_cos_dang,
        frac_dang
    ]).T

    header = (
        "# z_center[A]\t"
        "H_all_count_sum\tavg_theta_all_deg\tavg_cos_all\t"
        "H_dangling_count_sum\tavg_theta_dangling_deg\tavg_cos_dangling\tfraction_dangling"
    )

    np.savetxt("dangling_OH_orientation_z_timeavg.dat", out_data,
               header=header, comments="",
               fmt=["%10.3f", "%12.1f", "%12.4f", "%12.6f",
                    "%12.1f", "%12.4f", "%12.6f", "%12.6f"])
    print("结果已写入: dangling_OH_orientation_z_timeavg.dat")

    # 画角度图（所有 H vs dangling H）
    plt.figure()
    plt.plot(Z[mask_all], avg_theta_all[mask_all], "o-", label="all H")
    plt.plot(Z[mask_dang], avg_theta_dang[mask_dang], "s-", label="dangling H")
    plt.xlabel("z (Å)")
    plt.ylabel("<theta(OH, z)> (deg)")
    plt.title("O–H orientation vs z (all vs dangling)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("dangling_OH_theta_z.png", dpi=300)
    print("图像已保存为: dangling_OH_theta_z.png")

    # 画 cos(theta) + 悬挂机率
    fig, ax1 = plt.subplots()
    ax1.plot(Z[mask_all], avg_cos_all[mask_all], "o-", label="<cos theta> all H")
    ax1.plot(Z[mask_dang], avg_cos_dang[mask_dang], "s-", label="<cos theta> dangling H")
    ax1.set_xlabel("z (Å)")
    ax1.set_ylabel("<cos theta(OH, z)>")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(Z[frac_mask], frac_dang[frac_mask], "k--", label="fraction dangling")
    ax2.set_ylabel("fraction dangling H")
    ax2.legend(loc="upper right")

    plt.title("O–H orientation and dangling H fraction vs z")
    fig.tight_layout()
    plt.savefig("dangling_OH_costheta_z.png", dpi=300)
    print("图像已保存为: dangling_OH_costheta_z.png")


if __name__ == "__main__":
    main()
