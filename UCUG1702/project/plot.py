import mdtraj as mda
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

data_path = './data/dump-surface.lammpstrj'

layer_num = 40

min_pos, max_pos = None, None

class H2O :
	def __init__ (self, pO : np.ndarray, pH1 : np.ndarray, pH2 : np.ndarray) :
		self.pO = pO
		self.pH1 = pH1
		self.pH2 = pH2

		def fix_pH(pH) :
			dis = abs(pH - pO)
			for i in range(3) :
				if dis[i] > (max_pos[i] - min_pos[i]) / 2 :
					if pH[i] > pO[i] :
						pH[i] -= (max_pos[i] - min_pos[i])
					else :
						pH[i] += (max_pos[i] - min_pos[i])

		if np.linalg.norm(pO - pH1) > 20 :
			fix_pH(pH1)
		if np.linalg.norm(pO - pH2) > 20 :
			fix_pH(pH2)

def toArr(h2o : list[H2O]) -> tuple[np.ndarray, np.ndarray, np.ndarray] :
	n = len(h2o)
	pO = np.zeros((n, 3))
	pH1 = np.zeros((n, 3))
	pH2 = np.zeros((n, 3))
	h2o = sorted(h2o, key=lambda x: x.pO[2])
	for i in range(n) :
		pO[i, :] = h2o[i].pO
		pH1[i, :] = h2o[i].pH1
		pH2[i, :] = h2o[i].pH2
	return pO, pH1, pH2

data_raw : list[str]

with open(data_path, 'r') as f:
	data_raw = f.readlines()

data_raw = [line.strip() for line in data_raw]

cur_frame : list[H2O] = None
frames : list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

line_id = 0

process = tqdm(total=len(data_raw), desc="Parsing LAMMPS trajectory")
while line_id < len(data_raw) :
	cur_line = data_raw[line_id].strip()
	if cur_line.startswith("ITEM: TIMESTEP") :
		if cur_frame is not None :
			frames.append(toArr(cur_frame))
		cur_frame = []
		if min_pos is None :
			min_pos = np.zeros(3)
			max_pos = np.zeros(3)
			for x in range(3) :
				a, b = data_raw[line_id + 5 + x].strip().split()
				min_pos[x], max_pos[x] = float(a), float(b)
		line_id += 9
		process.update(9)
		continue
	def get_pos(line : str, type : int) -> np.ndarray :
		parts = line.strip().split()
		# print(parts)
		assert type == int(parts[1]), f"Expected type {type}, got {parts[1]}"
		return np.array([float(parts[2]), float(parts[3]), float(parts[4])])
	pO = get_pos(data_raw[line_id], 1)
	pH1 = get_pos(data_raw[line_id + 1], 2)
	pH2 = get_pos(data_raw[line_id + 2], 2)
	cur_frame.append(H2O(pO, pH1, pH2))
	process.update(3)
	line_id += 3

process.close()
assert max([frame[0].shape[0] for frame in frames]) == min([frame[0].shape[0] for frame in frames]), \
	"Number of H2O molecules must be consistent across frames."

print(f"Total frames parsed: {len(frames)} with {frames[0][0].shape[0]} H2O molecules each.")

print("box bounds:", min_pos, max_pos)

# use binary search to find the upper bound index of z_cut in frame
def get_upper_bound(frame : tuple[np.ndarray, np.ndarray, np.ndarray], z_cut : float) -> int :
	l, r = 0, frame[0].shape[0]
	while l < r :
		mid = (l + r) // 2
		if frame[0][mid, 2] <= z_cut :
			l = mid + 1
		else :
			r = mid
	return l

def get_layer(layer_id : int) -> tuple[float, float] :
	layer_min_z = min_pos[2] + (max_pos[2] - min_pos[2]) / layer_num * layer_id
	layer_max_z = min_pos[2] + (max_pos[2] - min_pos[2]) / layer_num * (layer_id + 1)
	return layer_min_z, layer_max_z
def get_layer_molecules(frame : tuple[np.ndarray, np.ndarray, np.ndarray], layer_id : int, prev_idx : int) -> tuple[np.ndarray, np.ndarray, np.ndarray] :
	_, layer_max_z = get_layer(layer_id)
	end_idx = get_upper_bound(frame, layer_max_z)
	return frame[0][prev_idx:end_idx, :], frame[1][prev_idx:end_idx, :], frame[2][prev_idx:end_idx, :]

lst_idx = 0
layers_center = []
angles = []
cnt_hbs = []
prev_idx = [0 for _ in range(len(frames))]
for layer_id in range(layer_num) :
	layer_min_z, layer_max_z = get_layer(layer_id)
	layers_center.append(0.5 * (layer_min_z + layer_max_z))
	layers = []

	layer_angle = []

	layer_cnt_hb = []
	
	for frame_id, frame in enumerate(frames) :
		layers.append(get_layer_molecules(frame, layer_id, prev_idx[frame_id]))
		prev_idx[frame_id] = get_upper_bound(frame, layer_max_z)

	for layer in layers :
		if layer[0].shape[0] == 0 :
			layer_angle.append(np.nan)
			continue
		direct = 0.5 * (layer[2] + layer[1]) - layer[0]
		# get angle between direct and z axis
		angle = (direct[:, 2] / np.linalg.norm(direct, axis=1)).mean().item()
		layer_angle.append(angle)

	for i, layer in enumerate(layers) :
		cnt_hb = 0
		if layer[0].shape[0] == 0 :
			layer_cnt_hb.append(0)
			continue
		p1i = layer[1] - layer[0]
		p2i = layer[2] - layer[0]
		# get unit vectors
		u1i = p1i / np.linalg.norm(p1i, axis=1, keepdims=True)
		u2i = p2i / np.linalg.norm(p2i, axis=1, keepdims=True)
		for j in range(layer[0].shape[0]) :
			neighbors = []
			pj = frames[i][0] - layer[0][j]
			uj = pj / np.linalg.norm(pj, axis=1, keepdims=True)
			dis = np.linalg.norm(pj, axis=1)
			dot1 = np.clip(np.dot(uj, u1i[j]), -1, 1)
			dot2 = np.clip(np.dot(uj, u2i[j]), -1, 1)
			min_angle = np.min(np.vstack([np.arccos(dot1), np.arccos(dot2)]), axis=0)
			
			cnt_hb += ((min_angle < np.deg2rad(30)) & (dis < 3.5)).sum().item() / layer[0].shape[0]
		layer_cnt_hb.append(cnt_hb)

	angle_avg = np.array(layer_angle).mean()
	cnt_hb_avg = np.array(layer_cnt_hb).mean()
	tqdm.write(f"Layer {layer_id} molecules has {cnt_hb_avg} H-bonds.\n")
	angles.append(angle_avg)
	cnt_hbs.append(cnt_hb_avg)

layers_center = np.array(layers_center)
angles = np.array(angles)
cnt_hbs = np.array(cnt_hbs)
print(angles)
print(cnt_hbs)
# plot the angles and directions on two subplots
plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.plot(layers_center, angles, marker='o')
plt.xlabel('Layer Center Z (nm)')
plt.ylabel('Average Angle (rad)')
plt.title('Average Angle vs Layer Height')
plt.subplot(2, 1, 2)
plt.plot(layers_center, cnt_hbs, marker='o', color='orange')
plt.xlabel('Layer Center Z (nm)')
plt.ylabel('Average Number of H-Bonds')
plt.title('Average Number of H-Bonds vs Layer Height')
plt.tight_layout()
plt.savefig('./layer_properties.png')

