from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import os

log_path = [
	"freeLB-10",
	"freeLB-40",
	"PGD-10",
	"PGD-40",
	"std"
]
log_label = [
	"FreeLB-10",
	"FreeLB-40",
	"PGD-10",
	"PGD-40",
	"Standard"
]

# also the name of figure file
items = [
	("test-0", "Original"),
	("test-10", "PGD-10 Attack"),
	("test-40", "PGD-40 Attack"),
	("train", "Training")
]

subitems_idx = {
	"acc" : 0,
	"loss" : 1
}

ylim = {
	"acc" : (0.7, 1.0),
	"loss" : (0, 0.058)
}

subitems_name = [
	"accuracy",
	"loss"
]

def getEventAccumulator(log_path : str) -> event_accumulator.EventAccumulator:
	files = os.listdir(os.path.join("run", log_path))
	if len(files) == 0:
		print(f"No files found in {log_path}")
		return None
	# get newest file
	ea = event_accumulator.EventAccumulator(os.path.join("run", log_path))
	ea.Reload()
	return ea

if __name__ == "__main__":
	# create 4 subplots for each item
	figs = [plt.subplots(nrows=1, ncols=2, figsize=(10, 6)) for _ in range(len(items))]
	
	for fig, ax in figs :
		for (tag, subitem_idx) in subitems_idx.items() :
			ax[subitem_idx].set_title(f"{subitems_name[subitem_idx]}")
			ax[subitem_idx].set_xlabel("Steps")
			ax[subitem_idx].set_ylabel(f"{tag.capitalize()}")
			ax[subitem_idx].set_ylim(ylim[tag])
			ax[subitem_idx].grid()

	for i, (path, label) in enumerate(zip(log_path, log_label)) :
		ea = getEventAccumulator(path)

		for subitem_tag, subitem_idx in subitems_idx.items() :
			for i, item in enumerate(items) :
				fig, ax = figs[i]
				dtList = ea.scalars.Items(f"{item[0]}/{subitem_tag}")
				x = np.array([dt.step for dt in dtList])
				y = np.array([dt.value for dt in dtList])

				ax[subitem_idx].plot(x, y, label=label)

	for i, (fig, ax) in enumerate(figs) :
		for subitem_idx in subitems_idx.values() :
			ax[subitem_idx].legend()
			ax[subitem_idx].grid()
		fig.tight_layout()
		fig.savefig(f"figures/plot-{items[i][1]}.png", dpi=300)

		