import torch
import os
import torchaudio
import pickle
# make mfcc data
def make_mfcc(mp4_path : str, n_mfcc : int = 39, sr : int = 16000, duration = None) :
	waveform, sample_rate = torchaudio.load(mp4_path)
	

if __name__ == "__main__" :
	files = os.listdir("data/hw3_videos")
	for file in files :
		path = os.path.join("data/hw3_mfcc", file.split(".mp4")[0] + ".mfcc")
		f = open(path, "wb")
		tensor = make_mfcc(os.path.join("data/hw3_videos", file))
		pickle.dump(tensor, f, True)