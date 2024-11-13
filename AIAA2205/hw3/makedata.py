import torch
import os
import librosa
# make mfcc data
def make_mfcc(mp4_path : str, n_mfcc : int = 39, sr : int = 16000, duration = None) :
	audio, sr = librosa.load(mp4_path, sr=sr, duration=duration)
	# 计算MFCC特征
	mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
	# 转换为 PyTorch tensor
	mfcc_tensor = torch.tensor(mfcc).float()

	return mfcc_tensor

if __name__ == "__main__" :
	files = os.listdir("data/hw3_videos")
	