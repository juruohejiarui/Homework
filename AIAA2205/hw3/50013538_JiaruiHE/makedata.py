import cv2
import numpy as np
import os
from tqdm import tqdm

def extractFrames(videoPath, numFrames=32) :
	cap = cv2.VideoCapture(videoPath)
	totFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frames = []
	for i in range(totFrames) :
		ret, frame = cap.read()
		if not ret :
			break
		frames.append(frame)
	cap.release()
	indices = np.linspace(0, totFrames - 1, numFrames, dtype=np.int32)
	padFrames = [frames[idx] for idx in indices]
	return padFrames

if __name__ == "__main__" :
	mp4Files = os.listdir("data/hw3_videos")
	for mp4File in tqdm(mp4Files) :
		os.mkdir(os.path.join("data/hw3_32fpv", mp4File))
		frames = extractFrames(os.path.join("data/hw3_videos", mp4File))
		for idx, frame in enumerate(frames) :
			path = os.path.join("data/hw3_32fpv", mp4File, f"frame_{idx:08d}.jpg")
			cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 75])