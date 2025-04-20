import cv2
import mediapipe as mp
import numpy as np

def detect() -> tuple[list[tuple[float, float]], list[int]] :
	global face_cascade, camera, mpHands, mpDraw, faceRecognizer
	if "face_cascade" not in globals() :
		face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		camera = cv2.VideoCapture(0)
		mpHands = mp.solutions.hands.Hands()
		mpDraw = mp.solutions.drawing_utils
		faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
	ret, frame = camera.read()
	if not ret :
		print("Failed to grab frame")
		return []

	imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# detect hands
	results = mpHands.process(imgRGB)

	handsRes, faceRes = [], []
	if results.multi_hand_landmarks :
		for handLms in results.multi_hand_landmarks :
			for id, lm in enumerate(handLms.landmark) :
				h, w, c = imgRGB.shape
				cx, cy = int(lm.x * w), int(lm.y * h)
				cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
			mpDraw.draw_landmarks(frame, handLms, mp.solutions.hands.HAND_CONNECTIONS)
			# get the geogemtry center of the marks
			t = np.array([[lm.x, lm.y] for lm in handLms.landmark]).mean(axis=0)
			handsRes.append((t[0], t[1]))
	# cv2.imshow("Hands", frame)
	return (handsRes, faceRes)

if __name__ == "__main__" :
	while True :
		(hands, faces) = detect()
		print(hands)
		if cv2.waitKey(1) & 0xff == ord('q') :
			break
		
	camera.release()
	cv2.destroyAllWindows()