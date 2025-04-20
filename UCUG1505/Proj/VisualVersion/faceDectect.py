import cv2

def detectCamera() -> list[(int, int, int, int)] :
	global face_cascade, cap
	if "face_cascade" not in globals() :
		face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		cap = cv2.VideoCapture(0)
		print(type(cap), type(face_cascade))
	return []

if __name__ == "__main__" :
	detectCamera()