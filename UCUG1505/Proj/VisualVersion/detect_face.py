import cv2

# 打开摄像头
cap = cv2.VideoCapture(0)

# 加载正脸与侧脸级联
frontal_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

def detect() -> tuple[int, int] | None :
    ret, frame = cap.read()
    if not ret:
        return None
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测正脸
    faces = frontal_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    # 检测侧脸（左右侧皆可）
    profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

    # 合并结果（示例：直接拼接，生产可加去重）
    lrgSz = 0
    res = None
    
    detections = list(faces) + list(profiles)
    # 绘制框
    for (x, y, w, h) in detections:
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
        if (w < 80 or h < 80) : continue
        if lrgSz < w * h :
            lrgSz = w * h
            res = (x + w / 2, y + h / 2)

    # cv2.imshow('Head Detection', frame)
    res = (res[0] / frame.shape[1], res[1] / frame.shape[0]) if res else None
    # print(res)
    return res

if __name__ == "__main__" :
    while True :
        detect()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()