import cv2

# 打开摄像头
cap = cv2.VideoCapture(0)

# 加载正脸与侧脸级联
frontal_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测正脸
    faces = frontal_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    # 检测侧脸（左右侧皆可）
    profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

    # 合并结果（示例：直接拼接，生产可加去重）
    detections = list(profiles) + list(faces)
    
    # 绘制框
    for (x, y, w, h) in detections:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
        # print(x, y, w, h)
        
        detections 

    cv2.imshow('Head Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
