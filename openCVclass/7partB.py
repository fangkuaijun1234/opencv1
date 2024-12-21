import cv2

# 读取图片
img = cv2.imread('mytou.jpg')

# 转换为灰阶图片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 人脸检测器
face_cascade = cv2.CascadeClassifier(r'E:\OpenCV\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

# 检测人脸
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

# 检查是否检测到人脸
if len(faces) == 0:
    print("没有检测到人脸")
else:
    # 绘制人脸区域
    for (x, y, w, h) in faces:
        print(f"人脸检测位置: x={x}, y={y}, w={w}, h={h}")
        # 在人脸区域添加矩形框
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 提取人脸区域的灰度图和原图
        face_gray = gray[y:y + h, x:x + w]
        face_area = img[y:y + h, x:x + w]

        # 眼睛检测
        path_of_haarcascade_eye = r"E:\OpenCV\opencv\sources\data\haarcascades\haarcascade_eye.xml"
        eye_cascade = cv2.CascadeClassifier(path_of_haarcascade_eye)

        # 检测眼睛
        eyes = eye_cascade.detectMultiScale(face_gray)
        if len(eyes) == 0:
            print("没有检测到眼睛")
        else:
            for (ex, ey, ew, eh) in eyes:
                print(f"眼睛检测位置: ex={ex}, ey={ey}, ew={ew}, eh={eh}")
                # 在眼睛区域添加矩形框
                cv2.rectangle(face_area, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

# 保存处理后的图片
save_image_path = "my-drawing.jpg"
cv2.imwrite(save_image_path, img)

# 显示图片（可选）
cv2.imshow("Detected Faces and Eyes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
