import cv2
import dlib
import numpy as np

# 모자 이미지 불러오기
hat_img = cv2.imread('hat.png', -1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def add_hat(frame, hat_img, face_landmarks):
    # 모자의 크기를 얼굴 크기에 맞게 조정
    hat_width = face_landmarks.part(16).x - face_landmarks.part(0).x
    hat_height = int(hat_width * hat_img.shape[0] / hat_img.shape[1])
    hat_img_resized = cv2.resize(hat_img, (hat_width, hat_height))

    # 모자를 얼굴 위치에 맞게 배치
    y1, y2 = face_landmarks.part(19).y - hat_height, face_landmarks.part(19).y
    x1, x2 = face_landmarks.part(0).x, face_landmarks.part(16).x

    alpha_hat = hat_img_resized[:, :, 3] / 255.0
    alpha_frame = 1.0 - alpha_hat

    for c in range(0, 3):
        frame[y1:y2, x1:x2, c] = (alpha_hat * hat_img_resized[:, :, c] +
                                  alpha_frame * frame[y1:y2, x1:x2, c])
    return frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        frame = add_hat(frame, hat_img, landmarks)

    cv2.imshow("AR Face Hat", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
