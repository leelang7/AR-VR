import cv2
import numpy as np

# 마커 이미지를 불러옵니다.
marker_image = cv2.imread('marker.png', 0)
orb = cv2.ORB_create()

# 마커 이미지의 키포인트와 디스크립터를 찾습니다.
kp1, des1 = orb.detectAndCompute(marker_image, None)

# 웹캠 피드 설정
cap = cv2.VideoCapture(0)

def draw_cube(frame, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    frame = cv2.drawContours(frame, [imgpts[:4]], -1, (0, 255, 0), 3)
    for i, j in zip(range(4), range(4, 8)):
        frame = cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)
    frame = cv2.drawContours(frame, [imgpts[4:]], -1, (0, 0, 255), 3)
    return frame

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp2, des2 = orb.detectAndCompute(gray_frame, None)

    if des1 is not None and des2 is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        if des1.dtype == np.float32:
            des1 = des1.astype(np.uint8)
        if des2.dtype == np.float32:
            des2 = des2.astype(np.uint8)
        
        matches = bf.match(des1, des2)

        if len(matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = marker_image.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

            # 3D 객체 매핑을 위한 설정
            K = np.array([[800, 0, w / 2], [0, 800, h / 2], [0, 0, 1]], dtype=np.float32)
            D = np.zeros((4, 1), dtype=np.float32)

            # 추적 포인트 설정
            obj_points = np.array([[0, 0, 0], [0, h-1, 0], [w-1, h-1, 0], [w-1, 0, 0]], dtype=np.float32)
            img_points = np.array(dst, dtype=np.float32)

            if img_points.shape[0] >= 4:
                _, rvecs, tvecs = cv2.solvePnP(obj_points, img_points, K, D)

                # 3D 객체의 점 좌표
                axis = np.float32([[0, 0, 0], [0, h - 1, 0], [w - 1, h - 1, 0], [w - 1, 0, 0],
                                [0, 0, -h + 1], [0, h - 1, -h + 1], [w - 1, h - 1, -h + 1], [w - 1, 0, -h + 1]])

                imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, K, D)
                frame = draw_cube(frame, dst, imgpts)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
