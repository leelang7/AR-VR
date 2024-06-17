import cv2
import cv2.aruco as aruco

# ArUco 사전 선택 (DICT_6X6_250은 6x6 크기의 250개의 마커를 포함)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# ID가 0인 마커 생성
marker_image = aruco.generateImageMarker(aruco_dict, 0, 200)

# 이미지 저장
cv2.imwrite("marker2.png", marker_image)
