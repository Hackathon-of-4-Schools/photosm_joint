import cv2
import numpy as np
import mediapipe as mp

# Mediapipe 솔루션 초기화
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 파일 경로 설정
input_image_path = "image.jpg"  # 사용자가 업로드한 이미지 파일 경로
output_image_path = "output_image.jpg"  # 처리된 이미지를 저장할 파일 경로

# 이미지 불러오기
image = cv2.imread(input_image_path)
if image is None:
    print("이미지를 찾을 수 없습니다. 파일 경로를 확인해 주세요.")
else:
    # 이미지와 같은 크기의 검은색 배경 이미지 생성
    black_image = np.zeros_like(image)

    # BGR 이미지를 RGB로 변환
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Pose와 Hands 모델 초기화
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose, \
         mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        
        # 포즈 관절 추출
        pose_results = pose.process(rgb_image)
        hand_results = hands.process(rgb_image)

        # 포즈 관절 그리기
        if pose_results.pose_landmarks:
            # 랜드마크를 black_image에 그림
            mp_drawing.draw_landmarks(
                black_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),  # 관절 연결 선 (파란색)
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3)  # 관절 점 (흰색)
            )

            # 추가 목 부분 및 얼굴 연결선 그리기
            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            nose = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

            # 어깨 중심 (목 위치) 계산
            neck_x = int((left_shoulder.x + right_shoulder.x) / 2 * image.shape[1])
            neck_y = int((left_shoulder.y + right_shoulder.y) / 2 * image.shape[0])
            neck_point = (neck_x, neck_y)

            # 코의 좌표
            nose_point = (int(nose.x * image.shape[1]), int(nose.y * image.shape[0]))

            cv2.line(black_image, neck_point, nose_point, (255, 255, 255), 2)

            cv2.circle(black_image, neck_point, 3, (0, 0, 255), -1)

        # 손가락 관절 그리기
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    black_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),  # 손가락 연결 선 (파란색)
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3)  # 손가락 관절 점 (흰색)
                )

        # 처리된 이미지 저장
        cv2.imwrite(output_image_path, black_image)
        print(f"이미지 처리 완료: {output_image_path}")
