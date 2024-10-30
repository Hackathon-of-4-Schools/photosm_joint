import cv2
import numpy as np
import mediapipe as mp
import torch
import torchvision.transforms.functional as F
import math
from PIL import Image
import sys

LIMBSEQ = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
        [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
        [1, 16], [16, 18], [3, 17], [6, 18]]
COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
        [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
        [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

IMAGE_SIZE = 256

np.set_printoptions(threshold=sys.maxsize)

def get_label_tensor(keypoint):
    canvas = np.zeros((IMAGE_SIZE,IMAGE_SIZE, 3)).astype(np.uint8)
    keypoint = trans_keypoins(keypoint)
    stickwidth = 4
    for i in range(18):
        x, y = keypoint[i, 0:2]
        if x == -1 or y == -1:
            continue
        cv2.circle(canvas, (int(x), int(y)), 4,COLORS[i], thickness=-1)
    joints = []
    for i in range(17):
        Y = keypoint[np.array(LIMBSEQ[i])-1, 0]
        X = keypoint[np.array(LIMBSEQ[i])-1, 1]            
        cur_canvas = canvas.copy()
        if -1 in Y or -1 in X:
            joints.append(np.zeros_like(cur_canvas[:, :, 0]))
            continue
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, COLORS[i])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

        joint = np.zeros_like(cur_canvas[:, :, 0])
        cv2.fillConvexPoly(joint, polygon, 255)
        joint = cv2.addWeighted(joint, 0.4, joint, 0.6, 0)
        joints.append(joint)
    pose = F.to_tensor(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))

    tensors_dist = 0
    e = 1
    for i in range(len(joints)):
        im_dist = cv2.distanceTransform(255-joints[i], cv2.DIST_L1, 3)
        im_dist = np.clip((im_dist / 3), 0, 255).astype(np.uint8)
        tensor_dist = F.to_tensor(Image.fromarray(im_dist))
        tensors_dist = tensor_dist if e == 1 else torch.cat([tensors_dist, tensor_dist])
        e += 1

    label_tensor = torch.cat((pose, tensors_dist), dim=0)
    if int(keypoint[14, 0]) != -1 and int(keypoint[15, 0]) != -1:
        y0, x0 = keypoint[14, 0:2]
        y1, x1 = keypoint[15, 0:2]
        face_center = torch.tensor([y0, x0, y1, x1]).float()
    else:
        face_center = torch.tensor([-1, -1, -1, -1]).float()
    return label_tensor, face_center   


def trans_keypoins(keypoints):
    # find missing index
    missing_keypoint_index = keypoints == -1

    # # crop the white line in the original dataset
    # keypoints[:,0] = (keypoints[:,0]-40)

    # # resize the dataset
    # scale_w = 1.0/176.0 * IMAGE_SIZE
    # scale_h = 1.0/256.0 * IMAGE_SIZE

    # keypoints[:,0] = keypoints[:,0]*scale_w - IMAGE_SIZE
    # keypoints[:,1] = keypoints[:,1]*scale_h - IMAGE_SIZE
    keypoints[missing_keypoint_index] = -1
    return keypoints

# Mediapipe 솔루션 초기화
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# 파일 경로 설정
input_image_path = "laying_man.jpg"  # 사용자가 업로드한 이미지 파일 경로
output_image_path = "output_image.jpg"  # 처리된 이미지를 저장할 파일 경로
# 이미지 불러오기
image = cv2.imread(input_image_path)
if image is None:
    print("이미지를 찾을 수 없습니다. 파일 경로를 확인해 주세요.")
else:
    # 이미지와 같은 크기의 검은색 배경 이미지 생성
    black_image = np.ones_like(image) * 255
    # BGR 이미지를 RGB로 변환
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Pose와 Hands 모델 초기화
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        # 포즈 관절 추출
        pose_results = pose.process(rgb_image)

        open_pose_order_list = [
        ]

        def convert_landmark(landmark):
            return np.array([
                landmark.x,
                landmark.y
            ]) * IMAGE_SIZE
        
        def get_neck():
            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            # 어깨 중심 (목 위치) 계산
            neck_x = (left_shoulder.x + right_shoulder.x) / 2
            neck_y = (left_shoulder.y + right_shoulder.y) / 2
            neck_point = [neck_x, neck_y]
            return np.array(neck_point) * IMAGE_SIZE


        open_pose_order_list.append(convert_landmark(pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]))
        open_pose_order_list.append(get_neck()) # NECK이라서 중앙값 구해야함
        open_pose_order_list.append(convert_landmark(pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]))
        open_pose_order_list.append(convert_landmark(pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]))
        open_pose_order_list.append(convert_landmark(pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]))
        open_pose_order_list.append(convert_landmark(pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]))
        open_pose_order_list.append(convert_landmark(pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]))
        open_pose_order_list.append(convert_landmark(pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]))
        open_pose_order_list.append(convert_landmark(pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]))
        open_pose_order_list.append(convert_landmark(pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]))
        open_pose_order_list.append(convert_landmark(pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]))
        open_pose_order_list.append(convert_landmark(pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]))
        open_pose_order_list.append(convert_landmark(pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]))
        open_pose_order_list.append(convert_landmark(pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]))
        open_pose_order_list.append(convert_landmark(pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]))
        open_pose_order_list.append(convert_landmark(pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]))
        open_pose_order_list.append(convert_landmark(pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]))
        open_pose_order_list.append(convert_landmark(pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]))
        open_pose_order_list = np.array(open_pose_order_list)
        np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)
        label_tensor, face_center = get_label_tensor(open_pose_order_list)
        # label_tensor = np.reshape(label_tensor, [IMAGE_SIZE, IMAGE_SIZE, label_tensor.shape[0]])
        label_tensor = label_tensor.T
        print(label_tensor.dtype)
        np.save("joint_data", label_tensor)
        # print(np.reshape(label_tensor, newshape=[IMAGE_SIZE, IMAGE_SIZE, label_tensor.shape[0]]))
        
        # print(open_pose_order_list)

        # 포즈 관절 그리기
        # if pose_results.pose_landmarks:
        #     # 랜드마크를 black_image에 그림
        #     mp_drawing.draw_landmarks(
        #         black_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #         mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),  # 관절 연결 선 (파란색)
        #         mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=3)  # 관절 점 (흰색)
        #     )
        #     # 추가 목 부분 및 얼굴 연결선 그리기
        #     left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        #     right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        #     nose = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        #     # 어깨 중심 (목 위치) 계산
        #     neck_x = int((left_shoulder.x + right_shoulder.x) / 2 * image.shape[1])
        #     neck_y = int((left_shoulder.y + right_shoulder.y) / 2 * image.shape[0])
        #     neck_point = (neck_x, neck_y)
        #     # 코의 좌표
        #     nose_point = (int(nose.x * image.shape[1]), int(nose.y * image.shape[0]))
        #     cv2.line(black_image, neck_point, nose_point, (0, 0, 0), 2)
        #     cv2.circle(black_image, neck_point, 3, (0, 0, 255), -1)

        #     cv2.putText(black_image, "")

        # # 처리된 이미지 저장
        # cv2.imwrite(output_image_path, black_image)
        # print(f"이미지 처리 완료: {output_image_path}")

        # import numpy as np
        # from PIL import Image

        # img = Image.open(output_image_path)
        # img.show()

        # x = np.array(img)
        # print(x.shape)

        # x_2 = np.asarray(img)
        # print(x_2.shape)