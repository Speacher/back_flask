# 1.팔짱 끼는 자세를 탐지하는 함수
def is_crossing_arms(person):
    left_wrist = person['keypoints']['left_wrist']
    right_wrist = person['keypoints']['right_wrist']
    left_elbow = person['keypoints']['left_elbow']
    right_elbow = person['keypoints']['right_elbow']

    # 팔꿈치와 손목의 좌표 차이 계산
    distance_left = ((left_wrist['x'] - right_elbow['x']) ** 2 + (left_wrist['y'] - right_elbow['y']) ** 2) ** 0.5
    distance_right = ((right_wrist['x'] - left_elbow['x']) ** 2 + (right_wrist['y'] - left_elbow['y']) ** 2) ** 0.5

    # 팔짱 끼는 자세 판별 (임계값은 실험적으로 조정해야 함)
    threshold = 100  # 예시 임계값
    return distance_left < threshold or distance_right < threshold


# 2. 주머니에 손을 넣거나 손이 엉덩이 & 허벅지 근처에 너무 붙어 있는 자세
def is_hand_in_pocket(person):
    left_wrist = person['keypoints']['left_wrist']
    right_wrist = person['keypoints']['right_wrist']
    left_hip = person['keypoints']['left_hip']
    right_hip = person['keypoints']['right_hip']

    # 손목과 엉덩이 사이의 거리 계산
    distance_left = ((left_wrist['x'] - left_hip['x']) ** 2 + (left_wrist['y'] - left_hip['y']) ** 2) ** 0.5
    distance_right = ((right_wrist['x'] - right_hip['x']) ** 2 + (right_wrist['y'] - right_hip['y']) ** 2) ** 0.5

    # 주머니에 손을 넣는 자세 판별 (임계값은 실험적으로 조정해야 함)
    threshold = 10  # 예시 임계값
    return distance_left < threshold or distance_right < threshold


# 3. 걷는 동작 횟수 (단순화)
def count_walking_actions(data, threshold=50):
    """
    걷는 동작의 횟수를 세는 함수.

    :param data: 분석할 프레임 데이터. 각 프레임은 사람 객체의 리스트를 포함한다.
    :param threshold: 걷는 동작을 감지하기 위한 y좌표의 임계값.
    :return: 걷는 동작의 총 횟수.
    """
    walking_count = 0

    for frame_list in data:
        for person in frame_list:
            keypoints = person["keypoints"]
            left_ankle = keypoints["left_ankle"]
            right_ankle = keypoints["right_ankle"]

            # 왼쪽 발목과 오른쪽 발목의 y좌표를 비교
            if abs(left_ankle["y"] - right_ankle["y"]) > threshold:
                walking_count += 1
                break  # 한 프레임 내에서 한 사람만 걷는 것으로 간주

    return walking_count


# 4. 손이 얼굴에 가까워지는 행동의 횟수를 세는 함수.
def count_hand_to_face_actions(data, threshold=50):
    """
    손이 얼굴에 가까워지는 행동의 횟수를 세는 함수.

    :param data: 분석할 프레임 데이터. 각 프레임은 사람 객체의 리스트를 포함한다.
    :param threshold: 손이 얼굴에 가까워지는 것을 감지하기 위한 거리 임계값.
    :return: 손이 얼굴에 가까워지는 행동의 총 횟수.
    """
    hand_to_face_count = 0

    for frame_list in data:
        for person in frame_list:
            keypoints = person["keypoints"]
            left_wrist = keypoints["left_wrist"]
            right_wrist = keypoints["right_wrist"]
            nose = keypoints["nose"]
            left_eye = keypoints["left_eye"]
            right_eye = keypoints["right_eye"]

            # 손목과 얼굴 키포인트 사이의 거리 계산
            distances = [
                (left_wrist["x"] - nose["x"]) ** 2 + (left_wrist["y"] - nose["y"]) ** 2,
                (right_wrist["x"] - nose["x"]) ** 2 + (right_wrist["y"] - nose["y"]) ** 2,
                (left_wrist["x"] - left_eye["x"]) ** 2 + (left_wrist["y"] - left_eye["y"]) ** 2,
                (right_wrist["x"] - right_eye["x"]) ** 2 + (right_wrist["y"] - right_eye["y"]) ** 2
            ]

            # 임계값보다 거리가 가까우면 카운트
            if any(distance < threshold ** 2 for distance in distances):
                hand_to_face_count += 1
                break  # 한 프레임 내에서 한 사람만 해당 행동을 하는 것으로 간주

    return hand_to_face_count


# 5. 뒷짐을 지는 자세의 횟수를 세는 함수.
def count_hands_behind_back_actions(data, threshold=50):
    """
    뒷짐을 지는 자세의 횟수를 세는 함수.

    :param data: 분석할 프레임 데이터. 각 프레임은 사람 객체의 리스트를 포함한다.
    :param threshold: 손이 등 뒤에 있는 것을 감지하기 위한 거리 임계값.
    :return: 뒷짐을 지는 자세의 총 횟수.
    """
    hands_behind_back_count = 0

    for frame_list in data:
        for person in frame_list:
            keypoints = person["keypoints"]
            left_wrist = keypoints["left_wrist"]
            right_wrist = keypoints["right_wrist"]
            left_shoulder = keypoints["left_shoulder"]
            right_shoulder = keypoints["right_shoulder"]
            left_hip = keypoints["left_hip"]
            right_hip = keypoints["right_hip"]

            # 손목이 등 뒤에 있는지 확인
            if (left_wrist["y"] > left_shoulder["y"] and left_wrist["y"] > left_hip["y"]) or \
                    (right_wrist["y"] > right_shoulder["y"] and right_wrist["y"] > right_hip["y"]):
                # 손목끼리의 거리가 임계값보다 작은지 확인
                distance_between_wrists = ((left_wrist["x"] - right_wrist["x"]) ** 2 + (
                        left_wrist["y"] - right_wrist["y"]) ** 2) ** 0.5
                if distance_between_wrists < threshold:
                    hands_behind_back_count += 1
                    break  # 한 프레임 내에서 한 사람만 해당 행동을 하는 것으로 간주

    return hands_behind_back_count
