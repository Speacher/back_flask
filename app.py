from flask import Flask, request, jsonify
from ultralytics import YOLO
from flask_cors import CORS
import os
import json
import cv_functions as cf
import open_ai
import open_ai as oa


app = Flask(__name__)
CORS(app)



# [CV] cv 모델 구동시켜서 결과값 dirty json 받기
def save_results_to_dirty_json(results, dirty_json):
    all_results = []

    # 각 결과를 JSON으로 변환하고 리스트에 추가
    for r in results:
        # result_json 변수가 실제 JSON 문자열을 반환하는지 확인이 필요합니다.
        # 여기서는 r이 JSON 문자열을 직접 반환하는 객체라고 가정하고 있습니다.
        result_json = r.tojson()

        # JSON 문자열을 파이썬 딕셔너리로 로드
        try:
            data_json = json.loads(result_json)
            all_results.append(data_json)
        except json.JSONDecodeError as e:
            print(f"JSON 변환 중 오류 발생: {e}")

    # 리스트를 JSON 파일로 저장
    try:
        with open(dirty_json, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        print(f"결과가 '{dirty_json}' 파일로 저장되었습니다.")
    except IOError as e:
        print(f"파일 저장 중 오류 발생: {e}")


# [CV] cv 모델 구동시켜서 dirty json 파일 얻기
def load_cv_model(mp4_name, dirty_json_name):
    model = YOLO("yolo8n_pose.pt")
    results = model.predict(mp4_name)
    save_results_to_dirty_json(results, dirty_json_name)


# [CV] dirty json -> useful json
def make_data_set(src_json, dst_json):
    # COCO 데이터셋의 키포인트 이름
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    # JSON 파일 로드
    with open(src_json, 'r') as file:
        data = json.load(file)

    # 각 person 객체에 keypoint_name2id 추가
    for sublist in data:
        for person in sublist:
            person['keypoint_name2id'] = {name: index for index, name in enumerate(keypoint_names)}
            person['keypoints'] = {
                name: {
                    "x": person['keypoints']['x'][index],
                    "y": person['keypoints']['y'][index],
                    "visible": person['keypoints']['visible'][index]
                } for index, name in enumerate(keypoint_names)
            }

    # JSON 파일에 변경 사항 저장
    with open(dst_json, 'w') as file:
        json.dump(data, file, indent=4)


# [CV] 분석된 파일에서 유의미한 결과값 얻기
# gpt script 로도 return
def get_cv_results(useful_json_name):
    # JSON 파일 로드
    with open(useful_json_name, 'r') as file:
        data = json.load(file)

    # 1. 팔짱 끼는 자세를 한 사람 수 세기
    crossing_arms_count = sum(cf.is_crossing_arms(person[0]) for person in data)
    print(f"팔짱 끼는 자세 횟수: {crossing_arms_count}")

    # 2. 주머니에 손을 넣는 자세를 한 사람 수 세기
    hands_in_pockets_count = sum(cf.is_hand_in_pocket(person[0]) for person in data)
    print(f"주머니에 손을 넣거나 손이 엉덩이 & 허벅지 근처에 너무 붙어 있는 자세 횟수: {hands_in_pockets_count}")

    # 3. 걷는 동작 횟수
    walking_actions = cf.count_walking_actions(data)
    print("걷는 동작 횟수:", walking_actions)

    # 4. 손이 얼굴에 가까워지는 행동의 횟수를 세는 함수.
    hand_to_face_actions = cf.count_hand_to_face_actions(data)
    print("손이 얼굴에 가까워지는 행동 횟수:", hand_to_face_actions)

    # 5. 뒷짐을 지는 자세의 횟수를 세는 함수.
    hands_behind_back_actions = cf.count_hands_behind_back_actions(data)
    print("뒷짐을 지는 자세 횟수:", hands_behind_back_actions)

    ##################
    # val 명칭 수정 요망#
    ##################
    results_dict = {
        "crossing_arms_count": crossing_arms_count,
        "hands_in_pockets_count": hands_in_pockets_count,
        "walking_actions": walking_actions,
        "hand_to_face_actions": hand_to_face_actions,
        "hands_behind_back_actions": hands_behind_back_actions
    }
    json_result = json.dumps(results_dict, indent=2)

    # 모델 돌려서 나온 결과값 예시,, gpt_result 함수에 전달
    # nlp 모델 용으로 남겨둔 주석
    # oa.gpt_result(json_result)

    return json_result


def action(mp4_name):
    # 1. mp4 파일 받기
    # 2. cv model에 전달하여 분석 >> dirty json 파일 받기
    # 3. make_data_set 함수에 dirty json 파일 전달하여 useful json 파일 얻기
    # 4. get_cv_results 함수에 useful json 파일 전달하여 유의미한 결과값 얻기
    # 5. 유의미한 결과값 반환

    # 1,2
    dirty_json_name = 'all_results.json'
    model = load_cv_model(mp4_name, dirty_json_name)

    # 3
    useful_json_name = 'all_results_mod1.json'
    make_data_set(dirty_json_name, useful_json_name)

    # 4
    result = get_cv_results(useful_json_name)

    # 5
    return result


@app.route('/api/predict', methods=['POST'])
def predict():
    file = request.files['file']  # 요청에서 파일을 가져옴

    # 임시 파일을 생성하고 저장
    temp_path = "temp_file.mp4"
    file.save(temp_path)
    # 임시 파일을 모델에 입력
    result = action(temp_path)
    print(result)
    print(type(result))
    # 임시 파일 삭제
    os.remove(temp_path)
    # 결과를 JSON 형태로 반환
    return result




# NLP################### NLP##################
# NLP################### NLP##################
# NLP##################  NLP##################


import torch
from transformers import pipeline
import librosa
from collections import Counter


from transformers import pipeline
import librosa
from collections import Counter
def STT_test(audio_file_path, device='cpu'):
    pipe = pipeline(
        "automatic-speech-recognition",
        model='whisper',
        chunk_length_s=30,
        generate_kwargs={"language": "<|ko|>", "task": "transcribe"},
        device=device,
    )
    transcription = pipe(audio_file_path)\

    # mp3 파일 불러오기
    y, sr = librosa.load(audio_file_path)

    # 음성 활성화 감지 및 무음 부분 제거
    y_trimmed, index = librosa.effects.trim(y, top_db=20)

    # 끝에서부터 5초 전까지의 오디오 데이터 추출
    end_time = librosa.get_duration(y=y)  # 전체 오디오의 길이(초)
    start_time = end_time - 5  # 끝에서 5초 전의 시간 계산
    y_five_seconds = y_trimmed[int(start_time * sr):int(end_time * sr)]

    # 전체 오디오의 길이(초) 계산
    total_time_seconds = len(y_trimmed) / sr

    speaking_rate = len(transcription['text']) / total_time_seconds

    stt = transcription['text']

    with open('transcription.txt', 'w', encoding='utf-8') as file:
        file.write(stt)

    with open('transcription.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    words = text.split()

    # 단어 카운트
    word_count = Counter(words)

    # 예시 사용
    unnecessary_words = ["가지고", "거", "거든", "거든요" , "게" ,"그", "그거","그게", "그냥", "그래도", "그래서", "그러고 나서", "그러고 보니", "그러고는",
        "그러니까", "그러다보니", "그러면",  "그런 건", "그런 것", "그런데", "그럼", "그럼에도 불구하고",
        "그렇게",  "그렇지만" ,"그리고" ,"근데", "기로", "기에" ,"나", "니까" ,"다", "더니" ,"더라", "더라고요", "되게", "또" ,"막" "뭐", "뭐랄까",
        "뭐지", "뭔가", "별로", "서요" ,"아", "아마", "아마도" ,"아무래도" ,"아무튼" ,"아서" ,
        "약간" ,"좀" ,"어" ,"어느" ,"어디" ,"어디까지" ,"어디로" ,"어디서", "어디에" ,"어때","어떤" ,"어떻게" ,"어떻게 보면", "어떻게든", "어쩄든", "어쩌다가" ,"어쩌면" ,"어쩜" ,"어짜나" ,"에요" ,"예요",
        "왜냐하면", "요" ,"음" ,"이" ,"이거" ,"이게","이런", "이런 식으로" ,"이렇게" , "이상하게" ,"이제", "자" ,"저" ,"저거", "저게", "저렇게" ,"좀"  ,"지요" ,"진짜"
        ]

    # 필요하지 않은 단어의 빈도수 계산
    unnecessary_word_count = {word: word_count[word] for word in unnecessary_words}
    result = {
        "script" : stt,
        "time" : total_time_seconds,
        "speed" : speaking_rate,
        "filler_word" : unnecessary_word_count
        }
    # 파일명을 원하는 대로 지정합니다.
    file_name = "nlp_result.json"

    # JSON 파일로 저장합니다.
    with open(file_name, 'w') as json_file:
        json.dump(result, json_file)


    os.remove('transcription.txt')
    return result

@app.route('/api/predict2', methods=['POST'])
def predict2():
    file = request.files['file']  # 요청에서 파일을 가져옴

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 임시 파일을 생성하고 저장
    temp_path = "temp_file2.mp3"
    file.save(temp_path)


    result = STT_test(temp_path)

    # 임시 파일 삭제
    os.remove(temp_path)
    print(result)
    print('=================')
    # 결과를 JSON 형태로 반환
    json_result = json.dumps(result, indent=2)
    print(json_result)
    return json_result


@app.route('/api/gpt', methods=['POST'])
def predict3():
    data = request.get_json()
    cv_json_result = data.get('cv_json_result')
    nlp_json_result = data.get('nlp_json_result')

    gpt_result = open_ai.get_gpt_result(nlp_json_result, cv_json_result)

    return gpt_result;


if __name__ == '__main__':
    app.run(debug=True)




