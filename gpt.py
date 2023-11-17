import torch
from ultralytics import YOLO
import cv2
from openai import OpenAI
import json

# open ai 제 개인 키 입니다.
client = OpenAI(
    api_key="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",
)


# gpt api로 키워드 전달하고 피드백 문장 생성받는 함수
def gpt_result(json_string):
    python_dict = json.loads(json_string)

    question = ("나는 사용자가 업로드한 발표 영상을 분석해서 1초당 말하는 단어 개수와 불필요한 단어 사용 총 횟수등을 분석하였어 "
                "이 결과들을 이용하여 사용자에게 더 나은 발표를 위한 피드백을 제공하고싶은데 이 제공되는 키워드를 이용하여 피드백을 만들어줘")
    question += str(list(python_dict.items()))

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
        max_tokens=256,
        model="gpt-3.5-turbo",
    )
    print(chat_completion.choices[0].message)


# cv 모델 구동시켜서 결과값 json 형태로 받는 함수
def save_results_to_json(results, file_name='all_results.json'):
    """
    결과 리스트를 받아서 각 결과를 JSON 포맷으로 변환 후 파일로 저장하는 함수

    Parameters:
    - results: JSON으로 변환할 결과 객체들의 리스트
    - file_name: 결과를 저장할 JSON 파일의 이름, 기본값은 'all_results.json'
    """
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
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        print(f"결과가 '{file_name}' 파일로 저장되었습니다.")
    except IOError as e:
        print(f"파일 저장 중 오류 발생: {e}")


# cv 모델 구동시키는 함수
def load_model():
    model = YOLO("yolo8n_pose.pt")
    results = model.predict("yun_action_test.mp4")
    save_results_to_json(results)


if __name__ == '__main__':
    # model = load_model()

    # 모델 돌려서 나온 결과값 예시,, gpt_result 함수에 전달
    json_string = '{"1초당 말하는 단어 개수": 2.3, "불필요한 단어 사용 총 횟수": 30}'
    gpt_result(json_string)