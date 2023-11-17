from openai import OpenAI
import json


# gpt api로 키워드 전달하고 피드백 문장 생성받는 함수
def get_gpt_result(cv_json_result, nlp_json_result):
    print("get_gpt_result 함수 실행")

    # open ai private key
    client = OpenAI(
        api_key="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",
    )

    # cv_dict_result = json.loads(cv_json_result)
    # nlp_dict_result = json.loads(nlp_json_result)

    question = ("나는 이번에 발표자를 위한 발표 ai 피드백 서비스 프로젝트에 참여하고 있어" +
                "우리 프로젝트의 주요 내용은 사용자의 발표 동영상을 받아서 ai모델로 발표 영상을 분석한 뒤 피드백을 주는거야" +
                "발표를 분석하여 얻은 결과값은 다음과 같아\n" +
                "Script(발표 대본)" + "Speed(말하기 속도, 초당 말한 글자 수)" + "Time(발표 영상 길이)" + "Filler word(불필요한 단어들 및 해당 단어를 말한 횟수)" +
                "crossing_arms_count(팔짱낀 횟수)" + "hands_in_pockets_count(주머니에 손넣은 횟수)" + "walking_actions(걷는 동작 횟수)" +
                "count_hand_to_face_actions(얼굴 만진 횟수)" + "hands_behind_back_actions(뒷짐진 횟수)" +
                "위의 키워드에 해당하는 내용을 제공해줄게\n")

    question += json.dumps(nlp_json_result, ensure_ascii=False, indent=2)
    question += json.dumps(cv_json_result, ensure_ascii=False, indent=2)
    # question += str(list(nlp_dict_result.items()))
    # question += str(list(cv_dict_result.items()))

    question += ("\n이제 위의 자료들을 기반으로 이 사용자에게 전달해줄 피드백 내용을 생성해줘\n" +
                 "(단, 불필요한 서론은 모두 제외하고, 사용자에게 직접적으로 전달할 피드백 내용만 작성해서 줘!!!)")
    question += "\n'너'가 피드백내용을 줄때 주의사항을 알려줄게.\n"
    question += ("'발표 동영상 분석 결과를 바탕으로 사용자에게 전달할 피드백 내용은 다음과 같습니다' 이런 시작하는 말을 빼고," +
                 "'말하기 속도(Speed): 발표 속도가 적절했습니다. 초당 약 5.15개의 글자를 말하셨습니다'" +
                 "이런 단순 정보전달도 빼고, 어떻게 발표를 수정해야 더 나은 발표가 될지 피드백하는 내용만 작성해줘." +
                 "1024 토큰 내에 핵심 피드백만 작성할 것")

    print("model에 전달 함수 실행")
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
        max_tokens=1024,
        model="gpt-3.5-turbo",
    )
    print(chat_completion.choices[0].message.content)

    print(type(chat_completion.choices[0].message.content))
    return chat_completion.choices[0].message.content