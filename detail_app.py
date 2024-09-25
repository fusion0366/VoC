import streamlit
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
# from kospellpy import spell_init
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import openai
import plotly.express as px
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from PIL import Image
import altair as alt

# OpenAI API 키 설정
openai.api_key = ''

# 폰트 설정
# fm.fontManager.addfont('./fonts/NanumBarunGothic.ttf')
# font_name = font_manager.FontProperties(fname='fonts\\NanumBarunGothic.ttf').get_name()
# rc('font', family=font_name)

# 한글 폰트 경로 설정
font_pt = './fonts/NanumBarunGothic.ttf'
font_manager.fontManager.addfont('./fonts/NanumBarunGothic.ttf')
font_name = font_manager.FontProperties(fname=font_pt).get_name()
rc('font', family=font_name)

# 데이터 타입 최적화 함수
def optimize_dtypes(dataframe):
    for column in dataframe.columns:
        col_type = dataframe[column].dtype

        # 결측치가 없는 경우에만 데이터 타입 변경
        if not dataframe[column].isnull().any():
            if col_type != object:
                c_min = dataframe[column].min()
                c_max = dataframe[column].max()

                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        dataframe[column] = dataframe[column].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        dataframe[column] = dataframe[column].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        dataframe[column] = dataframe[column].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        dataframe[column] = dataframe[column].astype(np.int64)

                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        dataframe[column] = dataframe[column].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        dataframe[column] = dataframe[column].astype(np.float32)
                    else:
                        dataframe[column] = dataframe[column].astype(np.float64)
            else:
                dataframe[column] = dataframe[column].astype('category')

    return dataframe


# 텍스트 전처리 함수
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^가-힣0-9a-zA-Z\s.]', '', text)  # 특수문자 제거
        text = text.replace('\n', '')  # 개행문자 제거
        return text
    else:
        return ""


# # 문장 분리 함수
# def split_sentences(df, column):
#     split_df = df[column].str.split('.').apply(lambda x: pd.Series(x)).stack().reset_index(level=1, drop=True).to_frame(column).reset_index(drop=True)
#     split_df = split_df[split_df[column].str.strip() != ''].reset_index(drop=True)
#     return split_df

# 메인 앱 콘텐츠
def detail_page():
    stTitle = st.columns([0.1, 3.0])
    stTitle[0].markdown('''
    <p class="mb-5" style="text-align: center;">
        <img alt="" loading="lazy" width="40" decoding="async" data-nimg="1" src="https://intense-ai.com/img/analyze.png" style="color: transparent;margin-top: 17px;">
    </p>
    ''', unsafe_allow_html=True)
    stTitle[1].header('분석 보고서 (상세)')

    # 선택된 옵션 표시하기

    stProject = st.columns([2.0, 1.5])
    project_option = stProject[0].selectbox('프로젝트', ['1. 국무조정실 에너지연구원', '2. 시니어 대상 조사', '3. 일반인 광고 조사', '4. 일반인 의견 조사',
                                                     '5. 2024 통신 서비스 관련 소비자 조사', '6. 2024 소비자 의견 조사', '7. 광고 관련 소비자 조사',
                                                     '8. 주류 관련 소비자 의견 조사'])
    round_option = stProject[1].selectbox('회차', ['1회차', '2회차', '3회차', '4회차', '5회차', '6회차', '7회차'])

    st.markdown('#### 상세 분석')
    stMulti1 = st.columns([1, 1])
    corNm = stMulti1[0].multiselect(
    "기관명",
    ['경제·인문사회연구회', '과학기술정책연구원', '국토연구원', '대외경제정책연구원', '산업연구원',
    '에너지경제연구원', '정보통신정책연구원', '통일연구원', '한국개발연구원', 'KDI국제정책대학원',
    '한국교육개발원', '한국청소년정책연구원', '한국해양수산개발원', '한국농촌경제연구원', '한국보건사회연구원',
    '육아정책연구소', '한국행정연구원', '한국환경연구원', '한국법제연구원', '한국여성정책연구원',
    '한국조세재정연구원', '한국직업능력연구원', '건축공간연구원', '한국교육과정평가원', '한국교통연구원',
    '한국노동연구원', '한국형사·법무정책연구원'])

    detailBiz = stMulti1[1].multiselect(
    "상세업무",
    ['경영지원 서비스', '연구지원 서비스', '교육 훈련', '정보 제공', '육성 및 촉진', '기본연구',
    '수탁연구', '위탁/공동연구', '학위(외국인 졸업생, ', '학위(내국인 졸업생, ', '학위(외국인 재학생, ',
    'KRIVET 메일 서비스', '커리어넷 홈페이지', '자격 시험 주관', '정보제공 서비스(KDIS, -내국인 졸업생',
    '정보제공 서비스(KDIS, -외국인 재학생'])

    stMulti2 = st.columns([1, 1])
    bizType = stMulti2[0].multiselect(
    "업무유형",
    ['운영지원', '교육훈련', '정보제공', '육성및촉진', '연구과제수행', '자격시험주관'])


    gendFlag = stMulti2[1].multiselect("성별", ['남성', '여성'])

    stMulti3 = st.columns([1, 1])
    ageDiv = stMulti3[0].multiselect("연령대", ['20대(19세 포함)', '30대', '40대', '50대', '60대 이상'])

    stMulti3[1].markdown("""
        <style>
        [data-testid="baseButton-primary"] {
            height: 70px;
            background-color: #0070C0;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    btnSearch = stMulti3[1].button('상세 분석', type="primary", use_container_width=True)

    subTab1, subTab2, subTab3, subTab4 = st.tabs(['감성 분석', '개선/불만 분석', '그룹핑', '워드 클라우드'])

    with subTab1:
        st.markdown('##### 감성 분석')

        i = 0

        stSub1 = st.columns([1, 1])

        df = pd.DataFrame()
        for cN in corNm:
            dic = {"기관별 감성분석": cN,
                   "감성": ["긍정", "부정"],
                   "값": np.random.randint(237, 872, size=2)
                   }
            df = df._append(pd.DataFrame(dic, index=np.arange(2)))
        if corNm:
            # st.write(df)
            # chart = alt.Chart(df, title='기관별 감성분석').mark_bar().encode(x='기관명', y='값', color='감성')
            # text = alt.Chart(df).mark_text(dx=0, dy=0, color='감성').encode(x='기관명', y='값', detail='감성', text=alt.Text('값:Q'))
            # stSub1[0].altair_chart(chart+text, use_container_width=True)

            if i > 1:
                i = 0
                stSub1 = st.columns([1, 1])

            stSub1[i].bar_chart(df, x="기관별 감성분석", y="값", color="감성", stack=False)
            i += 1


        df = pd.DataFrame()
        for dB in detailBiz:
            dic = {"업무유형별 감성분석": dB,
                   "감성": ["긍정", "부정"],
                   "값": np.random.randint(237, 872, size=2)
                   }
            df = df._append(pd.DataFrame(dic, index=np.arange(2)))
        if detailBiz:
            # st.write(df)
            if i > 1:
                i = 0
                stSub1 = st.columns([1, 1])

            stSub1[i].bar_chart(df, x="업무유형별 감성분석", y="값", color="감성", stack=False)
            i += 1

        df = pd.DataFrame()
        for dT in bizType:
            dic = {"상세업무별 감성분석": dT,
                   "감성": ["긍정", "부정"],
                   "값": np.random.randint(237, 872, size=2)
                   }
            df = df._append(pd.DataFrame(dic, index=np.arange(2)))
        if bizType:
            # st.write(df)

            if i > 1:
                i = 0
                stSub1 = st.columns([1, 1])
            stSub1[i].bar_chart(df, x="상세업무별 감성분석", y="값", color="감성", stack=False)
            i += 1

        df = pd.DataFrame()
        for gF in gendFlag:
            dic = {"성별 감성분석": gF,
                   "감성": ["긍정", "부정"],
                   "값": np.random.randint(237, 872, size=2)
                   }
            df = df._append(pd.DataFrame(dic, index=np.arange(2)))
        if gendFlag:
            # st.write(df)
            if i > 1:
                i = 0
                stSub1 = st.columns([1, 1])
            stSub1[i].bar_chart(df, x="성별 감성분석", y="값", color="감성", stack=False)
            i += 1

        df = pd.DataFrame()
        for aD in ageDiv:
            dic = {"연령대별 감성분석": aD,
                   "감성": ["긍정", "부정"],
                   "값": np.random.randint(237, 872, size=2)
                   }
            df = df._append(pd.DataFrame(dic, index=np.arange(2)))
        if ageDiv:
            # st.write(df)
            if i > 1:
                i = 0
                stSub1 = st.columns([1, 1])
            stSub1[i].bar_chart(df, x="연령대별 감성분석", y="값", color="감성", stack=False)
            i += 1

    with subTab2:
        st.markdown('##### 개선/불만 분석')


        i = 0

        stSub1 = st.columns([1, 1])

        df = pd.DataFrame()
        for cN in corNm:
            dic = {"기관별 개선/불만 분석": cN,
                   "구분": ["개선", "불만"],
                   "값": np.random.randint(237, 872, size=2)
                   }
            df = df._append(pd.DataFrame(dic, index=np.arange(2)))
        if corNm:
            # st.write(df)
            # chart = alt.Chart(df, title='기관별 감성분석').mark_bar().encode(x='기관명', y='값', color='감성')
            # text = alt.Chart(df).mark_text(dx=0, dy=0, color='감성').encode(x='기관명', y='값', detail='감성', text=alt.Text('값:Q'))
            # stSub1[0].altair_chart(chart+text, use_container_width=True)

            if i > 1:
                i = 0
                stSub1 = st.columns([1, 1])

            stSub1[i].bar_chart(df, x="기관별 개선/불만 분석", y="값", color="구분", stack=False)
            i += 1


        df = pd.DataFrame()
        for dB in detailBiz:
            dic = {"업무유형별 개선/불만 분석": dB,
                   "구분": ["개선", "불만"],
                   "값": np.random.randint(237, 872, size=2)
                   }
            df = df._append(pd.DataFrame(dic, index=np.arange(2)))
        if detailBiz:
            # st.write(df)
            if i > 1:
                i = 0
                stSub1 = st.columns([1, 1])

            stSub1[i].bar_chart(df, x="업무유형별 개선/불만 분석", y="값", color="구분", stack=False)
            i += 1

        df = pd.DataFrame()
        for dT in bizType:
            dic = {"상세업무별 개선/불만 분석": dT,
                   "구분": ["개선", "불만"],
                   "값": np.random.randint(237, 872, size=2)
                   }
            df = df._append(pd.DataFrame(dic, index=np.arange(2)))
        if bizType:
            # st.write(df)

            if i > 1:
                i = 0
                stSub1 = st.columns([1, 1])
            stSub1[i].bar_chart(df, x="상세업무별 개선/불만 분석", y="값", color="구분", stack=False)
            i += 1

        df = pd.DataFrame()
        for gF in gendFlag:
            dic = {"성별 개선/불만 분석": gF,
                   "구분": ["개선", "불만"],
                   "값": np.random.randint(237, 872, size=2)
                   }
            df = df._append(pd.DataFrame(dic, index=np.arange(2)))
        if gendFlag:
            # st.write(df)
            if i > 1:
                i = 0
                stSub1 = st.columns([1, 1])
            stSub1[i].bar_chart(df, x="성별 개선/불만 분석", y="값", color="구분", stack=False)
            i += 1

        df = pd.DataFrame()
        for aD in ageDiv:
            dic = {"연령대별 개선/불만 분석": aD,
                   "구분": ["개선", "불만"],
                   "값": np.random.randint(237, 872, size=2)
                   }
            df = df._append(pd.DataFrame(dic, index=np.arange(2)))
        if ageDiv:
            # st.write(df)
            if i > 1:
                i = 0
                stSub1 = st.columns([1, 1])
            stSub1[i].bar_chart(df, x="연령대별 개선/불만 분석", y="값", color="구분", stack=False)
            i += 1

    with subTab3:
        st.markdown('##### 그룹핑')

        stGroup = st.columns([0.5, 1, 0.5])
        url = './img/tobecon.png'
        stGroup[0].empty()
        stGroup[1].image(url)
        stGroup[2].empty()

    with subTab4:
        st.markdown('##### 워드 클라우드')

        stWord0 = st.columns([1, 1, 1])
        stWord1 = st.columns([1, 1, 1])
        df = pd.read_excel("형태소_분리_긍정.xlsx")

        tokens_ko = df['Token'].tolist()
        ko = nltk.Text(tokens_ko, name='기사 내 명사')

        new_ko = []
        for word in ko:
            if len(str(word)) > 1 and word != ' ':
                new_ko.append(word)

        ko = nltk.Text(new_ko, name='기사 내 명사 두 번째')
        data = ko.vocab().most_common(150)
        data = dict(data)

        colors = ['y', 'dodgerblue', 'C2', 'c', 'm']
        colors = ['y']

        top_nouns = dict(ko.vocab().most_common(10))  # 딕셔너리 형태로 상위 30개 저장
        fig = plt.figure(figsize=(5, 5))  # 이미지 사이즈를 설정하고 이미지 생성
        y_height = range(0, len(top_nouns))  # y축 높이 지정
        plt.barh(y_height, top_nouns.values(), color=colors)  # 수평막대 그리기
        plt.title("긍정 키워드 Top 10")  # 차트 제목 설정
        plt.yticks(y_height, top_nouns.keys())  # y축 틱에 label 붙이기
        stWord0[0].pyplot(fig)


        alice_mask = np.array(Image.open('./img/cloud.png'))
        # font = 'C:\Windows\Fonts\gulim.ttc'  # 이 친구는 코랩과는 다르다 ㅋㅋㅋㅋ
        wc = WordCloud(font_path=font_pt, \
                       background_color="white", \
                       width=400, \
                       height=400, \
                       max_words=100, \
                       max_font_size=300, mask=alice_mask, colormap='inferno')
        wc = wc.generate_from_frequencies(data)

        fig = plt.figure()  # 스트림릿에서 plot그리기
        plt.title('긍정 키워드')
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        stWord1[0].pyplot(fig)


        df = pd.read_excel("형태소_분리_부정.xlsx")

        tokens_ko = df['Token'].tolist()
        ko = nltk.Text(tokens_ko, name='기사 내 명사')

        new_ko = []
        for word in ko:
            if len(str(word)) > 1 and word != ' ':
                new_ko.append(word)

        ko = nltk.Text(new_ko, name='기사 내 명사 두 번째')
        data = ko.vocab().most_common(150)
        data = dict(data)

        colors = ['m']
        top_nouns = dict(ko.vocab().most_common(10))  # 딕셔너리 형태로 상위 30개 저장
        fig = plt.figure(figsize=(4, 4))  # 이미지 사이즈를 설정하고 이미지 생성
        y_height = range(0, len(top_nouns))  # y축 높이 지정
        plt.barh(y_height, top_nouns.values(), color=colors)  # 수평막대 그리기
        plt.title("부정 키워드 Top 10")  # 차트 제목 설정
        plt.yticks(y_height, top_nouns.keys())  # y축 틱에 label 붙이기
        stWord0[1].pyplot(fig)

        alice_mask = np.array(Image.open('./img/cloud.png'))
        # font = 'C:\Windows\Fonts\gulim.ttc'  # 이 친구는 코랩과는 다르다 ㅋㅋㅋㅋ
        wc = WordCloud(font_path=font, \
                       background_color="white", \
                       width=400, \
                       height=400, \
                       max_words=100, \
                       max_font_size=300, mask=alice_mask, colormap='nipy_spectral')
        wc = wc.generate_from_frequencies(data)

        fig = plt.figure()  # 스트림릿에서 plot그리기
        plt.title('부정 키워드')
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        stWord1[1].pyplot(fig)


        df = pd.read_excel("형태소_분리_개선.xlsx")

        tokens_ko = df['Token'].tolist()
        ko = nltk.Text(tokens_ko, name='기사 내 명사')

        new_ko = []
        for word in ko:
            if len(str(word)) > 1 and word != ' ':
                new_ko.append(word)

        ko = nltk.Text(new_ko, name='기사 내 명사 두 번째')
        data = ko.vocab().most_common(150)
        data = dict(data)

        colors = ['c']
        top_nouns = dict(ko.vocab().most_common(10))  # 딕셔너리 형태로 상위 30개 저장
        fig = plt.figure(figsize=(5, 5))  # 이미지 사이즈를 설정하고 이미지 생성
        y_height = range(0, len(top_nouns))  # y축 높이 지정
        plt.barh(y_height, top_nouns.values(), color=colors)  # 수평막대 그리기
        plt.title("개선 키워드 Top 10")  # 차트 제목 설정
        plt.yticks(y_height, top_nouns.keys())  # y축 틱에 label 붙이기
        stWord0[2].pyplot(fig)

        alice_mask = np.array(Image.open('./img/cloud.png'))
        font = 'C:\Windows\Fonts\gulim.ttc'  # 이 친구는 코랩과는 다르다 ㅋㅋㅋㅋ
        wc = WordCloud(font_path=font, \
                       background_color="white", \
                       width=400, \
                       height=400, \
                       max_words=100, \
                       max_font_size=300, mask=alice_mask, colormap='nipy_spectral')
        wc = wc.generate_from_frequencies(data)

        fig = plt.figure()  # 스트림릿에서 plot그리기
        plt.title('개선 키워드')
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        stWord1[2].pyplot(fig)
    # if btnSearch:
    #     st.markdown('##### 워드 클라우드')
