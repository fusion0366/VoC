import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import plotly.express as px

def upload_page():
    stTitle = st.columns([0.1, 3.0])
    stTitle[0].markdown('''
    <p class="mb-5" style="text-align: center;">
        <img alt="" loading="lazy" width="40" decoding="async" data-nimg="1" src="https://intense-ai.com/img/upload.png" style="color: transparent;margin-top: 17px;">
    </p>
    ''', unsafe_allow_html=True)
    stTitle[1].header('프로젝트 결과 등록')

    # 사용자 정의 형식으로 선택 상자 생성하기


    # 선택된 옵션 표시하기

    stProject = st.columns([2.0, 1.5])
    project_option = stProject[0].selectbox('프로젝트', ['1. 국무조정실 에너지연구원', '2. 시니어 대상 조사', '3. 일반인 광고 조사', '4. 일반인 의견 조사',
                                                      '5. 2024 통신 서비스 관련 소비자 조사', '6. 2024 소비자 의견 조사', '7. 광고 관련 소비자 조사',
                                                      '8. 주류 관련 소비자 의견 조사'])
    round_option = stProject[0].selectbox('회차', ['1회차', '2회차', '3회차', '4회차', '5회차', '6회차', '7회차'])
    uploaded_file = stProject[1].file_uploader("데이터 파일 업로드 (CSV, Excel, Pickle)", type=['csv', 'xlsx', 'pickle'])

    df = None

    with st.spinner('분석중....'):
        # 업로드된 파일로부터 데이터프레임 로드
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith(('.pkl', '.pickle')):
                df = pickle.load(uploaded_file, encoding='utf-8-sig')

            # 데이터 타입 최적화
            df = optimize_dtypes(df)

            # 텍스트 칼럼 선택
            if df is not None:
                target_column = st.sidebar.selectbox("텍스트 변수 선택", df.columns)
                st.session_state['target_column'] = target_column  # 세션 상태에 텍스트 칼럼 저장

            # 메인 콘텐츠 영역
            tab0, tab1, tab2, tab3, tab4 = st.tabs(['원본', 'N/N+ Sentence', '감정 분석', '개선/불만 분석', '단어분석'])

            with tab0:
                st.markdown('#### 원본 데이터')
                st.dataframe(df)


            with tab1:
                split_df = pd.read_csv("[생성형AI]voc_국무조정500.csv")

                # 문장 분리된 데이터
                st.markdown('#### N+ sentence')
                st.dataframe(split_df)

            with tab2:
                # 감정 분석
                st.markdown('#### 감정 분석')
                sentiment = pd.read_excel("voc_scipt_생소리_500_N+_분류예측.xlsx")
                sentiment = sentiment.drop(columns='label1')

                label_counts = sentiment['label2'].value_counts()
                label_percentages = (label_counts / label_counts.sum()) * 100

                label_counts_formatted = label_counts.apply(lambda x: f"{x:d}")
                label_percentages_formatted = label_percentages.apply(lambda x: f"{x:.2f}")

                label_summary = pd.DataFrame({
                    '개수': label_counts_formatted,
                    '비율': label_percentages_formatted
                }).transpose()

                st.dataframe(label_summary, width=1400)

                stSentiment = st.columns([1.0, 1.0, 1.0])
                i = 0
                for label in label_counts.index:
                    stSentiment[i].markdown(f"#### {label} 감정")
                    label_data = sentiment[sentiment['label2'] == label][['SCRIPT']]
                    stSentiment[i].dataframe(label_data, width=450)
                    i += 1


            with tab3:
                # 불만/개선 분석
                st.markdown('#### 불만/개선 분석')
                sentiment = pd.read_excel("voc_scipt_생소리_500_N+_분류예측.xlsx")
                sentiment = sentiment.drop(columns='label2')

                label_counts = sentiment['label1'].value_counts()
                label_percentages = (label_counts / label_counts.sum()) * 100

                label_counts_formatted = label_counts.apply(lambda x: f"{x:d}")
                label_percentages_formatted = label_percentages.apply(lambda x: f"{x:.2f}")

                label_summary = pd.DataFrame({
                    '개수': label_counts_formatted,
                    '비율': label_percentages_formatted
                }).transpose()

                st.dataframe(label_summary, width=1400)

                stSentiment = st.columns([1.0, 1.0, 1.0])
                i = 0
                for label in label_counts.index:
                    stSentiment[i].markdown(f"#### {label} 의견")
                    label_data = sentiment[sentiment['label1'] == label][['SCRIPT']]
                    # st.dataframe(label_data, height=400, width=700)
                    stSentiment[i].dataframe(label_data, width=450)
                    i += 1

            with tab4:
                st.markdown('#### 단어 분석')
                stWord = st.columns([1.0, 1.0, 1.0])

                stWord[0].markdown(f"#### 긍정 단어 분석")
                split_df = pd.read_excel("형태소_분리_긍정.xlsx")
                stWord[0].dataframe(split_df, width=420)

                stWord[1].markdown(f"#### 부정 단어 분석")
                split_df = pd.read_excel("형태소_분리_부정.xlsx")
                stWord[1].dataframe(split_df, width=420)

                stWord[2].markdown(f"#### 개선 단어 분석")
                split_df = pd.read_excel("형태소_분리_개선.xlsx")
                stWord[2].dataframe(split_df, width=420)



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


