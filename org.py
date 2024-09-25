import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from kospellpy import spell_init
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import openai
import plotly.express as px
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# OpenAI API 키 설정
openai.api_key = ''

# 폰트 설정
font_name = font_manager.FontProperties(fname='fonts\\NanumBarunGothic.ttf').get_name()
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
def main_page():
    try:
        # 페이지 제목
        st.title('VoC 데이터 분석📊')

        # 페이지 설명
        st.write('''
            데이터를 업로드하고, 관심 있는 결과를 얻어보세요.
        ''')

        st.sidebar.title('🗣️VoC 데이터 분석')

        # 데이터 파일 업로드
        uploaded_file = st.sidebar.file_uploader("데이터 파일 업로드 (CSV, Excel, Pickle)", type=['csv', 'xlsx', 'pickle'])
        df = None

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
            tab1, tab2, tab3, tab4, tab5 = st.tabs(['N/N+ Sentence', '감정 분석', '개선/불만 분석', '그룹핑', '모델 활용'])

            with tab1:
                # 원본 =텍스트 전처리 데이터 데이터프레임 생성
                origin = df[target_column]
                origin_df = origin.apply(clean_text)
                st.session_state['origin_df'] = origin_df

                split_df = pd.read_csv("voc_script_생소리_500_N+.csv")

                # 원본 데이터
                st.markdown('#### N sentence')
                st.dataframe(origin_df, height=400, width=700)

                # 데이터프레임을 CSV로 변환
                csv = origin_df.to_csv(index=False).encode('cp949')

                # 다운로드 버튼 생성
                st.download_button(
                    label="다운로드",
                    data=csv,
                    file_name='origin_sentences.csv',
                    mime='text/csv',
                )

                # 문장 분리된 데이터
                st.markdown('#### N+ sentence')
                st.dataframe(split_df, height=400, width=700)

                # 데이터프레임을 CSV로 변환
                csv = split_df.to_csv(index=False).encode('cp949')

                # 다운로드 버튼 생성
                st.download_button(
                    label="다운로드",
                    data=csv,
                    file_name='split_sentences.csv',
                    mime='text/csv',
                )

                st.markdown('#### N/N+ sentence 비교')
                values = []
                for column in split_df.columns:
                    values.extend(split_df[column].replace('', pd.NA).dropna().tolist())
                new_df = pd.DataFrame(values, columns=['의견'])
                st.session_state['new_df'] = new_df

                # 막대그래프 시각화
                count_data_chart = pd.DataFrame({
                    'Type': ['N sentence', 'N+ sentence'],
                    'Count': [origin_df.shape[0], new_df.shape[0]]
                })
                fig = px.bar(count_data_chart, x='Type', y='Count', color='Type',
                             color_discrete_map={'N sentence': 'lightblue', 'N+ sentence': 'blue'})
                st.plotly_chart(fig)

                # 표 형태
                count_data_table = pd.DataFrame({
                    'N sentence': [origin_df.shape[0]],
                    'N+ sentence': [new_df.shape[0]]
                }, index=['개수'])
                st.table(count_data_table)

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

                # fig, ax = plt.subplots()
                # ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
                # ax.axis('equal')
                # st.pyplot(fig)

                plot = pd.DataFrame({'label': label_counts.index, 'count': label_counts.values})
                fig = px.pie(plot, values='count', names='label', hole=0.3)
                st.plotly_chart(fig)

                st.table(label_summary)

                # 공백 추가
                st.markdown("<br><br>", unsafe_allow_html=True)

                for label in label_counts.index:
                    st.markdown(f"#### {label} 감정")
                    label_data = sentiment[sentiment['label2'] == label][['SCRIPT']]
                    st.dataframe(label_data, height=400, width=700)

                    # 데이터프레임을 CSV로 변환
                    csv = label_data.to_csv(index=False).encode('cp949')

                    # 다운로드 버튼 생성
                    st.download_button(
                        label="다운로드",
                        data=csv,
                        file_name=f'{label}_sentences.csv',
                        mime='text/csv',
                    )

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

                # fig, ax = plt.subplots()
                # ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
                # ax.axis('equal')
                # st.pyplot(fig)

                plot = pd.DataFrame({'label': label_counts.index, 'count': label_counts.values})
                fig = px.pie(plot, values='count', names='label', hole=0.3)
                st.plotly_chart(fig)

                st.table(label_summary)

                # 공백 추가
                st.markdown("<br><br>", unsafe_allow_html=True)

                for label in label_counts.index:
                    st.markdown(f"#### {label} 의견")
                    label_data = sentiment[sentiment['label1'] == label][['SCRIPT']]
                    st.dataframe(label_data, height=400, width=700)

                    # 데이터프레임을 CSV로 변환
                    csv = label_data.to_csv(index=False).encode('cp949')

                    # 다운로드 버튼 생성
                    st.download_button(
                        label="다운로드",
                        data=csv,
                        file_name=f'{label}_sentences.csv',
                        mime='text/csv',
                    )

            with tab4:
                st.markdown('#### 그룹핑')

            with tab5:
                st.markdown('#### 감정 분석 테스트')

                def load_model(model_path):
                    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)
                    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                    model.eval()
                    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
                    return model, tokenizer

                def predict(text, model, tokenizer):
                    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
                    with torch.no_grad():
                        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                        predictions = torch.argmax(outputs.logits, dim=-1)
                        predicted_label = predictions.item()
                    return predicted_label

                # 모델과 토크나이저 로드
                model_path = "model_state_multi_language_젤잘나옴.pth"
                model, tokenizer = load_model(model_path)

                # 라벨 맵핑
                label_map = {0: '긍정', 1: '부정', 2: '중립'}

                # 사용자 입력
                user_input = st.text_input("감정 분석을 위한 텍스트를 작성해주세요")

                if st.button("예측하기"):
                    if user_input:
                        prediction = predict(user_input, model, tokenizer)
                        st.markdown(
                            f"""
                            <div style="background-color: #f0f0f5; padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
                                <h5>예측 라벨</h5>
                                <p>{label_map[prediction]}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.write("예측을 위한 텍스트를 작성해주세요")

                # 공백 추가
                st.markdown("<br><br>", unsafe_allow_html=True)

                st.markdown('#### 개선/불만 분석 테스트')

                def load_model(model_dir):
                    model = BertForSequenceClassification.from_pretrained(model_dir)
                    tokenizer = BertTokenizer.from_pretrained(model_dir)
                    return model, tokenizer

                def predict(text, model, tokenizer):
                    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
                    with torch.no_grad():
                        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                        predictions = torch.argmax(outputs.logits, dim=-1)
                        predicted_label = predictions.item()
                    return predicted_label

                # 모델과 토크나이저 로드
                model_dir = "개선불만예측모델"
                model, tokenizer = load_model(model_dir)

                # 라벨 맵핑
                label_map = {0: '개선', 1: '불만', 2: '없음'}

                # 사용자 입력
                user_input = st.text_input("개선/불만 분석을 위한 텍스트를 작성해주세요")

                if st.button("예측하기", key="predict_button"):
                    if user_input:
                        prediction = predict(user_input, model, tokenizer)
                        st.markdown(
                            f"""
                            <div style="background-color: #f0f0f5; padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
                                <h5>예측 라벨</h5>
                                <p>{label_map[prediction]}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.write("예측을 위한 텍스트를 작성해주세요")

    except UnicodeDecodeError as e:
        st.error("업로드한 파일의 인코딩 형식이 올바르지 않습니다. UTF-8 인코딩 형식으로 파일을 저장해주세요.")


main_page()
