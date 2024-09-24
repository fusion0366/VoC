import random

import streamlit
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from kospellpy import spell_init
from networkx.algorithms.bipartite import color
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
def analyze_page():
    stTitle = st.columns([0.1, 3.0])
    stTitle[0].markdown('''
    <p class="mb-5" style="text-align: center;">
        <img alt="" loading="lazy" width="40" decoding="async" data-nimg="1" src="https://intense-ai.com/img/analyze.png" style="color: transparent;margin-top: 17px;">
    </p>
    ''', unsafe_allow_html=True)
    stTitle[1].header('분석 보고서 (기본)')

    # 선택된 옵션 표시하기

    stProject = st.columns([2.0, 1.5, 0.5])
    project_option = stProject[0].selectbox('프로젝트', ['1. 국무조정실 에너지연구원', '2. 시니어 대상 조사', '3. 일반인 광고 조사', '4. 일반인 의견 조사',
                                                      '5. 2024 통신 서비스 관련 소비자 조사', '6. 2024 소비자 의견 조사', '7. 광고 관련 소비자 조사',
                                                      '8. 주류 관련 소비자 의견 조사'])
    round_option = stProject[1].selectbox('회차', ['1회차', '2회차', '3회차', '4회차', '5회차', '6회차', '7회차'])
    stProject[2].markdown("""
    <style>
    [data-testid="baseButton-primary"] {
        height: 800px;
        background-color: #0070C0;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    btnSearch = stProject[2].button('상세 분석', type="primary", use_container_width=True)

    # 메인 콘텐츠 영역
    # tab1, tab2 = st.tabs(['기초 분석', '상세 분석'])

    if btnSearch:
        # with tab1:

        df = pd.read_excel("국무조정실_VOC 원본 데이터_보정240813.xlsx")

        # 데이터 타입 최적화
        df = optimize_dtypes(df)

        stTitle1 = st.columns([1, 1])
        stTitle1[0].markdown("##### ♠ 성별 분석")
        stTitle1[1].markdown("##### ♠ 기관별 분석")

        stChart1 = st.columns([1, 1])

        genderCnt = df['성별'].value_counts()
        plot = pd.DataFrame({'성별': genderCnt.index, '건수': genderCnt.values})
        fig = px.pie(plot, values='건수', names='성별', hole=0.3)
        stChart1[0].plotly_chart(fig)

        corpCnt = df['기관명'].value_counts()
        plot = pd.DataFrame({'기관': corpCnt.index, '건수': corpCnt.values})
        fig = px.bar(plot, x='기관', y='건수')
        stChart1[1].plotly_chart(fig)

        stTitle2 = st.columns([1, 1])
        stTitle2[0].markdown("##### ♠ 연령대 분석")
        stTitle2[1].markdown("##### ♠ 근속연수 분석")

        stChart2= st.columns([1, 1])

        plot = pd.DataFrame({'연령대': df['연령대'].value_counts().index, '건수': df['연령대'].value_counts().values})
        fig = px.pie(plot, values='건수', names='연령대', hole=0.3)
        stChart2[0].plotly_chart(fig)

        plot = pd.DataFrame({'근속연수': df['근속연수'].value_counts().index, '건수': df['근속연수'].value_counts().values})
        fig = px.pie(plot, values='건수', names='근속연수', hole=0.3)
        stChart2[1].plotly_chart(fig)


        stTitle3 = st.columns([1, 1])
        stTitle3[0].markdown("##### ♠ 응답자 거주지(지역) 분석")
        stTitle3[1].markdown("##### ♠ 응답자 거주지(지역) Map 분석")

        stChart3= st.columns([1, 1])

        plot = pd.DataFrame({'응답자 거주지(지역)': df['응답자 거주지(지역)'].value_counts().index, '건수': df['응답자 거주지(지역)'].value_counts().values})
        fig = px.pie(plot, values='건수', names='응답자 거주지(지역)', hole=0.3)
        stChart3[0].plotly_chart(fig)

        size = np.random.randint(1, 300, 20) * 100
        hex_colors = [generate_hex_color() for _ in range(2)]

        # plot = pd.DataFrame({'lat': df['lat'], 'lon': df['lon'], 'size': size, 'color': hex_colors})

        plot = pd.DataFrame({'lat': df['lat'], 'lon': df['lon']})
        plot.dropna(axis=0)
        # print(range(plot.shape[0]))
        # for i in range(plot.shape[0]):
        #     plot[i, 'lat'] = plot['lat'][i] + np.random.randn() / 50.0
        #     plot[i, 'lon'] = plot['lon'][i] + np.random.randn() / 50.0
        #
        # print(plot)
        # stChart3[1].map(data=plot, zoom=6, size='size', color='color')
        stChart3[1].map(data=plot, zoom=6)
        # with tab2:
            # 상세 분석

def generate_hex_color(alpha='80'):
    return '#{0:02X}{1:02X}{2:02X}{3}'.format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), alpha)