import streamlit as st
import pandas as pd

def project_page():
    stTitle = st.columns([0.1, 3.0])
    stTitle[0].markdown('''
    <p class="mb-5" style="text-align: center;">
        <img alt="" loading="lazy" width="40" decoding="async" data-nimg="1" src="https://intense-ai.com/img/project.png" style="color: transparent;margin-top: 17px;">
    </p>
    ''', unsafe_allow_html=True)
    stTitle[1].header('프로젝트 관리')

    con1, con2 = st.columns([1.8, 1.0])

    dfm = pd.DataFrame(data={'프로젝트': ['국무조정실 에너지연구원', '시니어 대상 조사', '일반인 광고 조사', '일반인 의견 조사', '2024 통신 서비스 관련 소비자 조사',
                                     '2024 소비자 의견 조사', '광고 관련 소비자 조사', '주류 관련 소비자 의견 조사'],
                            '거래처': ['(주)★★★★', '(주)○○○○', '(주)■■■■■', '(주)◆◆◆◆◆◆◆', '(주)●●●●●●', '(주)□□□□□□□',
                                    '(주)▽▽▽▽▽▽', '(주)♠♠♠♠♠♠'],
                            '기간': ['2024.01.03 ~ 2024.12.31', '2024.02.03 ~ 2024.07.02', '2024.04.11 ~ 2024.10.30',
                                   '2024.04.015 ~ 2024.05.30', '2024.05.17 ~ 2024.11.30', '2024.05.19 ~ 2024.12.15',
                                   '2024.06.01 ~ 2024.11.30', '2024.06.03 ~ 2024.11.15'],
                            '구분': ['기타', '기타', '광고/컨셉', '공공부문 실태/인식', 'IT/이동통신', '기타', '디지털/가전', '기타'],
                            '모집인원': ['1,200', '850', '2,500', '2,800', '3,500', '5,000', '2,700', '2,300']})

    dfd = pd.DataFrame(data={'회차': ['1회차', '2회차', '3회차', '4회차', '5회차', '6회차', '7회차'],
                            '조사기간': ['2024.02.01 ~ 2024.02.10', '2024.03.01 ~ 2024.03.10', '2024.04.01 ~ 2024.04.10',
                                     '2024.05.01 ~ 2024.05.10', '2024.06.01 ~ 2024.06.10', '2024.07.01 ~ 2024.07.10',
                                     '2024.08.01 ~ 2024.08.10'],
                            '조사방법': ['전화조사', '우편조사', '방문조사', '온라인조사', '면접조사', '집단설문조사', '대면조사'],
                            '모집인원': [120, 60, 200, 323, 184, 212, 112]})
    with con1:
        st.subheader('프로젝트 정보')
        st.dataframe(dfm, width=900, hide_index=True, on_select="rerun", selection_mode="multi-row",)
        # st.dataframe(dfm, height=400, width=700)

    with con2:
        st.subheader('회차 정보')
        st.dataframe(dfd, width=600, hide_index=True, on_select="rerun", selection_mode="multi-row",)
        # st.dataframe(dfd)
