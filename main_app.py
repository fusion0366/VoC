import streamlit as st
from streamlit_option_menu import option_menu
from project_app import project_page
from upload_app import upload_page
from analyze_app import analyze_page
from detail_app import detail_page

# login Page 만들기
def main_page():

    with st.sidebar:
        st.markdown('''
                <p class="mb-5" style="text-align: center;">
                    <img alt="" loading="lazy" width="200" decoding="async" data-nimg="1" src="https://intense-ai.com/img/KMACVOC.png" style="color: transparent;">
                </p>
                ''', unsafe_allow_html=True)
        menu = option_menu('홍길동 / 공공리서치', ['프로젝트 관리', '결과 Upload', '분석 보고서(기본)', '분석 보고서(상세)'],
                           icons=['gear-wide-connected', 'bi bi-upload', 'bar-chart-line-fill', 'bi bi-robot'],
                           # menu_icon="/img/1320909.png", default_index=0,
                           menu_icon="bi bi-person-fill", default_index=0,
                           styles={
                               "container": {"padding": "5!important", "background-color": "#fafafa"},
                               "icon": {"font-size": "22px"},
                               "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "margin-top": "10px",
                                            "--hover-color": "#eee"},
                               "nav-link-selected": {"background-color": "#434B69", "icon-color": "#fafafa"}, })
                               # "nav-link-selected": {"background-color": "#ef494c"}, })


    if menu == '프로젝트 관리':
        project_page()
    elif menu == '결과 Upload':
        upload_page()
    elif menu == '분석 보고서(기본)':
        analyze_page()
    elif menu == '분석 보고서(상세)':
        detail_page()
    else: # ABOUT 페이지
        st.subheader('HOME')
