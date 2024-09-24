import streamlit as st # streamlit 모듈 import
from PIL import Image

from login_app import login_page
from main_app import main_page

img_dir = 'https://www.kmac.co.kr/kmac_favicon.ico'

st.set_page_config(layout="wide", page_title="KMAC VoC", page_icon=img_dir)

# 세션 상태 초기화
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False


# 로그인 상태에 따라 페이지 표시
if st.session_state['authenticated']:
    main_page()
else:
    login_page()