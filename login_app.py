import streamlit as st
from pyparsing import empty

# login Page 만들기
def login_page():

    empty1, con1, empty2 = st.columns([0.5, 1.0, 0.5])

    with empty1:
        empty()  # 여백부분1

    with empty2:
        empty()  # 여백부분1

    st.markdown("""
    <style>
    [data-testid="StyledFullScreenButton"] {
        visibility: hidden;
    }
    [data-testid="stToolbar"] {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    [data-testid="baseButton-secondaryFormSubmit"] {
        background-color: #0070C0;
        color: white;
        width: 100%;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    with con1.form(key="form"):
        lginfrm = st.columns([2, 2])
        url = 'http://3.35.101.137/_next/static/media/login_img.20f74788.jpg'
        url = './img/img_4.png'
        lginfrm[0].image(url, width=414)
        # username = lginfrm[1].text_input('')
        # lginfrm[1].title('KmAC VoC 분석 솔루션')
        # lginfrm[1].image('img/KMACVOC.png', width=200)

        lginfrm[1].markdown('''
        <p class="mb-5" style="text-align: center;">
            <img alt="" loading="lazy" width="200" decoding="async" data-nimg="1" src="https://intense-ai.com/img/KMACVOC.png" style="color: transparent;margin-top: 79px;">
        </p>
        ''', unsafe_allow_html=True)
        username = lginfrm[1].text_input('로그인 ID')
        password = lginfrm[1].text_input('비밀번호', type='password')
        submit = lginfrm[1].form_submit_button(label="로그인")
        # 'style="margin-top: 112px;"'
        if submit:
            if not username:
                st.error("로그인 ID를 입력하세요.")
            elif not password:
                st.error("비밀번호를 입력하세요.")
            else:
                st.session_state['authenticated'] = True

                # 현재 상태 값을 true로 변경 후 다시 실행하여 화면을 변경함
                st.rerun()
