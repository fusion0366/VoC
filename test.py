import streamlit as st
from vega_datasets import data
import pandas as pd
import numpy as np

source = data.barley()

st.bar_chart(source, x="year", y="yield", color="site", stack=False)
st.write(source)

detailBiz = st.multiselect(
    "상세업무",
    ['경영지원 서비스', '연구지원 서비스', '교육 훈련', '정보 제공', '육성 및 촉진', '기본연구',
     '수탁연구', '위탁/공동연구', '학위(외국인 졸업생, ', '학위(내국인 졸업생, ', '학위(외국인 재학생, ',
     'KRIVET 메일 서비스', '커리어넷 홈페이지', '자격 시험 주관', '정보제공 서비스(KDIS, -내국인 졸업생',
     '정보제공 서비스(KDIS, -외국인 재학생'])

df = pd.DataFrame()
for db in detailBiz:
    dic = {"col1": db,
        "col2": [ "긍정", "부정" ],
        "col3": np.random.randint(237, 872, size=2)
    }
    df = df._append(pd.DataFrame(dic, index = np.arange(2) ))

st.write(df)
st.bar_chart(df, x="col1", y="col3", color="col2", stack=False)

data = { 'a' : 100, 'b' : ["긍정", "부정"] , "col3": np.random.randint(237, 872, size=2)}
df = pd.DataFrame(data, index = np.arange(2) )

data = { 'a' : 200, 'b' : ["긍정", "부정"] , "col3": np.random.randint(237, 872, size=2)}

df = df._append(pd.DataFrame(data, index = np.arange(2) ))
print(df)