import pandas as pd 
from datetime import datetime 
import json

import streamlit as st

# st.set_page_config(
#     page_title="일기장 검색기",
#     page_icon="🎮",
#     # layout="wide",
#     initial_sidebar_state="auto",
# )
 
# Load data 
with open ("../instagram_diary/content/posts_1.json", "r") as f:
    data = json.load(f)
df = pd.DataFrame()
for i in range(len(data)):
    df = pd.concat([df,pd.DataFrame(data[i])])
df = df.map(lambda x: x.encode('latin-1').decode('utf-8') if isinstance(x, str) else x)
df = df.reset_index(drop=True)

#  전처리 
df.loc[df.title.isna(),'creation_timestamp']= [x.get('creation_timestamp') for x in df.loc[df.title.isna()]['media']]
df.loc[df.title.isna(),'title']= [x.get('title').encode('latin-1').decode('utf-8') for x in df.loc[df.title.isna()]['media']]
df["title"] = df.title.map(lambda x: x.replace('ㅅㅂ', '행복해').replace('시발', '사랑해').replace('ㅈ까', '응원해').replace('ㅈ됐다', '잘될거야').replace('해피아워', '좋은 시간').replace('해피 아워', '좋은 시간').replace('오빠', '김땡땡이').replace('유환', '땡땡').replace('스티븐', '탕탕').replace('남친', '인이설이').replace('귀염둥이', '굼바'))
df_index = df.title.map(lambda x: '복지' not in x)
df = df[df_index]
df['dt'] = [datetime.fromtimestamp(x) for x in df['creation_timestamp']]
df['year'] = df['dt'].map(lambda x: str(x.year))
df['month'] = df['dt'].map(lambda x: x.month)
df['day'] = df['dt'].map(lambda x: x.day)
df_nodup = df[["title", 'year', 'month', 'day']].drop_duplicates(keep='last')
df_nodup.reset_index(drop=True).head()



st.set_page_config(
    page_title="일기장 검색기",
    page_icon="🎮",
    # layout="wide",
    initial_sidebar_state="auto",
)

st.markdown("# Instagram 일기장 검색기 🔍")

st.markdown(f"안녕하세요! `yooniary` 일기장의 검색기입니다🤗 \n 현재 일기장에는 {df_nodup.shape[0]}개의 일기가 있어요. ")