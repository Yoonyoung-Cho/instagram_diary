#https://discuss.streamlit.io/t/how-to-display-a-list-of-images-in-groups-of-10-50-100/32935

import pandas as pd 
import json
import streamlit as st
import glob
from datetime import datetime


st.markdown('# 검색기🔍')
st.markdown('일기 작성 기간과 키워드를 입력해서 일기를 검색해보세요!')


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
df['uri'] = df.media.map(lambda x: x['uri'])

df_index = df.title.map(lambda x: '복지' not in x)
df = df[df_index]
df_index = df.uri.map(lambda x: 'jpg'  in x)
df = df[df_index]
df['dt'] = [datetime.fromtimestamp(x) for x in df['creation_timestamp']]
df['year'] = df['dt'].map(lambda x: str(x.year))
df['month'] = df['dt'].map(lambda x: x.month)
df['day'] = df['dt'].map(lambda x: x.day)
df['number'] = df.title.rank(method='dense').astype(int)
df_nodup = df[["title", 'year', 'month', 'day', 'number']].drop_duplicates(keep='last')
df_nodup.reset_index(drop=True).head()


keyword = st.text_input("검색하고 싶은 키워드를 입력하세요.", '----') 
year = st.multiselect("기간 선택",
               list(df.year.unique()),
               list(df.year.unique())[0:1],)

# search_click = st.button('검색하기')

# if search_click:
data = df_nodup.loc[df_nodup.year.isin(year),:]
data_index = data.title.map(lambda x: keyword in x)
data = data[data_index]
data = data[['number', 'title', 'year', 'month', 'day']]
# st.dataframe(data)

manuscripts = st.multiselect("일기 선택", data.title, data.title)
paths = df.loc[df.title.isin(manuscripts),'uri']
paths = [s for s in paths if "jpg" in s] 

n = st.number_input("Select Grid Width", 5, 10, 6)

groups = []
for i in range(0, len(paths), n):
    groups.append(paths[i:i+n])


try : 
    for group in groups:
        cols = st.columns(n)
        for i, image in enumerate(group):
            
            cols[i].image(f"../instagram_diary/{image}")

    txt = data.loc[data.title.isin(manuscripts),'title'].reset_index(drop=True)[0]
    diary_txt = st.text_area("일기내용", txt, label_visibility='hidden', height=400)
except: pass



# 참고 : https://www.youtube.com/watch?v=GG6WDeLmw6w