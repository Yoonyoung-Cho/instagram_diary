#https://discuss.streamlit.io/t/how-to-display-a-list-of-images-in-groups-of-10-50-100/32935

import pandas as pd 
import json
import streamlit as st
import glob
from datetime import datetime

# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()


from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate


st.markdown('# 일기장 검색기 🔍')
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
df['uri'] = df.media.map(lambda x: x['uri'])
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


# manuscripts = st.multiselect("일기 선택", data.title, data.title)
# paths = df.loc[df.title.isin(manuscripts),'uri']
# paths = [s for s in paths if "jpg" in s] 

# n = st.number_input("Select Grid Width", 5, 10, 6)

# groups = []
# for i in range(0, len(paths), n):
#     groups.append(paths[i:i+n])

# for group in groups:
#         cols = st.columns(n)
#         for i, image in enumerate(group):
            
#             cols[i].image(f"../instagram_diary/{image}")

# txt = data.loc[data.title.isin(manuscripts),'title'].reset_index(drop=True)[0]
# diary_txt = st.text_area("일기내용", txt, label_visibility='hidden', height=400)

# try : 
#     for group in groups:
#         cols = st.columns(n)
#         for i, image in enumerate(group):
            
#             cols[i].image(f"../instagram_diary/{image}")

#     txt = data.loc[data.title.isin(manuscripts),'title'].reset_index(drop=True)[0]
#     diary_txt = st.text_area("일기내용", txt, label_visibility='hidden', height=400)
# except: pass


# 일기 제목 요약 
prompt_txt = """당신은 유능한 작가입니다. 제공된 일기장 데이터의 제목을 지어주세요. 
8글자 이내의 제목을 작성하고, 맥락에 맞는 제목이어야 합니다.
제목만을 답변으로 제공하세요. 단계별로 생각하고 답변을 작성해주세요. 

Question: context의 일기의 제목을 지어주세요. 
Context: 맛잇엇던 아웃백허ㅣ식 가려고할때 없는@칸타카드로 구질구질하게 재결제하러가야해서 웃겻다 파티계시니까 분위기가많이 달라져서 너무 좋다 파 티 조 아
Answer: 파티 좋아

Question: {question} 
Context: {context} 
Answer:
"""

prompt=ChatPromptTemplate(input_variables=['context', 'question'],
                #    metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt', 'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'},
                   messages=[HumanMessagePromptTemplate(
                       prompt=PromptTemplate(input_variables=['context', 'question'], 
                                             template=prompt_txt))])

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)



rag_chain2 = (
        # {"context": data.title.reset_index(drop=True)[i], "question": RunnablePassthrough()}
        prompt
        | llm
        | StrOutputParser()
    )

summary = []
for i in range(0,len(data.title)):
    response = rag_chain2.invoke({'context':data.title.reset_index(drop=True)[i], 'question':"일기의 제목을 지어주세요."})
    summary.append(response)
    
title_summary = pd.concat([pd.DataFrame(summary, columns=['summary']), data.title.reset_index(drop=True)], axis=1)


df_tmp = pd.merge(df, title_summary, on='title')
manuscripts = st.multiselect("일기 선택", summary, summary)
title_index = title_summary.loc[title_summary.summary.isin(manuscripts),'title'].to_list()

paths = df_tmp.loc[df_tmp.title.isin(title_index),'uri']
paths = [s for s in paths if "jpg" in s] 

n = st.number_input("Select Grid Width", 5, 10, 6)

groups = []
for i in range(0, len(paths), n):
    groups.append(paths[i:i+n])
    
for group in groups:
        cols = st.columns(n)
        for i, image in enumerate(group):
            
            cols[i].image(f"../instagram_diary/{image}")
            
if len(df_tmp.loc[df_tmp.summary.isin(manuscripts),'title'].unique())!=0 :
    txt = df_tmp.loc[df_tmp.summary.isin(manuscripts),'title'].unique()[0]
    diary_txt = st.text_area("일기내용", txt, label_visibility='hidden', height=400)



# 참고 : https://www.youtube.com/watch?v=GG6WDeLmw6w
# https://github.com/wjbmattingly/youtube-streamlit-image-grid/blob/main/demo.py
# https://wikidocs.net/235091