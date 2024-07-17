import streamlit as st
import json
import re 
import pandas as pd
from datetime import datetime
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import FAISS

# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()


# Load data 
with open ("instagram_diary/content/posts_1.json", "r") as f:
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
df_index = df.title.map(lambda x: '행크' in x)
df = data[df_index]
df['dt'] = [datetime.fromtimestamp(x) for x in df['creation_timestamp']]
df['year'] = df['dt'].map(lambda x: str(x.year))
df['month'] = df['dt'].map(lambda x: x.month)
df['day'] = df['dt'].map(lambda x: x.day)
df_nodup = df[["title", 'year', 'month', 'day']].drop_duplicates(keep='last')
print(df_nodup.shape)
df_nodup.reset_index(drop=True).head()
















# RAG --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
loader = DataFrameLoader(df_nodup, page_content_column="title")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
split_docs = loader.load_and_split(text_splitter=text_splitter)
vectorstore = FAISS.load_local('./db/faiss', HuggingFaceBgeEmbeddings(), allow_dangerous_deserialization=True)

k = 6
bm25_retriever = BM25Retriever.from_documents(split_docs)
bm25_retriever.k = k
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)

prompt = hub.pull("rlm/rag-prompt")

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)





# Streamlit --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="일기장 검색기",
    page_icon="🎮",
    # layout="wide",
    initial_sidebar_state="auto",
)

st.markdown("# Instagram 일기장 검색기 🔍")

st.markdown(f"안녕하세요! `yooniary` 일기장의 검색기입니다🤗 \n 현재 일기장에는 {df_nodup.shape[0]}개의 일기가 있어요. ")

keyword = st.text_input("검색하고 싶은 키워드를 입력하세요.", '카겜') 

year = st.multiselect("기간 선택",
               list(df.year.unique()),
               ['2021', '2022'],)


search_click = st.button('검색하기')

# if search_click:
data = df_nodup.loc[df_nodup.year.isin(year),:]
data_index = data.title.map(lambda x: keyword in x)
data = data[data_index]
st.dataframe(data)

query = st.text_input("알고싶은 내용에 대해서 입력하세요.") 

if query:
    st.text(ensemble_retriever.get_relevant_documents(query))


    response = rag_chain.invoke(query)
    st.text(response)