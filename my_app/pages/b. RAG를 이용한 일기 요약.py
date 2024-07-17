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
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate

# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()

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



loader = DataFrameLoader(df_nodup, page_content_column="title")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
split_docs = loader.load_and_split(text_splitter=text_splitter)
# vectorstore = FAISS.load_local('../db/faiss', HuggingFaceBgeEmbeddings(), allow_dangerous_deserialization=True)
vectorstore = FAISS.load_local('../db/faiss_openai', OpenAIEmbeddings(), allow_dangerous_deserialization=True)

k = 6
bm25_retriever = BM25Retriever.from_documents(split_docs)
bm25_retriever.k = k
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)

# prompt = hub.pull("rlm/rag-prompt")

prompt_txt = """당신은 유능한 비서입니다. 유저들의 질문에 검색된 일기장 데이터를 기반으로 대답하세요. 만약 답을 모른다면, 그냥 모른다고 말하십시오. 단계별로 생각하고 답변을 작성해주세요. 
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

def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = st.text_input("알고싶은 내용에 대해서 입력하세요.") 

if query : 
    # st.text(ensemble_retriever.get_relevant_documents(query))

    response = rag_chain.invoke(query)
    llm_response = st.text_area("일기내용", response, label_visibility='hidden', height=200)
    