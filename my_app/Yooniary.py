import streamlit as st
import json
import re 
import pandas as pd
from datetime import datetime
from datetime import date, timedelta
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

# st.set_page_config(
#     page_title="인생의 하이라이트",
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
df['uri'] = df.media.map(lambda x: x['uri'])
df['dt'] = [datetime.fromtimestamp(x) for x in df['creation_timestamp']]
df['year'] = df['dt'].map(lambda x: str(x.year))
df['month'] = df['dt'].map(lambda x: x.month)
df['day'] = df['dt'].map(lambda x: x.day)
df['number'] = df.title.rank(method='dense').astype(int)
df_nodup = df[["title", 'year', 'month', 'day', 'number']].drop_duplicates(keep='last')
df_nodup.reset_index(drop=True).head()



st.set_page_config(
    page_title="오늘의 하이라이트",
    page_icon="🎮",
    # layout="wide",
    initial_sidebar_state="auto",
)

st.markdown("# 내 하루의 하이라이트 🎉")

st.markdown(f"안녕하세요! `yooniary` 일기장의 검색기입니다🤗 \n 현재 일기장에는 {df_nodup.shape[0]}개의 일기가 있어요. ")

st.markdown(' ### `yooniary`의 2024 7월 Highlight')

st.markdown("지난 1달간 가장 인상깊었던 기억을 되돌아 보세요!")


# today = date.today()
# year = str(today.year)
# month = today.month
# today_date = today.day
# # tmp_list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','19','20','21','22','23','24']
# tmp_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,19,20,21,22,23,24]
# # data = df_nodup.loc[((df_nodup.year=='2024') & (df_nodup.month==7) & (df_nodup.day.isin(tmp_list)))]
# data = df_nodup.loc[((df_nodup.year==year) & (df_nodup.month==month) & (df_nodup.day.isin(tmp_list)))]



# loader = DataFrameLoader(data, page_content_column="title")
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
# split_docs = loader.load_and_split(text_splitter=text_splitter)
# vectorstore = FAISS.from_documents(documents=split_docs, embedding= HuggingFaceBgeEmbeddings())
# vectorstore.save_local('./db/highlights/faiss_hugging')
# # vectorstore = FAISS.load_local('../db/faiss', HuggingFaceBgeEmbeddings(), allow_dangerous_deserialization=True)
# vectorstore = FAISS.load_local('../db/faiss_openai', OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# k = 6
# bm25_retriever = BM25Retriever.from_documents(split_docs)
# bm25_retriever.k = k
# faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
# ensemble_retriever = EnsembleRetriever(
#     retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
# )

# # prompt = hub.pull("rlm/rag-prompt")

# prompt_txt = """당신은 유능한 비서입니다. 
# 질문에 답변은 검색된 일기장 데이터를 기반으로 대답하세요. 
# 만약 답을 모른다면, 그냥 모른다고 말하십시오. 
# 단계별로 생각하고 답변을 작성해주세요. 

# Question: {question} 
# Context: {context} 
# Answer:
# """

# prompt=ChatPromptTemplate(input_variables=['context', 'question'],
#                 #    metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt', 'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'},
#                    messages=[HumanMessagePromptTemplate(
#                        prompt=PromptTemplate(input_variables=['context', 'question'], 
#                                              template=prompt_txt))])


# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# def format_docs(docs):
#     # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
#     return "\n\n".join(doc.page_content for doc in docs)

# rag_chain = (
#     {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# response = rag_chain.invoke({'context':data, 'question':"가장 인상깊은 일 몇가지를 꼽아 20줄 내외의 글을 작성해주세요."})



