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
df['uri'] = df.media.map(lambda x: x['uri'])
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

prompt_txt = """당신은 유능한 비서입니다. 
context에서 일기장 내용이 주어지면 그 내용을 기반으로 질문에 친절한 말투로 답변하세요.
질문에 대한 답변만 하도록 하고, 만약 답을 모른다면, 그냥 모른다고 말하십시오. 
단계별로 생각하고 답변을 작성해주세요. 

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


st.markdown('# GPT에게 물어보세요 🔍')
st.markdown('일기를 기반으로 어느 날 무슨 일이 있었는지, 요약해드릴게요!')



query = st.text_input("알고싶은 내용에 대해서 입력하세요.") 


if query : 
    response = rag_chain.invoke(query)
    llm_response = st.text_area("일기내용", response, label_visibility='hidden', height=200)

    docs = ensemble_retriever.get_relevant_documents(query)

    docs_index = []
    for i in range(len(docs)):
        docs_index_tmp = df.title.map(lambda x: docs[i].page_content in x)
        docs_df=df[docs_index_tmp]
        docs_index.extend(docs_df.index.to_list())

    data = df.loc[docs_index]

    manuscripts = st.multiselect("일기 선택", data.title.unique(), data.title.unique())
    
    
    paths = data.loc[data.title.isin(manuscripts),'uri']
    paths = [s for s in paths if "jpg" in s] 
        
    

    n = st.number_input("Select Grid Width", 5, 10, 6)

    groups = []
    for i in range(0, len(paths), n):
        groups.append(paths[i:i+n])


    for group in groups:
        cols = st.columns(n)
        for i, image in enumerate(group):
            
            cols[i].image(f"../instagram_diary/{image}")

    txt = data.loc[data.title.isin(manuscripts),'title'].reset_index(drop=True)[0]
    diary_txt = st.text_area("일기내용", txt, label_visibility='hidden', height=400)

        