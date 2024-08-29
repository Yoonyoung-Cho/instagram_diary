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
from langchain_milvus import Milvus 
from langchain_core.runnables import ConfigurableField

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

st.markdown("# 내 인생의 하이라이트 🎉")

st.markdown(f"안녕하세요! `yooniary` 일기장의 검색기입니다🤗 \n 현재 일기장에는 {df_nodup.shape[0]}개의 일기가 있어요. ")

year_num = datetime.today().year
month_num = datetime.today().month

st.markdown(f' ### `yooniary`의 {year_num}년 {month_num}월 Highlight')

st.markdown("최근 1달간 가장 인상깊었던 기억을 되돌아 보세요!")

query = "가장 즐거웠던 일들을 요약해주세요."
# st.text(query)

# vectorstore = Milvus(
#     OpenAIEmbeddings(),
#     connection_args={"uri": "/Users/kakaogames/Documents/GAI/instagram_diary/db/milvus/insta_milvus.db"},
#     collection_name="mulvus_insta_v01",
# )
vectorstore = Milvus(
    OpenAIEmbeddings(),
    connection_args={"uri": "/Users/kakaogames/Documents/GAI/instagram_diary/db/milvus/insta_milvus2.db"},
    collection_name="mulvus_insta_v02",
)

query = '무슨 일이 있었나 사건별로 bullet point로 친절한 말투로 자세하게 설명해주세요.'

retriever = vectorstore.as_retriever().configurable_fields(
    search_kwargs=ConfigurableField(
        id="retriever_search_kwargs",
    )
)

prompt_txt = """당신은 유능한 비서입니다. 
context에서 일기장 내용이 주어지면 그 내용을 기반으로 질문에 친절한 말투로 답변하세요.
질문에 대한 답변만 하도록 하고, 만약 답을 모른다면, 그냥 모른다고 말하십시오. 
맞춤법에 유의하세요.
단계별로 생각하고 답변을 작성해주세요. 

Question: {question} 
Context: 나는 공덕으로 퇴근햇다 근데 일퇴리서 그런지 너무 …너무 지옥철이엇어서 당황. 사람들 다 밀고 난리도 어니엇다. 약간 피곤한채로 공덕도척햇는데 친구랑 무ㅜ먹지하다가 그때 못먹은 떡갈비 가 아니라 불고기 가억해내고 불고기 먹으러람 냠냠 정말 맛잇엇다 낙지가 처음에 잘못나와서 좀 당뢍해ㅛ지만 맛잇엇어 매운 불고기도 맛잇엇디 근데 꽤매웟음 그러고 마늘이ㅋㅋㅋㅋ너무매워 미쳣다이렇게 마운 마늘 진짜 너무 매워….말되나 수준으로 매유ㅗㅅ다. 친구랑 울면서 먹다가 나와서 2차로 떡볶이 순댜머금. 먹우면서 회사 얘기쫌 래쥬고~그러고 한강 산책하러고 핫는데 막상 도착하니 하기싫고 피곤해서 집에 갓다. 알고보니 나 오늘 갑상선약 안머금 정말 귀신같은 몸상태야 갑상선은 중요한거구나 싶음 아 병원도 다녀왔다 이거 실비 신청하면 안되는거였나? 여튼 아파서 슬프다 왜 안낫는거야… \n\n 어에에엥 나 일기썻는데 오늘 어ㅐ 없어졋지. 안썻나? 쏘조워 연화랑 점심먹운날 아침은 유즈코쇼랑 치주김밥 존맛 점심 맛잇엇고 재밋엇구 티탐은 조금 힘들었고 집가고싶엇고~식단 기록은 망해써 걸국 안함. 
Answer: ### 📌 공덕에서의 하루 \n\n 공덕으로 지옥철을 타고 퇴근을 했던 날입니다. 친구랑 아주 매운 불고기를 먹었고 함께 먹은 마늘도 정말 매웠습니다. 너무 매워서 울면서 먹다가 떡볶이와 순대로 2차를 갔고, 산책을 가려고 했으나 피곤해서 그냥 집으로 갔습니다. ### 📌 쏘조워연화랑 점심 먹은 날 \n\n 아침은 유주코쇼와 치즈김밥을 먹었고 점심은 쏘조워연화와 맛있게 먹었습니다. 식단기록은 안했습니다.

Question: {question} 
Context: {context} 
Answer:
"""

prompt=ChatPromptTemplate(input_variables=['context', 'question'],
                   messages=[HumanMessagePromptTemplate(
                       prompt=PromptTemplate(input_variables=['context', 'question'], 
                                             template=prompt_txt))])


llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

expr_txt = f"month == {month_num} && year == '{year_num}'"

llm_response = rag_chain.with_config(
    configurable={
        "retriever_search_kwargs": dict(
            expr=expr_txt,
        )
    }
).invoke(query)

st.markdown(llm_response)




