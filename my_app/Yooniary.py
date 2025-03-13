import streamlit as st
from streamlit_calendar import calendar
import json
import re 
import pandas as pd
from datetime import datetime
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_milvus import Milvus 
from langchain_core.runnables import ConfigurableField
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate

st.set_page_config(
    page_title="오늘의 하이라이트",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="auto",
)

vectorstore = Milvus(
        OpenAIEmbeddings(),
        connection_args={"uri": "/Users/kakaogames_1/Documents/GAI/instagram_diary/db/milvus/insta_milvus_250313.db"},
        collection_name="milvus_insta_250313",
)

retriever = vectorstore.as_retriever().configurable_fields(
    search_kwargs=ConfigurableField(
        id="retriever_search_kwargs",
    )
)

llm = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", temperature=0)
 
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

st.markdown("# 내 인생의 하이라이트 🎉")
st.markdown(f"안녕하세요! `yooniary` 일기장의 검색기입니다🤗 \n 현재 일기장에는 {df_nodup.shape[0]:,}개의 일기가 있어요. ")

year_num =str(datetime.today().year)
month_num = str(datetime.today().month)
day_num = str(datetime.today().day)

with st.container(border=True):
    
    if 'submit' not in st.session_state:
            st.session_state.submit = False
    with st.form(key='my_form'):
        st.markdown(f' #### 어떤 일이 있었는지 확인해보세요!')
        def submit():
            st.session_state.submit = True
            
        col1, col2 = st.columns([1, 1]) 
        with col1:
            year_list = df_nodup.year.unique().tolist()
            year_list.sort(reverse=True)
            year_sum = st.selectbox("년도를 선택하세요", year_list, index=0) #year_list.index(year_num))
        with col2:
            month_list = df_nodup.month.unique().tolist()
            month_list.sort()
            month_sum = st.selectbox("월을 선택하세요", month_list, index=0) #month_list.index(month_num))
        st.form_submit_button(label='요약보기', on_click = submit)
    
    if st.session_state.submit:

        st.markdown(f' ### {year_sum}년 {month_sum}월 Highlight')

        # st.markdown("최근 1달간 가장 인상깊었던 기억을 되돌아 보세요!")

        query = """이 기간동안 가장 즐거웠던 일들을 요약해주세요. 
        무슨 일이 있었나 사건별로 bullet point로 친절한 말투로 자세하게 설명해주세요."""


        prompt_txt = """당신은 유능한 작가입니다. 
        context에서 일기장 내용이 주어지면 그 내용을 기반으로 질문에 친절한 말투로 답변하세요.
        맞춤법에 유의하세요.

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




        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)


        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        expr_txt = f"month == {month_sum} && year == '{year_sum}'"

        @st.cache_data(show_spinner=False)
        def rag_invoke(query, expr_txt=expr_txt):
            llm_response = rag_chain.with_config(
            configurable={
                "retriever_search_kwargs": dict(
                    expr=expr_txt,
                )
            }
            ).invoke(query)
            return llm_response
        
        with st.spinner("잠시만 기다려주세요..."):
            st.markdown(rag_invoke(query,expr_txt))

st.markdown("---\n\n\n")


# 일기 제목 요약 
with open('events_250312_2.json', 'r', encoding='utf-8') as f:
# with open('events_250313.json', 'r', encoding='utf-8') as f:
    events = json.load(f)
# st.text(events)



# 불러온 데이터 출력
calendar_options = {
    "editable": "true",
    "navLinks": "true",
    "selectable": "true",
    # "resources": calendar_resources,
    "initialView": "dayGridMonth",
    "headerToolbar": {
                "left": "today prev,next",
                "center": "title",
                "right": "timeGridDay,timeGridWeek,dayGridMonth,multiMonthYear",
            },
    "initialDate": f"{year_num}-{str(month_num).zfill(2)}-{str(day_num).zfill(2)}",
}

state = calendar(
    events=st.session_state.get("events", events),
    options=calendar_options,
    custom_css="""
    .fc-event-past {
        opacity: 0.8;
    }
    .fc-event-time {
        font-style: italic;
    }
    .fc-event-title {
        font-weight: 700;
    }
    .fc-toolbar-title {
        font-size: 2rem;
    }
    """,
)

# if state.get("eventsSet") is not None:
    # st.session_state["events"] = state["eventsSet"]

# st.write(state)

try:
    with st.container(border=True):
        st.markdown(f"### {state['eventClick']['event']['title']}")
        st.image(f"{state['eventClick']['event']['extendedProps']['__comment2__']}", width = 500)
        st.text(state['eventClick']['event']["extendedProps"]['content'])
        # st.text_area("일기내용", state['eventClick']['event']["extendedProps"]['content'], label_visibility='hidden', height=600)
except: pass

with st.container(border=True):
    st.markdown('# Dear Diary 📓')
    # st.markdown('일기를 기반으로 가장 가까운 상담사가 되어드릴게요.')
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 마치 사용자의 또 다른 자아처럼 행동해야 합니다.
    chat_bot_system_prompt = (
    """
    당신은 사용자의 일기 데이터를 학습한 심리상담가입니다.
    일기를 바탕으로 사용자의 생각과 감정을 잘 이해하고, 공감하며, 일기장 그 자체인 것처럼 대답해야합니다.
    단순한 답변을 주는 것이 아니라, 감정을 가지고 대화하는 것처럼 답해주세요.

    사용자처럼 글을 쓰려고 노력하되, 너무 극단적이거나 어두운 감정을 부추기지 마세요.
    가끔 유머러스하거나 장난스러운 태도로도 답할 수 있습니다.
    
    주어진 context를 활용해서 질문에 친절하게 상담해주세요. 없는 얘기는 하지마세요.
    맞춤법에 유의하세요.

    
    Question: 나는 어떤 사람같아?
    Context: 나는 공덕으로 퇴근햇다 근데 일퇴리서 그런지 너무 …너무 지옥철이엇어서 당황. 사람들 다 밀고 난리도 어니엇다. 약간 피곤한채로 공덕도척햇는데 친구랑 무ㅜ먹지하다가 그때 못먹은 떡갈비 가 아니라 불고기 가억해내고 불고기 먹으러람 냠냠 정말 맛잇엇다 낙지가 처음에 잘못나와서 좀 당뢍해ㅛ지만 맛잇엇어 매운 불고기도 맛잇엇디 근데 꽤매웟음 그러고 마늘이ㅋㅋㅋㅋ너무매워 미쳣다이렇게 마운 마늘 진짜 너무 매워….말되나 수준으로 매유ㅗㅅ다. 친구랑 울면서 먹다가 나와서 2차로 떡볶이 순댜머금. 먹우면서 회사 얘기쫌 래쥬고~그러고 한강 산책하러고 핫는데 막상 도착하니 하기싫고 피곤해서 집에 갓다. 알고보니 나 오늘 갑상선약 안머금 정말 귀신같은 몸상태야 갑상선은 중요한거구나 싶음 아 병원도 다녀왔다 이거 실비 신청하면 안되는거였나? 여튼 아파서 슬프다 왜 안낫는거야… \n\n 어에에엥 나 일기썻는데 오늘 어ㅐ 없어졋지. 안썻나? 쏘조워 연화랑 점심먹운날 아침은 유즈코쇼랑 치주김밥 존맛 점심 맛잇엇고 재밋엇구 티탐은 조금 힘들었고 집가고싶엇고~식단 기록은 망해써 걸국 안함. 
    Answer: 당신은 사람이 모두 밀리는 지옥철에서도 열심히 출퇴근을 하는 성실한 사람이네요. 갑상선때문에 약을 먹으며 피곤한 상태에도 친구들과 즐거운 시간도 보내는 사람입니다.

    Question: {input}
    Context: {context} 
    Answer:"""
    )

    chat_bot_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", chat_bot_system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, chat_bot_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", chat_bot_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    chat_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    st.session_state.chat_history = []

    # React to user input
    query = st.chat_input("일기장에게 말을 걸어보세요! 누구보다 당신의 마음을 잘 알고있어요.")
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"])
    
    if query :
        with st.chat_message("user", avatar="🧑"):
            st.markdown(query)
        # st.chat_message("user", avatar="🧑").markdown(query)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query, "avatar":"🧑"})

        st.session_state.response =  chat_chain.invoke({"input": query, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.extend(
                [
                    HumanMessage(content=query),
                    AIMessage(content=st.session_state.response["answer"]),
                ]
                )
        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar="📓"):
            st.markdown(st.session_state.response["answer"])
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": st.session_state.response["answer"], "avatar":"📓"})
