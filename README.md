## GPT를 이용한 인생의 하이라이트 보기⭐️
* `2024.07.25`에 GAI 스터디에서 발표한 인스타그램 일기장 요약
* `2024.08.22`에 GAI 스터디에서 2차 발표


## 매일 일기 쓰는 나 제법 멋져요 
* 인스타그램에 2016년부터 쓰던 일기장
* 쓰기 시작한지 8년이 되어 제법 글이 쌓였으나 원하는 날짜, 원하는 내용을 찾아볼 수가 없다 😭🥹
* 아카이빙을 잘하고 싶다는 욕구
* 다른 분들도 개인 데이터를 활용했는데 나도 일기 데이터를 잘 활용해 볼 수 있는 방법이 없을까?


## 구현하고 싶은 기능 
- [x] 내 인생의 하이라이트 뽑아 보기
- [x] 키워드 검색 기능 
- [x] 특정 사건 또는 일기 요약 기능
- [ ] Q&A 기능 ( 나에 대해서 얼마나 잘 알까? )

## 구현 방법
* 데이터 : 인스타그램 '내 활동'에서 다운로드 가능
* RAG - Ensemble Retriever
* Milvus
* LangChain
* LangSmith
* Streamlit
* GPT 3.5 turbo 모델 사용


## Demo 

### 내 인생의 하이라이트 뽑아 보기
![그림1](https://github.com/user-attachments/assets/674aa7fa-b8fb-4871-a129-312a084af049)
* 주어진 기간의 일기 중 가장 즐거운 일기를 뽑아 요약 
* Milvus의 metadata를 이용


### 검색
![image](https://github.com/user-attachments/assets/46e32d49-1388-44b0-ab31-bd55ad46318d)
* 키워드 검색 가능
* 날짜 설정 가능
* 키워드가 포함된 일기 조회 가능
* LLM으로 일기마다 제목을 붙였음

### 일기 요약 
![image](https://github.com/user-attachments/assets/2ef9be68-d7fe-4c7b-8888-cee4615ac238)
* 즐거웠던 사건에 대해 물어보면 일기장 데이터를 기반으로 질문에 답변.


그 외 아직 개발중 🤗
