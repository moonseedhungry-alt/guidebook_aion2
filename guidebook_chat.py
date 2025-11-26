import streamlit as st
from guidebook_rag import get_rag_chain # 분리한 파일 import

# 페이지 설정
st.set_page_config(page_title="AION2 가이드 봇", page_icon="🛡️")

# === Helper Function: 채팅 이력 포맷팅 ===
def format_chat_history(messages):
    """
    Streamlit 세션의 메시지 리스트를 LLM이 이해하기 쉬운 텍스트로 변환합니다.
    최근 5턴(10개 메시지)만 유지하여 토큰을 절약합니다.
    """
    formatted_history = []
    # 시스템 메시지 제외하고 최근 대화만 가져오기
    recent_messages = messages[-10:] 
    
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "AI"
        content = msg["content"]
        formatted_history.append(f"{role}: {content}")
        
    return "\n".join(formatted_history)

# === Main UI ===
st.title("🛡️ AION2 게임 가이드 (Context)")

# 1. 체인 로딩 (캐싱)
@st.cache_resource
def load_chain():
    return get_rag_chain()

chain = load_chain()

# 2. 세션 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. 대화 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 참고 문서"):
                for src in message["sources"]:
                    st.markdown(f"- **{src['title']}** ({src['score']:.2f})")

# 4. 입력 처리
if query := st.chat_input("질문을 입력하세요..."):
    # 사용자 메시지 표시 및 저장
    st.chat_message("user").write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    if chain:
        with st.chat_message("assistant"):
            container = st.empty()
            container.markdown("⏳ 생각 중...")
            
            # [핵심] 현재 채팅 이력을 문자열로 변환
            chat_history_str = format_chat_history(st.session_state.messages[:-1])
            
            try:
                # [핵심] 질문과 히스토리를 함께 전달
                result = chain.invoke({
                    "question": query,
                    "chat_history": chat_history_str
                })
                
                answer = result['answer']
                sources = result['context']
                
                container.markdown(answer)
                
                # 출처 UI 생성
                source_data = []
                with st.expander("📚 참고 문서 확인"):
                    for doc in sources:
                        score = doc.metadata.get('relevance_score', 0)
                        title = doc.metadata.get('title', '제목 없음')
                        source_data.append({"title": title, "score": score})
                        st.markdown(f"**[{title}]** ({score:.2f})")
                        st.caption(doc.page_content[:100] + "...")

                # AI 응답 저장
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": source_data
                })
                
            except Exception as e:
                container.error(f"에러 발생: {e}")