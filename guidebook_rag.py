import os
import sys
from operator import itemgetter
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel



CONFIG = {
    "index_name": "aion2-guide-rag",
    "embedding_model": "text-embedding-3-small",
    "llm_model": "gpt-4o-mini",
    "rerank_model": "rerank-multilingual-v3.0"
}

def get_rag_chain():
    """
    RAG 체인을 생성하여 반환합니다.
    이 체인은 invoke 시 {'question': '...', 'chat_history': '...'} 형태의 입력을 받습니다.
    """
    # 환경 변수 로드
    load_dotenv()

    # 2. Retriever & Reranker 설정
    embeddings = OpenAIEmbeddings(model=CONFIG["embedding_model"])
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=CONFIG["index_name"],
        embedding=embeddings
    )
    
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
    
    compressor = CohereRerank(
        model=CONFIG["rerank_model"],
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        top_n=3
    )
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    # 3. 프롬프트 템플릿 (Chat History 추가)
    # 문맥 유지를 위해 '이전 대화 내용' 섹션을 추가했습니다.
    template = """
    당신은 AION2 게임 가이드 AI입니다.
    아래의 [이전 대화 내용]과 [참고 문서]를 바탕으로 질문에 답변해주세요.
    
    1. 문서에 없는 내용은 지어내지 말고 모른다고 하세요.
    2. 이전 대화의 맥락을 고려하여 답변하세요.
    
    [이전 대화 내용]
    {chat_history}

    [참고 문서]
    {context}

    질문: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model=CONFIG["llm_model"], temperature=0)

    # 4. 문서 포맷팅 헬퍼
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 5. Chain 조립 (itemgetter 사용)
    # 입력이 Dictionary로 들어오므로 itemgetter로 각 요소를 뽑아냅니다.
    rag_chain = (
        RunnableParallel({
            "context": itemgetter("question") | compression_retriever, # 질문으로 문서 검색
            "question": itemgetter("question"),                        # 질문 그대로 통과
            "chat_history": itemgetter("chat_history")                 # 대화 이력 그대로 통과
        })
        .assign(answer=(
            RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
            | prompt 
            | model 
            | StrOutputParser()
        ))
    )
    
    return rag_chain

# === Main ===
# === 테스트용 Main 함수 ===
if __name__ == "__main__":
    import sys
    
    print("🧪 [TEST MODE] guidebook_rag.py 독립 실행 테스트")
    print("-" * 60)

    # 1. 체인 생성
    try:
        chain = get_rag_chain()
        if chain is None:
            print("❌ 체인 생성 실패: API Key를 확인하세요.")
            sys.exit(1)
        print("✅ RAG 체인 생성 완료")
    except Exception as e:
        print(f"❌ 초기화 중 에러 발생: {e}")
        sys.exit(1)

    # 2. 테스트 시나리오 설정
    # 상황: 사용자가 이전에 '강화'에 대해 물어봤고, 이어서 '마석'에 대해 묻는 상황
    test_history = """
    User: 아이템 강화 시스템에 대해 알려줘.
    AI: 아이템 강화는 강화석, 마석, 재조율 주문서 등을 통해 장비의 능력치를 향상시키는 시스템입니다.
    """
    test_query = "그럼 마석 각인은 구체적으로 어떻게 하는거야?"

    print(f"\n📝 [입력 데이터]")
    print(f"   - 이전 대화: {test_history.strip().replace(chr(10), ' ')}...") # 줄바꿈 제거 후 출력
    print(f"   - 현재 질문: {test_query}")
    print("\n⏳ 답변 생성 중... (Pinecone 검색 + Cohere Rerank + GPT 추론)")

    # 3. 체인 실행
    try:
        result = chain.invoke({
            "question": test_query,
            "chat_history": test_history
        })

        # 4. 결과 출력
        print("-" * 60)
        print(f"🤖 [AI 답변]\n{result['answer']}")
        print("-" * 60)
        
        print("📚 [참고 문서 (Cohere Rerank 결과)]")
        for i, doc in enumerate(result['context']):
            score = doc.metadata.get('relevance_score', 0)
            title = doc.metadata.get('title', '제목 없음')
            # score가 높을수록 연관성이 높은 문서입니다.
            print(f"   [{i+1}] 신뢰도: {score:.4f} | 제목: {title}")

    except Exception as e:
        print(f"❌ 실행 중 에러 발생: {e}")