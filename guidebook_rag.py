import os
import json
from operator import itemgetter
from dotenv import load_dotenv

# LangChain Core
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document

# Models & Stores
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereRerank

# Retrievers
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from DebugBM25Retriever import DebugBM25Retriever
from DebugPineconeRetriever import DebugPineconeRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

CONFIG = {
    "index_name": "aion2-guide-rag",
    "embedding_model": "text-embedding-3-large",
    "llm_model": "gpt-4o-mini",
    "rerank_model": "rerank-multilingual-v3.0",
    "local_data_path": "data/guide_docs.json" # 크롤링한 데이터 경로
}

def load_bm25_documents():
    """로컬 JSON 파일을 읽어 BM25용 Document 리스트를 반환합니다."""
    path = CONFIG["local_data_path"]
    
    if not os.path.exists(path):
        print(f"⚠️ 경고: '{path}' 파일이 없습니다. BM25 검색을 건너뜁니다.")
        return []

    print(f"📂 BM25 인덱싱을 위해 '{path}' 로딩 중...")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # JSON -> Document 객체 변환
        docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
        
        # BM25도 청크 단위로 검색해야 정확하므로 분할 수행
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)
        
        print(f"✅ BM25 인덱스 생성 완료 (총 {len(split_docs)}개 청크)")
        return split_docs
        
    except Exception as e:
        print(f"❌ BM25 데이터 로딩 실패: {e}")
        return []

def get_rag_chain():
    """
    Hybrid Search (Pinecone + BM25) -> Rerank -> LLM 체인 생성
    """
    load_dotenv()

    # 1. Pinecone Retriever 설정 (Vector Search)
    embeddings = OpenAIEmbeddings(model=CONFIG["embedding_model"])
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=CONFIG["index_name"],
        embedding=embeddings
    )
    # Reranker에게 보낼 후보군 (Vector)
    # pinecone_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    pinecone_retriever = DebugPineconeRetriever(
        vectorstore=vector_store, 
        search_kwargs={"k": 5}
    )
    
    # 2. BM25 Retriever 설정 (Keyword Search) [추가됨]
    bm25_docs = load_bm25_documents()

    base_retriever = pinecone_retriever # 기본값은 Pinecone 단독
    
    if bm25_docs:
        bm25_retriever = DebugBM25Retriever.from_documents(bm25_docs)
        bm25_retriever.k = 5 # Reranker에게 보낼 후보군 (Keyword)

        # 3. Ensemble (Hybrid) 설정 [추가됨]
        # weights=[0.5, 0.5]: 벡터와 키워드 검색 결과를 반반씩 반영
        print("🔗 Hybrid Search(Pinecone + BM25) 모드로 동작합니다.")
        base_retriever = EnsembleRetriever(
            retrievers=[pinecone_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
    else:
        print("⚠️ Hybrid Search 실패 -> Pinecone 단독 모드로 동작합니다.")

    # 4. Cohere Rerank 설정 (재정렬)
    # compressor = CohereRerank(
    #     model=CONFIG["rerank_model"],
    #     cohere_api_key=os.getenv("COHERE_API_KEY"),
    #     top_n=5 # 최종적으로 LLM에게 줄 3개만 선별
    # )
    
    # Hybrid Retriever의 결과를 Cohere가 재정렬
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor,
    #     base_retriever=base_retriever
    # )

    # 5. 프롬프트 템플릿
    template = """
    당신은 AION2 게임 가이드 AI입니다.
    아래의 [이전 대화 내용]과 [참고 문서]를 바탕으로 질문에 답변해주세요.
    
    1. 문서에 없는 내용은 지어내지 말고 모른다고 하세요.
    2. 이전 대화의 맥락을 고려하여 답변하세요.
    3. 아이템, 스킬 명칭은 문서에 있는 그대로 정확히 사용하세요.
    4. 사용자가 특정 직업(예: 수호성, 호법성 등)에 대해 물었다면, 반드시 해당 직업의 문서만 참조하세요.
    5. 문서의 [metadata]나 제목을 확인하여 질문한 직업과 일치하는지 확인하세요.
    
    [이전 대화 내용]
    {chat_history}

    [참고 문서]
    {context}

    질문: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model=CONFIG["llm_model"], temperature=0)

    # 6. 문서 포맷팅 헬퍼
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 7. Chain 조립
    rag_chain = (
        RunnableParallel({
            "context": itemgetter("question") | base_retriever, 
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history") 
        })
        .assign(answer=(
            RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
            | prompt 
            | model 
            | StrOutputParser()
        ))
    )
    
    return rag_chain

# 테스트 실행 코드 (생략 가능)
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
    test_history = ""
    test_query = "속사는 어느 클래스의 스킬이야?"

    print(f"\n📝 [입력 데이터]")
    print(f"   - 이전 대화: {test_history.strip().replace(chr(10), ' ')}...") # 줄바꿈 제거 후 출력
    print(f"   - 현재 질문: {test_query}")
    print("\n⏳ 답변 생성 중... (Pinecone 검색 + BM25 + GPT 추론)")

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
        
        # print("📚 [참고 문서 (Cohere Rerank 결과)]")
        # for i, doc in enumerate(result['context']):
        #     score = doc.metadata.get('relevance_score', 0)
        #     title = doc.metadata.get('title', '제목 없음')
        #     # print(doc.page_content)
        #     # score가 높을수록 연관성이 높은 문서입니다.
        #     print(f"   [{i+1}] 신뢰도: {score:.4f} | 제목: {title}")

    except Exception as e:
        print(f"❌ 실행 중 에러 발생: {e}")