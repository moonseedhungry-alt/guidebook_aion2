from langchain_core.vectorstores import VectorStoreRetriever

class DebugPineconeRetriever(VectorStoreRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager=None):
        # 1. 부모 클래스의 원래 검색 기능 실행 (Pinecone 검색)
        results = super()._get_relevant_documents(query, run_manager=run_manager)
        
        # 2. 결과 로그 출력
        print(f"\n🌲 [Pinecone Debug] 검색어: '{query}'")
        print(f"   ㄴ 발견된 문서 수: {len(results)}개")
        for i, doc in enumerate(results):
            title = doc.metadata.get('title', '제목없음')
            # Pinecone은 기본적으로 score를 바로 주지 않지만, 필요하면 메타데이터 확인 가능
            print(f"      [{i+1}] {title}")
            print(doc.page_content)
            
        return results