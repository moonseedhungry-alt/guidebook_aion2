from langchain_community.retrievers import BM25Retriever

# 🛠️ 디버깅을 위해 기존 클래스를 상속받아 재정의 (Java의 Extends & Override와 동일)
class DebugBM25Retriever(BM25Retriever):
    def _get_relevant_documents(self, query: str, *, run_manager=None):
        # 1. 부모 클래스의 원래 검색 기능 실행
        results = super()._get_relevant_documents(query, run_manager=run_manager)
        
        # 2. 결과 가로채서 로그 출력
        print(f"\n🕵️ [BM25 Debug] 검색어: '{query}'")
        print(f"   ㄴ 발견된 문서 수: {len(results)}개")
        for i, doc in enumerate(results[:3]): # 상위 3개만 미리보기
            title = doc.metadata.get('title', '제목없음')
            print(f"      [{i+1}] {title} (유사도 점수 등은 BM25 내부 계산)")
            print(doc.page_content)
            
        return results