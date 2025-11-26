import time
import sys
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# 환경 변수 로드 (.env 파일 읽기)
load_dotenv()

# 설정값
INDEX_NAME = "aion2-guide-rag"  # Pinecone 콘솔에서 미리 만들어둔 인덱스 이름과 일치해야 합니다.

def collect_nc_guide_urls(start_id, end_id):
    print(f"🚀 수집 시작: CategoryId {start_id} ~ {end_id}")

    options = webdriver.ChromeOptions()
    # options.add_argument('--headless') # 잘 되는거 확인하시면 주석 해제하세요 (속도 향상)
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    all_urls = set()

    try:
        for cat_id in range(start_id, end_id + 1):
            url = f"https://aion2.plaync.com/ko-kr/guidebook/list#categoryId={cat_id}"
            print(f"\n🔄 [Category {cat_id}] 이동 중...")
            
            driver.get(url)

            try:
                # 1. 명확한 클래스명이 로딩될 때까지 대기
                wait = WebDriverWait(driver, 10)
                target_class = "ncgbg-guide-depth-2-guide-item-link"
                
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, target_class)))
                time.sleep(1) # 렌더링 안정화

                # 2. 해당 클래스를 가진 모든 요소를 찾음
                elements = driver.find_elements(By.CLASS_NAME, target_class)
                
                count = 0
                for el in elements:
                    # Selenium이 href를 가져올 때 자동으로 http://... 전체 경로로 변환해줍니다.
                    full_url = el.get_attribute("href")
                    
                    # 가끔 빈 링크나 중복이 있을 수 있으니 필터링
                    if full_url and "view?title=" in full_url:
                        all_urls.add(full_url)
                        count += 1
                
                print(f"   ✅ {count}개 수집 성공")

            except Exception:
                print(f"   ⚠️ [Category {cat_id}] 해당 카테고리에 글이 없거나 로딩 실패")
                continue

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
    
    finally:
        driver.quit()

    return list(all_urls)

# --- 실행 ---

    # 4234 ~ 4244 전체 수집
final_urls = collect_nc_guide_urls(4234, 4244)

print(f"\n🎉 최종 수집 결과: 총 {len(final_urls)}개")
for url in final_urls:
    print(url)

def load_split_docs(final_urls):
    print("1. 데이터 로딩 중... (Playwright 사용)")
    
    # PlaywrightURLLoader 설정
    loader = PlaywrightURLLoader(
        urls=final_urls,
        remove_selectors=["nav", "header", "footer", ".cookie-banner"], # 불필요한 UI 제거
        continue_on_failure=True
    )

    try:
        # 실제 브라우저를 띄워 렌더링 후 로딩
        docs = loader.load()
        
        print(f"-> 로드된 문서 개수: {len(docs)}")
        if docs:
            # 첫 번째 문서 내용 앞부분만 출력해서 제대로 긁어왔는지 확인
            print(f"\n[첫 번째 문서 샘플]\n{docs[0].page_content[:300]}...\n")
            
        print("2. 텍스트 분할 중...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        splits = text_splitter.split_documents(docs)
        print(f"-> 총 분할된 청크 수: {len(splits)}")
        for split in splits:
            print(split.page_content)
        
        # # TODO: 여기서 Pinecone 저장 로직 수행
        # # 3. 임베딩 및 Pinecone 저장
        # print(f"3. Pinecone('{INDEX_NAME}')에 데이터 저장 시작...")
        
        # embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # 최신 모델 사용 권장

        # # Pinecone에 문서 업로드 (Batch로 자동 처리됨)
        # vectorstore = PineconeVectorStore.from_documents(
        #     documents=splits,
        #     embedding=embeddings,
        #     index_name=INDEX_NAME
        # )
        
        print("🎉 모든 작업이 성공적으로 완료되었습니다!")

    except Exception as e:
        print(f"에러 발생: {e}")

load_split_docs(final_urls)