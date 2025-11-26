import os
import time
import sys
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# Selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# LangChain 관련
from langchain_core.documents import Document  # Document 객체 직접 생성
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

INDEX_NAME = "aion2-guide-rag"

# API Key 검증
if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
    print("❌ Error: .env 파일 확인 필요")
    sys.exit(1)

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

def process_and_save_docs(urls):
    """ Selenium으로 상세 페이지 로딩 -> Title, Desc, Body 파싱 -> Pinecone 저장 """
    if not urls:
        print("⚠️ URL이 없습니다.")
        return

    print(f"\n1. 데이터 로딩 및 구조화 중... (대상: {len(urls)}개)")
    
    # ... (Selenium 설정 부분은 기존과 동일) ...
    options = webdriver.ChromeOptions()
    options.add_argument('--headless') 
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 ...") # 기존 User-Agent 그대로 사용
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    valid_docs = []

    try:
        for url in urls:
            print(f"   ➡️ 접속: {url}")
            driver.get(url)
            
            try:
                wait = WebDriverWait(driver, 10)
                # 본문(article)이 뜰 때까지 대기
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "ncgbt-article")))
                time.sleep(1) 
                
                soup = BeautifulSoup(driver.page_source, "html.parser")
                
                # 1. 제목 수집 (ncgbt-cover-title)
                title_tag = soup.select_one(".ncgbt-cover-title")
                title = title_tag.get_text(strip=True) if title_tag else "제목 없음"

                # 2. 설명 수집 (ncgbt-cover-desc)
                desc_tag = soup.select_one(".ncgbt-cover-desc")
                desc = desc_tag.get_text(strip=True) if desc_tag else "설명 없음"

                # 3. 본문 수집 (ncgbt-article)
                # 여러 개의 article이 있을 수 있으므로 합침
                articles = soup.select(".ncgbt-article")
                body_text = []
                for article in articles:
                    text = article.get_text(separator="\n", strip=True)
                    if text:
                        body_text.append(text)
                
                full_body = "\n\n".join(body_text)
                
                if full_body:
                    # [핵심 전략] 검색 정확도를 위해 임베딩할 텍스트에 제목과 설명을 포함시킵니다.
                    # 이렇게 하면 AI가 문맥을 더 잘 이해합니다.
                    enriched_content = f"문서 제목: {title}\n문서 요약: {desc}\n\n내용:\n{full_body}"

                    # 메타데이터 구성 (Pinecone에 저장될 부가 정보)
                    metadata = {
                        "source": url,
                        "title": title,       # 나중에 답변 출처 표시에 사용
                        "description": desc   # 나중에 답변 보충 설명에 사용
                    }

                    doc = Document(
                        page_content=enriched_content, # 실제 벡터화되어 검색되는 내용
                        metadata=metadata              # 검색 후 따라오는 꼬리표 정보
                    )
                    
                    valid_docs.append(doc)
                    print(f"      ✅ 수집 성공: [{title}]")
                else:
                    print("      ⚠️ 본문 내용이 비어있음")
                    
            except Exception as e:
                print(f"      ⚠️ 파싱 에러: {e}")
                continue

    except Exception as e:
        print(f"❌ 브라우저 에러: {e}")
    finally:
        driver.quit()

    # ... (이후 텍스트 분할 및 저장 로직은 기존과 동일) ...
    # 다만, enriched_content 길이가 길어졌으므로 chunk_size 조절을 고려해볼 만합니다.
    if not valid_docs:
        return

    print("2. 텍스트 분할 중...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    splits = text_splitter.split_documents(valid_docs)
    
    # Pinecone 저장 부분 (기존 동일)
    print(f"3. Pinecone('{INDEX_NAME}')에 데이터 저장 시작...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    PineconeVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    print("🎉 저장 완료!")

if __name__ == "__main__":
    # 테스트 모드 설정
    # target_urls = [
    #     # "https://aion2.plaync.com/ko-kr/guidebook/view?title=%EA%B2%80%EC%84%B1%20%EC%8A%A4%ED%82%AC"
    #     "https://aion2.plaync.com/ko-kr/guidebook/view?title=%EA%B2%80%EC%84%B1"
    # ]
    target_urls = collect_nc_guide_urls(4234, 4244)

    process_and_save_docs(target_urls)