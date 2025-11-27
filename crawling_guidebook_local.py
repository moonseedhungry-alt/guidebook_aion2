import os
import time
import sys
import re
import json
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

DATA_DIR = "data"
JSON_FILE = os.path.join(DATA_DIR, "guide_docs.json")
INDEX_NAME = "aion2-guide-rag"
MODEL_NAME = "text-embedding-3-large"

def collect_nc_guide_urls(start_id, end_id):
    print(f"[*] Start Collection: CategoryId {start_id} ~ {end_id}")
    options = webdriver.ChromeOptions()
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    all_urls = set()
    try:
        for cat_id in range(start_id, end_id + 1):
            url = f"https://aion2.plaync.com/ko-kr/guidebook/list#categoryId={cat_id}"
            print(f"\n[*] Moving to Category {cat_id}...")
            driver.get(url)
            try:
                wait = WebDriverWait(driver, 10)
                target_class = "ncgbg-guide-depth-2-guide-item-link"
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, target_class)))
                time.sleep(1)
                elements = driver.find_elements(By.CLASS_NAME, target_class)
                count = 0
                for el in elements:
                    full_url = el.get_attribute("href")
                    if full_url and "view?title=" in full_url:
                        all_urls.add(full_url)
                        count += 1
                print(f"   [+] Collected {count} URLs")
            except Exception:
                print(f"   [-] No posts or loading failed for Category {cat_id}")
                continue
    except Exception as e:
        print(f"[!] Error: {e}")
    finally:
        driver.quit()
    return list(all_urls)

def process_and_save_docs(urls):
    if not urls:
        print("[!] No URLs provided.")
        return
    print(f"\n1. Loading and structuring data... ({len(urls)} URLs)")
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    valid_docs = []
    json_data_list = []
    try:
        for url in urls:
            print(f"   -> Accessing: {url}")
            driver.get(url)
            try:
                wait = WebDriverWait(driver, 10)
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "ncgbt-article")))
                time.sleep(1)
                soup = BeautifulSoup(driver.page_source, "html.parser")
                targets = soup.find_all(string=re.compile("보러가기"))
                for text_node in targets:
                    parent_link = text_node.find_parent("a")
                    if parent_link:
                        parent_link.decompose()
                    else:
                        if text_node.parent:
                            text_node.parent.decompose()
                title_tag = soup.select_one(".ncgbt-cover-title")
                title = title_tag.get_text(strip=True) if title_tag else "No Title"
                desc_tag = soup.select_one(".ncgbt-cover-desc")
                desc = desc_tag.get_text(strip=True) if desc_tag else "No Description"

                if "스킬" in title and "클래스" not in title:
                    print(f"      [Skill Parsing] Detected skill page: {title}")
                    skill_docs = []
                    article = soup.select_one(".ncgbt-article")
                    if article:
                        tables = article.find_all("table")
                        for table in tables:
                            section_name = "General Skill"
                            prev = table.find_previous(["h1", "h2", "h3", "h4", "h5", "h6", "p"])
                            if prev:
                                section_name = prev.get_text(strip=True)
                            
                            rows = table.find_all("tr")
                            if not rows: continue
                            
                            headers = [th.get_text(strip=True) for th in rows[0].find_all(["td", "th"])]
                            print(f"      [DEBUG] Headers: {headers}")
                            
                            if "명칭" not in headers:
                                print("      [DEBUG] '명칭' header not found, skipping table")
                                continue
                            
                            try:
                                name_idx = headers.index("명칭")
                                desc_idx = headers.index("설명") if "설명" in headers else -1
                                note_idx = headers.index("비고") if "비고" in headers else -1
                            except ValueError:
                                continue
                                
                            for row in rows[1:]:
                                cols = row.find_all("td")
                                if len(cols) <= name_idx: continue
                                s_name = cols[name_idx].get_text(separator=" ", strip=True)
                                s_desc = cols[desc_idx].get_text(separator="\n", strip=True) if desc_idx != -1 and len(cols) > desc_idx else ""
                                s_note = cols[note_idx].get_text(separator=" ", strip=True) if note_idx != -1 and len(cols) > note_idx else ""
                                skill_content = f"[{title}] {section_name} - {s_name}\n\nDescription:\n{s_desc}\n\nNote: {s_note}"
                                skill_metadata = {
                                    "source": url,
                                    "title": title,
                                    "description": desc,
                                    "skill_name": s_name,
                                    "skill_type": section_name,
                                    "category": "skill"
                                }
                                doc = Document(page_content=skill_content, metadata=skill_metadata)
                                valid_docs.append(doc)
                                json_data_list.append({
                                    "page_content": skill_content,
                                    "metadata": skill_metadata
                                })
                                skill_docs.append(s_name)
                        if skill_docs:
                            print(f"      [+] Structured {len(skill_docs)} skills")
                            continue

                articles = soup.select(".ncgbt-article")
                body_text = []
                for article in articles:
                    text = article.get_text(separator="\n", strip=True)
                    if text:
                        body_text.append(text)
                full_body = "\n\n".join(body_text)
                if full_body:
                    enriched_content = f"[{title}] Document.\nSummary: {desc}\n\nContent:\n{full_body}"
                    metadata = {
                        "source": url,
                        "title": title,
                        "description": desc
                    }
                    doc = Document(page_content=enriched_content, metadata=metadata)
                    valid_docs.append(doc)
                    json_data_list.append({
                        "page_content": enriched_content,
                        "metadata": metadata
                    })
                    print(f"      [+] Collected: [{title}]")
                else:
                    print("      [-] Empty content")
            except Exception as e:
                print(f"      [!] Parsing Error: {e}")
                continue
    except Exception as e:
        print(f"[!] Browser Error: {e}")
    finally:
        driver.quit()

    if not valid_docs:
        print("[!] No documents to save.")
        return

    json_filename = "data/guide_docs.json"
    print(f"\n2. Saving local file... ({json_filename})")
    try:
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(json_data_list, f, ensure_ascii=False, indent=4)
        print(f"   [+] Local save complete!")
    except Exception as e:
        print(f"   [!] Local save failed: {e}")

    print("\n3. Splitting text and uploading to Pinecone...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(valid_docs)
    print(f"   - Split {len(valid_docs)} docs into {len(splits)} chunks")
    print(f"   - Pinecone Index: '{INDEX_NAME}'")
    embeddings = OpenAIEmbeddings(model=MODEL_NAME)
    try:
        PineconeVectorStore.from_documents(documents=splits, embedding=embeddings, index_name=INDEX_NAME)
        print("   [+] Pinecone save complete!")
    except Exception as e:
        print(f"   [!] Pinecone save failed: {e}")

if __name__ == "__main__":
    target_urls = collect_nc_guide_urls(4234, 4244)
    process_and_save_docs(target_urls)