import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def inspect_page():
    url = "https://aion2.plaync.com/ko-kr/guidebook/view?title=%EC%88%98%ED%98%B8%EC%84%B1%20%EC%8A%A4%ED%82%AC"
    
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    try:
        print(f"Accessing {url}...")
        driver.get(url)
        
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "ncgbt-article")))
        time.sleep(2) # Wait for render
        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        # Save the relevant part (article) to a file
        article = soup.select_one(".ncgbt-article")
        if article:
            with open("skill_page_source.html", "w", encoding="utf-8") as f:
                f.write(article.prettify())
            print("Saved article HTML to skill_page_source.html")
        else:
            print("Could not find .ncgbt-article")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    inspect_page()
