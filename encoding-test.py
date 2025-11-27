import os

def find_non_utf8_files(path):
    print(f"Checking files in {path}...")
    for root, dirs, files in os.walk(path):
        # 가상환경 폴더 등은 검사 제외
        if 'venv' in root or '.git' in root or '.idea' in root:
            continue
            
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        f.read()
                except UnicodeDecodeError:
                    print(f"🚫 발견! 인코딩 문제 파일: {full_path}")
                except Exception as e:
                    print(f"⚠️ 기타 에러 ({file}): {e}")

# 현재 폴더 검사
find_non_utf8_files("./")
