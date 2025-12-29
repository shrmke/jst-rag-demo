import os
import json
import requests
import certifi

API_KEY = "zut_3KLRg7xxDLWeLNSqRLOTZmqM8HoLbh6JVNAafw"
CORPUS_KEY = "jst_pdf"

folder_path = "/Users/wangyaqi/Documents/项目/jst2/金盘上市公告"

url = f"https://api.vectara.io/v2/corpora/{CORPUS_KEY}/upload_file"




MAX_SIZE_BYTES = 10 * 1024 * 1024   # 10 MB
UPLOAD_LOG = "uploaded_files.txt"


headers = {
    "Accept": "application/json",
    "x-api-key": API_KEY
}

# 读取已上传文件记录
if os.path.exists(UPLOAD_LOG):
    with open(UPLOAD_LOG, "r", encoding="utf-8") as f:
        uploaded_files = set(line.strip() for line in f.readlines())
else:
    uploaded_files = set()

def save_uploaded_file(filename):
    with open(UPLOAD_LOG, "a", encoding="utf-8") as f:
        f.write(filename + "\n")
    uploaded_files.add(filename)

def upload_file(filepath):
    filename = os.path.basename(filepath)

    # 跳过已上传的文件
    if filename in uploaded_files:
        print(f"✔ 已上传过，跳过：{filename}")
        return

    # 文件大小检查
    size = os.path.getsize(filepath)
    if size >= MAX_SIZE_BYTES:
        print(f"⛔ 跳过（文件过大 ≥ 10MB）：{filename} —— 大小：{round(size/1024/1024,2)} MB")
        return

    table_extraction_config = {
        "extract_tables": False
    }

    files = {
        "file": (filename, open(filepath, "rb"), "application/pdf"),
        "table_extraction_config": (None, json.dumps(table_extraction_config), "application/json"),
    }

    print(f"Uploading: {filename}")

    try:
        response = requests.post(url, headers=headers, files=files, verify=certifi.where(), timeout=60)
        if response.status_code == 201:
            print(f"✓ 成功：{filename}")
            save_uploaded_file(filename)
        else:
            print(f"✗ 失败：{filename}")
            try:
                print(response.json())
            except Exception:
                print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"⚠ 上传异常（网络或SSL问题）：{filename}\n{e}")

if __name__ == "__main__":
    for fname in os.listdir(folder_path):
        path = os.path.join(folder_path, fname)
        if os.path.isfile(path) and fname.lower().endswith(".pdf"):
            upload_file(path)
