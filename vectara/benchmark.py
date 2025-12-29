import requests
import pandas as pd
from tqdm import tqdm
import time

query_KEY = "zwt_3KLRg9sfGyGw-fS1MXVtQZpLMCB01UGwNTsT_g"
CORPUS_KEY = "jst_pdf"
INPUT_XLSX = "/Users/wangyaqi/Documents/cursor_project/jst-rag-demo/jst-rag-demo/测试文件/金盘benchmark测试.xlsx"
OUTPUT_XLSX = "vectara_benchmark_output.xlsx"

QUERY_URL = f"https://api.vectara.io/v2/corpora/{CORPUS_KEY}/query"

headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "x-api-key": query_KEY,
}

df = pd.read_excel(INPUT_XLSX)
results = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理进度"):
    question = str(row["问题"])
    golden = str(row["标准回答"])

    # 修正后的 payload
    payload = {
        "query": question,
        "search": {
            "lexical_interpolation": 0.5,
            "limit": 30,
            "offset": 0,
            "context_configuration": {
                "sentences_before": 1,
                "sentences_after": 1
            },
            "reranker": {
                "type": "customer_reranker",
                "reranker_name": "qwen3-reranker",
                "limit": 10,
                "include_context": True
            }
        },
        "generation": {
            "generation_preset_name": "mockingbird-2.0",
            "max_used_search_results": 10,
            "response_language": "zho",
            "model_parameters": {
                "temperature": 0.2,
                "max_tokens": 600
            },
            "citations": {
                "style": "markdown",
                "url_pattern": "https://mydocs/{doc.id}",
                "text_pattern": "{doc.title}"
            },
            "enable_factual_consistency_score": True
        },
        "save_history": False,
        "intelligent_query_rewriting": False,
        "stream_response": False
    }

    start_time = time.time()
    try:
        res = requests.post(QUERY_URL, headers=headers, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        vectara_answer = data.get("summary", "")
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP Error {res.status_code}: {e}"
        try:
            # 尝试解析服务端返回的 JSON 错误信息
            error_details = res.json()
            error_msg += f"\n详细错误信息: {error_details}"
        except:
            # 如果不是 JSON，直接获取文本
            error_msg += f"\n响应文本: {res.text}"
        
        tqdm.write(f"❌ 请求出错: {question}")
        tqdm.write(error_msg)
        vectara_answer = f"请求失败: {error_msg}"
    except Exception as e:
        tqdm.write(f"❌ 未知错误: {e}")
        vectara_answer = f"请求失败: {e}"
    
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)

    results.append({
        "问题": question,
        "标准回答": golden,
        "Vectara回答": vectara_answer,
        "耗时(s)": elapsed_time
    })
    tqdm.write(vectara_answer) # 调试时可选打印回答

out_df = pd.DataFrame(results)
out_df.to_excel(OUTPUT_XLSX, index=False)
print("完成！结果文件：", OUTPUT_XLSX)
