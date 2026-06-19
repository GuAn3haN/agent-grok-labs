import requests
import time

url = "http://localhost:8000/v1/chat/completions"
prompts = ["讲个笑话", "解释什么是AI", "推荐一本书", "说一个成语"]
total = 0.0
for p in prompts:
    start = time.perf_counter()
    resp = requests.post(url, json={"messages":[{"role":"user","content":p}], "max_tokens":64})
    elapsed = time.perf_counter() - start
    total += elapsed
    print(f"'{p}' 耗时 {elapsed:.3f} 秒, 状态码 {resp.status_code}")
print(f"T_serial = {total:.2f} 秒")