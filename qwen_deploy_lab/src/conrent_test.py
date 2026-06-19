import requests, time, concurrent.futures

def call(prompt):
    t0 = time.time()
    r = requests.post("http://localhost:8000/v1/chat/completions", json={
        "messages":[{"role":"user","content":prompt}],
        "max_tokens":64
    })
    return time.time() - t0, r.status_code

prompts = ["讲个笑话", "解释什么是AI", "推荐一本书", "说一个成语"]
start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    futures = [ex.submit(call, p) for p in prompts]
    for f in concurrent.futures.as_completed(futures):
        _, code = f.result()
        if code != 200:
            print("ERROR")
T_parallel = time.time() - start
print(f"{T_parallel:.2f}")