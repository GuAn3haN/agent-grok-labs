import requests, time
start = time.perf_counter()
resp = requests.post("http://localhost:8000/v1/chat/completions", json={
    "messages":[{"role":"user","content":"写一首五言绝句"}],
    "max_tokens":128
})
tokens = len(resp.json()["choices"][0]["message"]["content"])
elapsed = time.perf_counter() - start
print(f"{tokens/elapsed:.1f}")