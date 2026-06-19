import requests
import sys

url = "http://localhost:8000/v1/chat/completions"
payload = {
    "messages": [{"role": "user", "content": "说个数字"}],
    "max_tokens": 10
}

fail_count = 0
for i in range(1, 11):
    try:
        resp = requests.post(url, json=payload, timeout=30)
        data = resp.json()
        # 检查响应中是否包含 "error" 字段（OpenAI 风格错误）
        if "error" in data:
            print(f"Request {i}: ❌ fail (error in response)")
            fail_count += 1
        else:
            # 可选：打印正常回答，便于观察
            content = data["choices"][0]["message"]["content"]
            print(f"Request {i}: ✅ success -> {content}")
    except Exception as e:
        print(f"Request {i}: ❌ exception: {e}")
        fail_count += 1

if fail_count == 0:
    print("\n测试5通过：连续10次请求无错误")
    sys.exit(0)
else:
    print(f"\n测试5失败：{fail_count} 次请求出现错误")
    sys.exit(1)