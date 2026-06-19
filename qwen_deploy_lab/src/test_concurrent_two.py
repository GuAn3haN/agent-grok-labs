#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试6：双请求并发
通过标准：
1. 两个请求均成功返回（状态码200，无error）
2. 两个请求耗时中，较大值 <= 较小值 * 1.5
3. 每个请求耗时 <= 测试2延迟的2倍（测试2延迟需手动输入或从文件读取）
"""

import requests
import concurrent.futures
import time
import sys

URL = "http://localhost:8000/v1/chat/completions"
PAYLOAD = {
    "messages": [{"role": "user", "content": "写一段200字的描述"}],
    "max_tokens": 200
}
TIMEOUT = 60

# 请根据你测试2的结果修改此值（单位：秒）
TEST2_LATENCY = 5.09   # 例如你之前测得的 5.09 秒

def send_request(worker_id):
    """发送请求，返回 (worker_id, 耗时秒数, 是否成功, 错误信息)"""
    start = time.perf_counter()
    try:
        resp = requests.post(URL, json=PAYLOAD, timeout=TIMEOUT)
        elapsed = time.perf_counter() - start
        if resp.status_code != 200:
            return (worker_id, elapsed, False, f"HTTP {resp.status_code}")
        data = resp.json()
        if "error" in data:
            return (worker_id, elapsed, False, f"error field: {data['error']}")
        return (worker_id, elapsed, True, None)
    except Exception as e:
        elapsed = time.perf_counter() - start
        return (worker_id, elapsed, False, str(e))

def main():
    print("开始双请求并发测试...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(send_request, i) for i in range(2)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # 收集结果
    times = []
    success = True
    for wid, t, ok, err in results:
        times.append(t)
        if ok:
            print(f"Worker {wid}: ✅ 成功, 耗时 {t:.3f} 秒")
        else:
            print(f"Worker {wid}: ❌ 失败, 耗时 {t:.3f} 秒, 错误: {err}")
            success = False

    if not success:
        print("\n❌ 测试6失败：存在请求失败")
        sys.exit(1)

    # 标准1：两个请求均正常结束（已经通过success检查）

    # 标准2：较大值 <= 较小值 * 1.5
    t_min = min(times)
    t_max = max(times)
    ratio_condition = t_max <= t_min * 1.5
    print(f"\n耗时比例: 较大值 {t_max:.3f} 秒, 较小值 {t_min:.3f} 秒, 比值 = {t_max/t_min:.2f}")
    if ratio_condition:
        print("✅ 耗时比例满足 <= 1.5")
    else:
        print("❌ 耗时比例不满足 <= 1.5")

    # 标准3：每个耗时 <= 测试2延迟的2倍
    threshold = TEST2_LATENCY * 2
    latency_condition = all(t <= threshold for t in times)
    print(f"单请求延迟阈值 (测试2的2倍): {threshold:.2f} 秒")
    if latency_condition:
        print("✅ 所有请求耗时均未超过阈值")
    else:
        print("❌ 存在请求耗时超过阈值")

    if ratio_condition and latency_condition:
        print("\n✅ 测试6通过")
        sys.exit(0)
    else:
        print("\n❌ 测试6失败")
        sys.exit(1)

if __name__ == "__main__":
    main()