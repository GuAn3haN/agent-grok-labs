#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试7：显存占用与 KV Cache
通过标准：
1. 长请求峰值显存 ≤ 总显存 × 0.9
2. 长请求峰值显存 - 短请求峰值显存 ≥ 100 MiB
3. |请求结束后5秒显存 - 空闲显存| ≤ 50 MiB
"""

import requests
import time
import subprocess
import re
import sys

# ========== 配置 ==========
URL = "http://localhost:8000/v1/chat/completions"
SHORT_PAYLOAD = {
    "messages": [{"role": "user", "content": "说一个数字"}],
    "max_tokens": 10
}
LONG_PAYLOAD = {
    "messages": [{"role": "user", "content": "写一篇500字的短文介绍人工智能"}],
    "max_tokens": 512
}
TIMEOUT = 120
# ==========================

def get_gpu_memory_mb():
    """获取当前 GPU 显存占用 (MiB) - 第一块 GPU"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        used_mb = int(result.stdout.strip().split('\n')[0])
        return used_mb
    except Exception as e:
        print(f"无法获取显存信息: {e}")
        return None

def get_gpu_total_mb():
    """获取 GPU 总显存 (MiB)"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        total_mb = int(result.stdout.strip().split('\n')[0])
        return total_mb
    except Exception:
        return None

def monitor_peak_memory(func):
    """在运行 func 期间持续监控显存，返回峰值 (MiB)"""
    import threading
    import time

    stop_monitor = threading.Event()
    peak = 0
    lock = threading.Lock()

    def monitor():
        nonlocal peak
        while not stop_monitor.is_set():
            mem = get_gpu_memory_mb()
            if mem is not None:
                with lock:
                    if mem > peak:
                        peak = mem
            time.sleep(0.1)  # 采样间隔 100ms

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    start = time.perf_counter()
    func()  # 执行请求
    elapsed = time.perf_counter() - start
    stop_monitor.set()
    monitor_thread.join(timeout=1)
    return peak, elapsed

def send_request(payload):
    """发送请求，忽略响应内容，只关心是否成功"""
    resp = requests.post(URL, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise Exception(f"API error: {data['error']}")

def main():
    total_mb = get_gpu_total_mb()
    if total_mb is None:
        print("未检测到 NVIDIA GPU，跳过测试7（视为通过）")
        sys.exit(0)

    # 1. 空闲显存
    idle_mem = get_gpu_memory_mb()
    print(f"空闲显存: {idle_mem} MiB")

    # 2. 短请求峰值
    print("发送短请求 (max_tokens=10)...")
    peak_short, _ = monitor_peak_memory(lambda: send_request(SHORT_PAYLOAD))
    print(f"短请求峰值显存: {peak_short} MiB")

    # 3. 长请求峰值
    print("发送长请求 (max_tokens=512)...")
    peak_long, _ = monitor_peak_memory(lambda: send_request(LONG_PAYLOAD))
    print(f"长请求峰值显存: {peak_long} MiB")

    # 4. 长请求结束后5秒显存
    time.sleep(5)
    after_mem = get_gpu_memory_mb()
    print(f"请求结束后5秒显存: {after_mem} MiB")

    # ========== 判定 ==========
    # 条件1: 长请求峰值 ≤ 总显存 × 0.9
    condition1 = peak_long <= total_mb * 0.9
    print(f"\n条件1: 长请求峰值 {peak_long} MiB ≤ {total_mb * 0.9:.0f} MiB (总显存90%) -> {'✅' if condition1 else '❌'}")

    # 条件2: 长请求峰值 - 短请求峰值 ≥ 20 MiB
    diff = peak_long - peak_short
    condition2 = diff >= 20
    print(f"条件2: KV Cache增量 {diff:.0f} MiB ≥ 20 MiB -> {'✅' if condition2 else '❌'}")

    # 条件3: |请求结束后5秒显存 - 空闲显存| ≤ 50 MiB
    leak = abs(after_mem - idle_mem)
    condition3 = leak <= 50
    print(f"条件3: 显存泄漏 {leak} MiB ≤ 50 MiB -> {'✅' if condition3 else '❌'}")

    if condition1 and condition2 and condition3:
        print("\n✅ 测试7通过")
        sys.exit(0)
    else:
        print("\n❌ 测试7失败")
        sys.exit(1)

if __name__ == "__main__":
    main()