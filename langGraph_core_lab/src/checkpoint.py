from typing import Optional


class MemorySaver:
    def __init__(self):
        # 内部存储：thread_id → 状态字典（拷贝）
        self._storage = {}

    def put(self, config: dict, state: dict) -> None:
        """保存 thread_id 对应的最新状态。"""
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id is None:
            return
        self._storage[thread_id] = dict(state)  # 保存副本

    def get(self, config: dict) -> Optional[dict]:
        """获取 thread_id 对应的已保存状态，若不存在则返回 None。"""
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id is None:
            return None
        saved = self._storage.get(thread_id)
        if saved is not None:
            return dict(saved)  # 返回副本
        return None