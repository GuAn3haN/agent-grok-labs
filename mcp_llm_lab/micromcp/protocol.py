import json
from typing import Any

def create_request(method: str, params: dict = None, request_id: Any = 1) -> str:
    """创建一个 JSON-RPC 2.0 请求字符串。"""
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params if params is not None else {}
    }
    return json.dumps(payload)

def create_response(request_id: Any, result: Any) -> str:
    """创建一个成功的 JSON-RPC 2.0 响应字符串。"""
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result
    }
    return json.dumps(payload)

def create_error(request_id: Any, code: int, message: str) -> str:
    """创建一个错误的 JSON-RPC 2.0 响应字符串。"""
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message
        }
    }
    return json.dumps(payload)

def parse_message(message_str: str) -> dict:
    """解析一个 JSON-RPC 消息，返回字典。"""
    return json.loads(message_str)