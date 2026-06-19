import json
from micromcp.protocol import create_request, parse_message

class MCPClient:
    def __init__(self, read_stream, write_stream):
        """
        read_stream: 无参数可调用对象，返回一个 JSON-RPC 消息字符串（阻塞）。
        write_stream: 接受一个字符串参数，发送消息。
        """
        self.read = read_stream
        self.write = write_stream
        self._request_id = 0
        self._pending = {}   # 暂存乱序到达的响应 {id: response_dict}

    def _send_and_receive(self, request_str: str) -> dict:
        """
        发送请求并等待匹配 id 的响应。
        处理乱序：收到的响应可能不是当前请求的，暂存到 pending 字典。
        返回解析后的完整响应字典（包含 jsonrpc, id, result/error 等）。
        """
        req = parse_message(request_str)
        req_id = req["id"]

        # 先检查 pending 中是否已有该 id 的响应
        if req_id in self._pending:
            resp = self._pending.pop(req_id)
            return resp

        self.write(request_str)
        while True:
            resp_str = self.read()
            resp = parse_message(resp_str)
            resp_id = resp.get("id")
            if resp_id == req_id:
                # 将可能暂存的响应合并处理
                return resp
            else:
                self._pending[resp_id] = resp

    def initialize(self) -> dict:
        """发送 initialize 请求，返回服务器能力信息（result 字段内容）。"""
        self._request_id += 1
        req = create_request("initialize", params={}, request_id=self._request_id)
        resp = self._send_and_receive(req)
        return resp["result"]

    def list_tools(self) -> list:
        """调用 tools/list，返回工具列表。"""
        self._request_id += 1
        req = create_request("tools/list", params={}, request_id=self._request_id)
        resp = self._send_and_receive(req)
        return resp["result"]

    def call_tool(self, tool_name: str, **kwargs) -> dict:
        """
        调用 tools/call，传入工具名和参数。
        返回工具执行结果（成功时）或包含 error 键的字典（失败时）。
        """
        self._request_id += 1
        req = create_request(
            "tools/call",
            params={"name": tool_name, "arguments": kwargs},
            request_id=self._request_id
        )
        resp = self._send_and_receive(req)
        if "result" in resp:
            return resp["result"]
        else:
            # 协议级错误或工具执行错误，返回 error 对象
            return {"error": resp.get("error", {}).get("message", "未知错误")}

    def tools_as_openai_schema(self) -> list[dict]:
        """
        将 self.list_tools() 返回的工具列表转换为 OpenAI 兼容的 tools 参数格式。
        """
        tools = self.list_tools()
        schema = []
        for tool in tools:
            schema.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["inputSchema"]
                }
            })
        return schema