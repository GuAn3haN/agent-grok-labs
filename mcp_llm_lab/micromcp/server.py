from micromcp.protocol import parse_message, create_response, create_error

class MCPServer:
    def __init__(self, server_name: str, version: str = "1.0"):
        self.server_info = {"name": server_name, "version": version}
        self.tools = {}

    def register_tool(self, name: str, description: str, input_schema: dict, handler):
        self.tools[name] = {
            "description": description,
            "inputSchema": input_schema,
            "handler": handler
        }

    def _handle_initialize(self, msg_id, params) -> str:
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": self.server_info
        }
        return create_response(msg_id, result)

    def _handle_tools_list(self, msg_id, params) -> str:
        tools_list = []
        for name, info in self.tools.items():
            tools_list.append({
                "name": name,
                "description": info["description"],
                "inputSchema": info["inputSchema"]
            })
        return create_response(msg_id, tools_list)

    def _handle_tools_call(self, msg_id, params) -> str:
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in self.tools:
            return create_error(msg_id, -32601, f"工具 '{tool_name}' 不存在")

        tool = self.tools[tool_name]
        required = tool["inputSchema"].get("required", [])
        for field in required:
            if field not in arguments:
                return create_error(msg_id, -32602, f"缺少必要参数: {field}")

        try:
            result = tool["handler"](**arguments)
            return create_response(msg_id, result)
        except Exception as e:
            return create_error(msg_id, -32603, f"工具执行错误: {str(e)}")

    def process_message(self, raw_message: str) -> str:
        try:
            msg = parse_message(raw_message)
        except Exception:
            return create_error(None, -32700, "解析错误")

        msg_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params", {})

        if method == "initialize":
            return self._handle_initialize(msg_id, params)
        elif method == "tools/list":
            return self._handle_tools_list(msg_id, params)
        elif method == "tools/call":
            return self._handle_tools_call(msg_id, params)
        else:
            return create_error(msg_id, -32601, f"方法 '{method}' 不存在")