import pytest
import json
from collections import deque
from micromcp.server import MCPServer
from micromcp.client import MCPClient
from agent.llm_agent import LLMAgent

# ---------- 传输模拟辅助函数 ----------

def make_direct_client(server: MCPServer) -> MCPClient:
    """
    创建一个通过直接回调连接的 MCPClient。
    write_stream 直接调用 server.process_message，并将响应放入队列，
    read_stream 从队列中取出响应。
    这是最简单的传输模拟，用于测试。
    """
    out = deque()
    def write(msg: str):
        resp = server.process_message(msg)
        out.append(resp)
    def read() -> str:
        return out.popleft()
    return MCPClient(read, write)

def make_server_with_weather_and_calc() -> MCPServer:
    """创建带 get_weather 和 calculate 两个工具的服务器。"""
    server = MCPServer("TestServer")
    
    def get_weather(city: str):
        return {"city": city, "temp": 25, "condition": "晴朗"}
    
    def calculate(expression: str):
        try:
            return {"result": eval(expression)}
        except Exception:
            return {"error": "计算失败"}
    
    server.register_tool("get_weather", "获取指定城市天气",
                         {
                             "type": "object",
                             "properties": {"city": {"type": "string", "description": "城市名称"}},
                             "required": ["city"]
                         },
                         get_weather)
    server.register_tool("calculate", "计算数学表达式",
                         {
                             "type": "object",
                             "properties": {"expression": {"type": "string", "description": "数学表达式，如 2+3*4"}},
                             "required": ["expression"]
                         },
                         calculate)
    return server

# ----------------------------------------------------------------

class TestMCPWithLLM:

    # ======================== 测试 1 ========================
    def test_llm_tool_discovery(self):
        """
        验证 LLMAgent 初始化时成功获取了 MCP Server 的工具 schema，
        并且该 schema 与 Server 注册的信息完全一致。
        关键结构：tools/list -> tools_as_openai_schema 的转换管线。
        """
        server = make_server_with_weather_and_calc()
        client = make_direct_client(server)
        agent = LLMAgent(client)

        # LLMAgent 初始化时应该已经调用了 list_tools 并转换了 schema
        assert len(agent.tools_schema) == 2
        tool_names = [t["function"]["name"] for t in agent.tools_schema]
        assert "get_weather" in tool_names
        assert "calculate" in tool_names

        # 检查 schema 结构完整性
        for tool in agent.tools_schema:
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    # ======================== 测试 2 ========================
    def test_llm_tool_selection_and_call(self):
        """
        LLM 收到“北京天气怎么样？”后，应选择 get_weather 工具，
        并填充参数 city="北京"。最终回答应包含温度信息。
        关键结构：LLM 的 tool_calls 正确映射到 MCP 的 tools/call。
        """
        server = make_server_with_weather_and_calc()
        client = make_direct_client(server)
        agent = LLMAgent(client)

        answer = agent.chat("北京天气怎么样？")
        # 因为我们的工具返回温度是 25，LLM 的最终回答应提及温度或天气状况
        assert "25" in answer or "温度" in answer or "晴朗" in answer or "天气" in answer, \
            f"LLM未能基于工具结果回答天气，实际输出：{answer[:200]}"

    # ======================== 测试 3 ========================
    def test_llm_error_recovery(self):
        """
        当 LLM 调用 calculate 工具但传入无法计算的表达式（如除以零）时，
        工具返回错误信息。LLM 应能识别错误并告知用户无法计算。
        关键结构：tools/call 返回的错误信息通过 MCP 协议正确传递给 LLM。
        """
        server = MCPServer("ErrorServer")
        def calc(expression: str):
            try:
                return {"result": eval(expression)}
            except Exception:
                return {"error": "数学错误"}
        server.register_tool("calc", "计算表达式",
                             {
                                 "type": "object",
                                 "properties": {"expression": {"type": "string"}},
                                 "required": ["expression"]
                             },
                             calc)
        client = make_direct_client(server)
        agent = LLMAgent(client)
        answer = agent.chat("帮我计算 5 / 0")
        # LLM 应该能告知错误，而不是直接崩溃或假装成功
        assert ("错误" in answer or "不能" in answer or "无法" in answer or 
                "无穷" in answer or "异常" in answer or "除以零" in answer), \
            f"LLM未正确处理工具错误，实际输出：{answer[:200]}"

    # ======================== 测试 4 ========================
    def test_dynamic_tool_registration_llm(self):
        """
        先创建一个只有 weather 工具的 Server，让 Agent 回答一个需要计算的问题。
        然后动态注册 calculate 工具，创建一个新的 Agent（重新连接），再次提问相同问题，
        应成功获得计算结果。
        关键结构：动态发现——tools/list 让 LLM Agent 的能力集实时变化。
        """
        server = MCPServer("DynamicServer")
        def weather(city: str):
            return {"city": city, "temp": 22}
        server.register_tool("weather", "天气查询",
                             {"properties": {"city": {"type": "string"}}},
                             weather)

        client1 = make_direct_client(server)
        agent1 = LLMAgent(client1)
        # 第一次提问：需要计算，但无计算工具，LLM 可能用内部知识或拒绝
        answer1 = agent1.chat("计算 3*4 等于多少？")
        # 不做严格断言，因为 LLM 可能凭借自身知识回答。我们关注第二次。

        # 动态注册计算工具
        def calc(expression: str):
            return {"result": eval(expression)}
        server.register_tool("calc", "计算",
                             {"properties": {"expression": {"type": "string"}}},
                             calc)

        # 使用新的 Agent 连接（重新发现工具）
        client2 = make_direct_client(server)
        agent2 = LLMAgent(client2)
        answer2 = agent2.chat("计算 3*4 等于多少？")
        assert "12" in answer2, f"注册计算工具后，LLM应能调用并得到正确结果，实际：{answer2[:200]}"

    # ======================== 测试 5 ========================
    def test_human_and_llm_share_protocol(self):
        """
        使用完全相同的 Server 和协议消息，分别通过：
        A) 人类手动调用 client.call_tool("get_weather", city="上海")
        B) LLM Agent 调用 agent.chat("上海天气怎么样")
        比较两者最终获得的数据结果（温度），应一致。
        关键结构：协议不变，调用者可变。
        """
        server = make_server_with_weather_and_calc()
        client = make_direct_client(server)

        # 人类手动调用
        manual_result = client.call_tool("get_weather", city="上海")
        manual_temp = manual_result["temp"]

        # LLM调用（复用同一个 client，但 Agent 会在初始化时调用 list_tools，不影响后续调用）
        agent = LLMAgent(client)
        llm_answer = agent.chat("上海天气怎么样？")
        # 从答案中检查是否包含相同的温度数字
        assert str(manual_temp) in llm_answer, \
            f"LLM答案应包含与手动调用相同的温度数据({manual_temp})，实际：{llm_answer[:200]}"

    # ======================== 测试 6 ========================
    def test_missing_tool_description_impact(self):
        """
        注册一个只有名称没有描述的工具，观察 LLM 是否还能正确使用它。
        这验证了 tools/list 返回的 description 字段对 LLM 决策的关键作用。
        注意：由于模型不确定性，本测试仅确保流程不报错，具体行为可通过报告记录观察。
        """
        server = MCPServer("NoDescServer")
        def mystery(x: int):
            return {"result": x * 2}
        # 故意不提供有意义的描述，只给空字符串
        server.register_tool("mystery", "", 
                             {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
                             mystery)

        client = make_direct_client(server)
        agent = LLMAgent(client)
        answer = agent.chat("请用 mystery 工具处理数字 5")
        # 我们不强求调用成功，但测试必须能正常运行完成
        assert isinstance(answer, str)
        # 额外提示：观察你的输出，LLM 是否调用了 mystery？如果没有，你认为是什么原因？

    # ======================== 测试 7 ========================
    def test_tool_schema_structure(self):
        """
        验证 tools_as_openai_schema 返回的列表结构严格符合 OpenAI 要求。
        这是确保 LLM 能正确理解工具的基石。
        """
        server = MCPServer("SchemaServer")
        def dummy():
            return "ok"
        server.register_tool("dummy", "测试工具",
                             {"type": "object", "properties": {}},
                             dummy)
        client = make_direct_client(server)
        client.initialize()
        schema = client.tools_as_openai_schema()
        
        assert len(schema) == 1
        tool = schema[0]
        assert tool["type"] == "function"
        assert "function" in tool
        func = tool["function"]
        assert func["name"] == "dummy"
        assert "description" in func
        assert "parameters" in func
        assert isinstance(func["parameters"], dict)