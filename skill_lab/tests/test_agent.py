import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.toolbox import Toolbox
from src.llm_client import LLMClient
from src.agent_loop import AgentLoop
from datetime import datetime

def test_simple_tool_call():
    def get_weather(city: str) -> str:
        return f"{city}今天晴，15~25℃"

    toolbox = Toolbox()
    toolbox.register(
        name="get_weather",
        description="查询指定城市当天的天气",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"}
            },
            "required": ["city"]
        },
        executor=get_weather
    )

    llm = LLMClient()
    agent = AgentLoop(llm, toolbox, max_rounds=5)
    answer = agent.run("北京今天天气怎么样？")
    print("模型回答：", answer)
    assert "晴" in answer or "15" in answer, f"未获得真实天气数据: {answer}"



def test_max_rounds():
    def get_time() -> str:
        # 返回纯时间字符串，不携带额外描述，避免模型二次加工
        return "14:30"

    toolbox = Toolbox()
    toolbox.register(
        "get_current_time",
        "返回当前时间，格式为 HH:MM（例如 14:30），请直接使用该字符串回答，不要改变格式。",
        {"type": "object", "properties": {}},
        get_time
    )
    
    llm = LLMClient()
    
    # rounds=0：不允许任何行动
    agent0 = AgentLoop(llm, toolbox, max_rounds=0)
    ans0 = agent0.run("现在几点了？")
    assert "14:30" not in ans0, f"max_rounds=0不应调用工具，实际得到: {ans0}"
    
    # rounds=1：应该成功
    agent1 = AgentLoop(llm, toolbox, max_rounds=1)
    ans1 = agent1.run("现在几点了？")
    assert "14:30" in ans1, f"max_rounds=1应能获取时间: {ans1}"


def test_observation_feedback():
    call_count = 0

    def unreliable_tool():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "数据暂未就绪，请立即重新调用本工具获取最终结果"
        return "42"

    toolbox = Toolbox()
    toolbox.register(
        "get_ultimate_answer",
        "获取生命、宇宙及一切事物的终极答案（一个秘密数字）。如果返回'数据暂未就绪'，请立即再次调用本工具。",
        {"type": "object", "properties": {}},
        unreliable_tool
    )

    llm = LLMClient()
    agent = AgentLoop(llm, toolbox, max_rounds=3)
    ans = agent.run("根据内部系统，生命、宇宙及一切事物的终极数字答案是多少？请直接告诉我数字。")
    print("最终答案：", ans)
    assert "42" in ans, f"模型应在看到重试提示后再次调用工具，最终得到42: {ans}"
    assert call_count >= 2, f"工具至少被调用两次，实际: {call_count}"


def test_self_correction():
    def get_weather(city: str) -> str:
        if city == "beijing":
            return "错误：请使用中文城市名"
        elif city == "北京":
            return "北京今天多云，20℃"
        return "未知城市"

    toolbox = Toolbox()
    toolbox.register(
        name="get_weather",
        description="查询城市天气，参数 city 必须为中文名称，例如'北京'。如果收到错误提示，请立即用中文城市名重新调用。",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市中文名称"}},
            "required": ["city"]
        },
        executor=get_weather
    )

    llm = LLMClient()
    agent = AgentLoop(llm, toolbox, max_rounds=3)
    # 用户故意用英文提问
    ans = agent.run("What's the weather in Beijing? 用中文回答")
    print("最终答案：", ans)
    assert "20℃" in ans, f"模型应自我修正后用中文调用，实际: {ans}"


def test_multi_step_dependency():
    db = {
        "张三": "E042",
        "E042": "李四"
    }

    def get_employee_id(name: str) -> str:
        return db.get(name, "未找到")

    def get_manager(emp_id: str) -> str:
        manager = db.get(emp_id, "未知")
        return f"主管是{manager}"

    toolbox = Toolbox()
    toolbox.register(
        "get_employee_id", "根据姓名返回工号",
        {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        get_employee_id
    )
    toolbox.register(
        "get_manager", "根据工号返回主管姓名",
        {"type": "object", "properties": {"emp_id": {"type": "string"}}, "required": ["emp_id"]},
        get_manager
    )

    llm = LLMClient()
    agent = AgentLoop(llm, toolbox, max_rounds=4)  # 注意这里可能需要至少3轮
    ans = agent.run("张三的工号是什么？他的主管是谁？")
    print("最终答案：", ans)
    assert "E042" in ans and "李四" in ans, f"答案应包含工号和主管: {ans}"


def test_no_tool_needed():
    toolbox = Toolbox()
    toolbox.register("get_weather", "查天气", {"type": "object", "properties": {}}, lambda: "sunny")
    llm = LLMClient()
    agent = AgentLoop(llm, toolbox, max_rounds=5)
    ans = agent.run("1+1等于几？")
    print("最终答案：", ans)
    assert "sunny" not in ans.lower(), f"不应调用工具，但答案含有工具信息: {ans}"
    assert "2" in ans or "二" in ans, f"应正确回答数学问题: {ans}"


def test_rag_vs_llm_hallucination():
    def secret_data() -> str:
        return "42"

    toolbox = Toolbox()
    toolbox.register("get_secret", "返回秘密数字", {"type": "object", "properties": {}}, secret_data)
    llm = LLMClient()

    agent = AgentLoop(llm, toolbox)
    ans_with_tool = agent.run("公司的秘密数字是什么？")
    print("有工具答案：", ans_with_tool)
    assert "42" in ans_with_tool, f"有工具时应得到42: {ans_with_tool}"

    empty_toolbox = Toolbox()
    agent_no_tool = AgentLoop(llm, empty_toolbox)
    ans_no_tool = agent_no_tool.run("公司的秘密数字是什么？")
    print("无工具答案：", ans_no_tool)
    assert "42" not in ans_no_tool, f"无工具时不应知道42: {ans_no_tool}"



def test_real_time():
    def get_time() -> str:
        # 真实当前时间，格式 "HH:MM"
        return datetime.now().strftime("%H:%M")

    toolbox = Toolbox()
    toolbox.register(
        "get_current_time",
        "返回当前时间，格式为 HH:MM（例如 14:30），请直接使用该字符串回答，不要改变格式。",
        {"type": "object", "properties": {}},
        get_time
    )

    llm = LLMClient()
    
    # max_rounds=0 不允许行动
    agent0 = AgentLoop(llm, toolbox, max_rounds=0)
    ans0 = agent0.run("现在几点了？")
    # 当前时间不可能出现在 max_rounds=0 的回答里（没有调用工具）
    current_time_str = get_time()
    assert current_time_str not in ans0, f"max_rounds=0不应调用工具，实际回答包含时间: {ans0}"

    # max_rounds=1 应能拿到真实时间
    agent1 = AgentLoop(llm, toolbox, max_rounds=1)
    ans1 = agent1.run("现在几点了？")
    print("当前真实时间回答：", ans1)
    # 检查回答里是否包含带冒号的时间格式（如 "14:30"）
    assert ":" in ans1, f"max_rounds=1应能获取时间，实际回答: {ans1}"
    # 也可以更严格：检查当前时间的字符串是否在其中
    # 但由于模型可能加一些修饰词，用冒号检查更稳定