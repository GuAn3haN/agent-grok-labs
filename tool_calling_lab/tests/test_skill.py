import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.toolbox import Toolbox
from src.llm_client import LLMClient
from src.agent_loop import AgentLoop
from datetime import datetime
from src.skill import Skill

def test_skill_standalone_execution():
    # 模拟数据库
    db = {"张三": "E042", "E042": "李四"}

    # 内部工具函数
    def get_employee_id(name: str) -> str:
        return db.get(name, "未找到")

    def get_manager(emp_id: str) -> str:
        manager = db.get(emp_id, "未知")
        return f"主管是{manager}"

    # 创建 Skill
    skill = Skill(
        name="get_employee_info",
        description="根据员工姓名，返回其工号和主管姓名",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        }
    )

    # 注册内部工具——注意：用的是 skill._internal_toolbox
    skill._internal_toolbox.register(
        "get_employee_id", "根据姓名返回工号",
        {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        get_employee_id
    )
    skill._internal_toolbox.register(
        "get_manager", "根据工号返回主管",
        {"type": "object", "properties": {"emp_id": {"type": "string"}}, "required": ["emp_id"]},
        get_manager
    )

    # executor：这就是“原本由模型承担”的流程逻辑
    def executor(params, toolbox):
        name = params["name"]
        emp_id = toolbox.execute("get_employee_id", {"name": name})
        if "未找到" in emp_id:
            return f"未找到员工 {name}"
        manager_info = toolbox.execute("get_manager", {"emp_id": emp_id})
        return f"{name}的工号是{emp_id}，{manager_info}"

    skill.set_executor(executor)

    # 直接执行——不经过 LLM
    result = skill.run({"name": "张三"})
    assert "E042" in result and "李四" in result, f"Skill 执行结果不正确: {result}"







def test_tool_calling_multi_step():
    db = {"张三": "E042", "E042": "李四"}

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
        "get_manager", "根据工号返回主管",
        {"type": "object", "properties": {"emp_id": {"type": "string"}}, "required": ["emp_id"]},
        get_manager
    )

    llm = LLMClient()
    agent = AgentLoop(llm, toolbox, max_rounds=5)
    answer = agent.run("张三的工号是什么？他的主管是谁？")
    assert "E042" in answer and "李四" in answer, f"Tool Calling 结果不正确: {answer}"




def test_skill_mode_single_call():
    db = {"张三": "E042", "E042": "李四"}

    # 1. 构建 Skill（逻辑同测试 1）
    skill = Skill(
        name="get_employee_info",
        description="根据员工姓名，返回其工号和主管姓名",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        }
    )

    def get_employee_id(name: str) -> str:
        return db.get(name, "未找到")

    def get_manager(emp_id: str) -> str:
        return db.get(emp_id, "未知") + "（主管）"  # 简单标记以识别

    skill._internal_toolbox.register(
        "get_employee_id", "根据姓名返回工号",
        {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        get_employee_id
    )
    skill._internal_toolbox.register(
        "get_manager", "根据工号返回主管",
        {"type": "object", "properties": {"emp_id": {"type": "string"}}, "required": ["emp_id"]},
        get_manager
    )

    def executor(params, toolbox):
        name = params["name"]
        emp_id = toolbox.execute("get_employee_id", {"name": name})
        if "未找到" in emp_id:
            return f"未找到员工 {name}"
        manager_info = toolbox.execute("get_manager", {"emp_id": emp_id})
        return f"{name}的工号是{emp_id}，{manager_info}"

    skill.set_executor(executor)

    # 2. 将 Skill 注册到全局工具箱（只暴露 Skill，内部工具不可见）
    global_toolbox = Toolbox()
    global_toolbox.register(
        skill.name,
        skill.description,
        skill.parameters,
        # 关键：用 lambda 将 skill.run 适配成 Toolbox 需要的 executor 签名
        lambda **kwargs: skill.run(kwargs)
    )

    # 3. 用 AgentLoop 运行
    llm = LLMClient()
    agent = AgentLoop(llm, global_toolbox, max_rounds=3)
    answer = agent.run("帮我查一下张三的工号和他的主管是谁？")

    assert "E042" in answer and "李四" in answer, f"Skill 模式结果不正确: {answer}"

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))



def test_skill_mode_single_call():
    """测试3：Skill模式——模型只看到任务接口，单次调用即完成"""
    
    # ---------- 1. 准备数据 ----------
    db = {"张三": "E042", "E042": "李四"}
    
    # ---------- 2. 构建 Skill ----------
    skill = Skill(
        name="get_employee_info",
        description="根据员工姓名，返回其工号和主管姓名",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        }
    )
    
    # 内部工具函数（对模型不可见）
    def get_employee_id(name: str) -> str:
        return db.get(name, "未找到")
    
    def get_manager(emp_id: str) -> str:
        manager = db.get(emp_id, "未知")
        return f"主管是{manager}"
    
    # 注册到 Skill 的内部工具箱
    skill._internal_toolbox.register(
        "get_employee_id",
        "根据姓名返回工号",
        {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        get_employee_id
    )
    skill._internal_toolbox.register(
        "get_manager",
        "根据工号返回主管",
        {"type": "object", "properties": {"emp_id": {"type": "string"}}, "required": ["emp_id"]},
        get_manager
    )
    
    # 执行逻辑：这里编写的流程原本是模型在多轮 Think→Act→Observe 中自行决定的
    def executor(params, toolbox):
        name = params["name"]
        emp_id = toolbox.execute("get_employee_id", {"name": name})
        if "未找到" in emp_id:
            return f"未找到员工 {name}"
        manager_info = toolbox.execute("get_manager", {"emp_id": emp_id})
        return f"{name}的工号是{emp_id}，{manager_info}"
    
    skill.set_executor(executor)
    
    # ---------- 3. 将 Skill 注册到全局工具箱 ----------
    global_toolbox = Toolbox()
    
    # 适配层：Toolbox 传参是 **kwargs，Skill.run 要的是 dict
    def invoke_skill(**kwargs):
        return skill.run(kwargs)
    
    global_toolbox.register(
        skill.name,
        skill.description,
        skill.parameters,
        invoke_skill
    )
    
    # ---------- 4. 运行 Agent ----------
    llm = LLMClient()
    agent = AgentLoop(llm, global_toolbox, max_rounds=3)
    answer = agent.run("帮我查一下张三的工号和他的主管是谁？")
    
    # ---------- 5. 断言 ----------
    assert "E042" in answer, f"应包含工号: {answer}"
    assert "李四" in answer, f"应包含主管: {answer}"
    
    # 可选：观察 action_count（你可以在 AgentLoop 中临时打印）
    # print("Skill 模式行动次数:", agent.action_count)  # 应为 1



def test_skill_internal_error_handling():
    """测试4：Skill 内部错误处理，模型无感知"""

    # 模拟一个“第一次失败，第二次成功”的工具
    call_counter = 0

    def unreliable_emp_id(name: str) -> str:
        nonlocal call_counter
        call_counter += 1
        if call_counter == 1:
            return "错误：服务超时，请重试"
        return "E042"

    def get_manager(emp_id: str) -> str:
        # 简单返回，与之前一致
        return f"主管是李四"

    # 创建 Skill
    skill = Skill(
        name="get_employee_info",
        description="根据员工姓名，返回其工号和主管姓名",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        }
    )

    # 注册内部工具
    skill._internal_toolbox.register(
        "get_employee_id",
        "根据姓名返回工号",
        {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        unreliable_emp_id
    )
    skill._internal_toolbox.register(
        "get_manager",
        "根据工号返回主管",
        {"type": "object", "properties": {"emp_id": {"type": "string"}}, "required": ["emp_id"]},
        get_manager
    )

    # 执行器：带重试逻辑
    def executor(params, toolbox):
        name = params["name"]
        max_retries = 3
        for attempt in range(max_retries):
            emp_id = toolbox.execute("get_employee_id", {"name": name})
            if "超时" not in emp_id:
                break
            # 可选：实际场景中可增加等待或日志
        else:
            # 全部重试失败
            return "错误：多次尝试后仍无法获取工号"

        manager_info = toolbox.execute("get_manager", {"emp_id": emp_id})
        return f"{name}的工号是{emp_id}，{manager_info}"

    skill.set_executor(executor)

    # 直接执行（不经过模型），验证内部重试生效
    result = skill.run({"name": "张三"})
    assert "E042" in result and "李四" in result, f"重试后应成功，实际: {result}"
    assert call_counter == 2, f"应调用2次（第一次失败，第二次成功），实际: {call_counter}"


def test_decision_locus_comparison():
    """测试5：决策权对比——Tool Calling vs Skill 的工具调用总次数"""

    db = {"张三": "E042", "E042": "李四"}

    # ---------- Tool Calling 模式 ----------
    def get_employee_id(name: str) -> str:
        return db.get(name, "未找到")
    def get_manager(emp_id: str) -> str:
        manager = db.get(emp_id, "未知")
        return f"主管是{manager}"

    toolbox_tc = Toolbox()
    toolbox_tc.register("get_employee_id", "...", 
                         {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
                         get_employee_id)
    toolbox_tc.register("get_manager", "...",
                         {"type": "object", "properties": {"emp_id": {"type": "string"}}, "required": ["emp_id"]},
                         get_manager)

    llm = LLMClient()
    agent_tc = AgentLoop(llm, toolbox_tc, max_rounds=5)
    answer_tc = agent_tc.run("张三的工号是什么？他的主管是谁？")
    tc_actions = agent_tc.action_count  # 需要你在 AgentLoop 中设为实例属性

    # ---------- Skill 模式 ----------
    skill = Skill("get_employee_info", "...", 
                  {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]})
    skill._internal_toolbox.register("get_employee_id", "...",
                                     {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
                                     get_employee_id)
    skill._internal_toolbox.register("get_manager", "...",
                                     {"type": "object", "properties": {"emp_id": {"type": "string"}}, "required": ["emp_id"]},
                                     get_manager)

    def executor(params, toolbox):
        name = params["name"]
        emp_id = toolbox.execute("get_employee_id", {"name": name})
        if "未找到" in emp_id:
            return f"未找到员工 {name}"
        manager_info = toolbox.execute("get_manager", {"emp_id": emp_id})
        return f"{name}的工号是{emp_id}，{manager_info}"
    skill.set_executor(executor)

    global_toolbox = Toolbox()
    def invoke_skill(**kwargs):
        return skill.run(kwargs)
    global_toolbox.register(skill.name, skill.description, skill.parameters, invoke_skill)

    agent_skill = AgentLoop(llm, global_toolbox, max_rounds=3)
    answer_skill = agent_skill.run("帮我查一下张三的工号和他的主管是谁？")
    skill_actions = agent_skill.action_count

    # 验证结果都正确
    print(f"Tool Calling 模式工具调用次数: {tc_actions}")
    print(f"Skill 模式工具调用次数: {skill_actions}") 
    assert "E042" in answer_tc and "李四" in answer_tc
    assert "E042" in answer_skill and "李四" in answer_skill

    # 核心断言：Skill 模式下模型只调了一次工具，而在 Tool Calling 模式下至少两次
    assert skill_actions == 1, f"Skill 模式下模型应只调用1次工具，实际: {skill_actions}"
    assert tc_actions >= 2, f"Tool Calling 模式下模型至少调用2次工具，实际: {tc_actions}"