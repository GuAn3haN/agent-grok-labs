class Toolbox:
    """管理工具定义和执行"""

    def __init__(self):
        self.tools_def = []          # 存储工具定义（给模型看）
        self._executors = {}         # 存储实际函数

    def register(self, name: str, description: str, parameters: dict, executor: callable):
        """注册一个工具"""
        # 1. 构建 OpenAI function calling 的 JSON 定义
        tool_def = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }
        self.tools_def.append(tool_def)
        # 2. 绑定执行函数
        self._executors[name] = executor

    def get_definitions(self) -> list[dict]:
        """返回所有工具定义，用于传入模型"""
        return self.tools_def

    def execute(self, tool_name: str, arguments: dict) -> str:
        """执行工具，并返回字符串结果"""
        executor = self._executors.get(tool_name)
        if not executor:
            return f"错误：未找到工具 {tool_name}"
        try:
            result = executor(**arguments)
            return str(result)   # 确保返回字符串
        except Exception as e:
            return f"工具执行错误：{e}"