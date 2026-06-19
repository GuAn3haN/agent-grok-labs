# src/skill.py
from src.toolbox import Toolbox

class Skill:
    def __init__(self, name: str, description: str, parameters: dict):
        self.name = name
        self.description = description
        self.parameters = parameters
        self._internal_toolbox = Toolbox()
        self._executor = None

    def set_executor(self, executor: callable):
        """
        executor 签名: def execute(params: dict, toolbox: Toolbox) -> str
        """
        self._executor = executor

    def run(self, params: dict) -> str:
        if not self._executor:
            return "错误：Skill 未配置执行逻辑"
        return self._executor(params, self._internal_toolbox)

    def to_tool_definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }