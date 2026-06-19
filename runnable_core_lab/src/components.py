# src/components.py
import time
from .runnable import Runnable

class PromptTemplate(Runnable):
    input_type = dict
    output_type = str
    def __init__(self, template: str):
        super().__init__()
        self.template = template
        self._tracer = None   # 添加这一行，确保属性存在

    def invoke(self, input: dict, config=None) -> str:
        if self._tracer:
            self._tracer.add("PromptTemplate.invoke")
        return self.template.format(**input)
    
class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content
        self._tracer = None   # 添加这一行，确保属性存在


class FakeLLM(Runnable):
    input_type = str
    output_type = FakeLLMResponse
    def __init__(self, responses: dict | None = None, delay: float = 0.0):
        super().__init__()
        self.responses = responses if responses is not None else {}
        self.delay = delay

    def invoke(self, input: str, config=None) -> FakeLLMResponse:
    # 如果有预定义映射，直接使用；否则回退到默认回显
        if self._tracer:
            self._tracer.add("FakeLLM.invoke")
        if input in self.responses:
            full_response = self.responses[input]
        else:
            full_response = f"Echo: {input}"
    # 模拟延迟（可选）
        if self.delay > 0:
            time.sleep(self.delay)   
        return FakeLLMResponse(content=full_response)

def stream(self, input: str, config=None):
    # 1. 确定完整响应内容
    if input in self.responses:
        full_response = self.responses[input]
    else:
        full_response = f"Echo: {input}"

    # 2. 逐字符产出（注意缩进，必须在函数内部）
    for char in full_response:
        if self.delay > 0:
            time.sleep(self.delay)
        yield FakeLLMResponse(content=char)



class RunnableLambda(Runnable):
    input_type = object
    output_type = object
    def __init__(self, func):
        super().__init__()        
        self.func = func
        self._tracer = None   # 添加这一行，确保属性存在
    def invoke(self, input, config=None):
        if self._tracer:
            self._tracer.add("RunnableLambda.invoke")
        return self.func(input)