# src/runnable.py
from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator, List, Generic, TypeVar, Optional

from src.tracer import Tracer

# --- 1. 定义泛型变量（类型安全的基石）---
# Input 是“输入类型”，Output 是“输出类型”
# contravariant=True 表示输入类型支持逆变（子类可以接受更宽泛的输入）
# covariant=True 表示输出类型支持协变（子类可以返回更具体的输出）
Input = TypeVar("Input", contravariant=True)
Output = TypeVar("Output", covariant=True)


# --- 2. 抽象基类 Runnable ---
class Runnable(ABC, Generic[Input, Output]):
    def __init__(self):
        self._tracer = None

    def with_tracer(self, tracer: "Tracer") -> "Runnable[Input, Output]":
        self._tracer = tracer
        return self
    
    async def ainvoke(self, input: Input, config: Optional[dict] = None) -> Output:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.invoke, input, config)
    """
    所有可运行组件的抽象基类。
    子类必须实现 invoke 方法。
    """

    @abstractmethod
    def invoke(self, input: Input, config: Optional[dict] = None) -> Output:
        """同步执行核心逻辑，必须被子类重写。"""
        pass

    def stream(self, input: Input, config: Optional[dict] = None) -> Iterator[Output]:
        """
        流式执行，返回一个生成器。
        默认实现：调用 invoke 并 yield 单个结果。
        如果子类（如 FakeLLM）需要真正的流式输出，必须重写此方法。
        """
        yield self.invoke(input, config)

    def batch(self, inputs: List[Input], config: Optional[dict] = None) -> List[Output]:
        with ThreadPoolExecutor() as executor:
            # 注意：lambda 需要捕获 config，否则 config 不会传递
            return list(executor.map(lambda x: self.invoke(x, config), inputs))
    def __or__(self, other: "Runnable[Output, Any]") -> "RunnableSequence[Input, Any]":
        """
        重载 | 操作符，将当前 Runnable 与另一个 Runnable 串联。
        返回一个 RunnableSequence 对象，它也是一个 Runnable。
        """
        return RunnableSequence(self, other)


# --- 3. 管道序列类 RunnableSequence ---
class RunnableSequence(Runnable[Input, Output]):
    """
    将多个 Runnable 串联起来，上一个的输出作为下一个的输入。
    """

    def __init__(self, first: Runnable, last: Runnable):
        # 内部用元组存储步骤，后续可以扩展为支持多个步骤
        self.steps = (first, last)
        # 推导类型
        self.input_type = getattr(first, 'input_type', object)
        self.output_type = getattr(last, 'output_type', object)

    def invoke(self, input: Input, config: Optional[dict] = None) -> Output:
        data = input
        for step in self.steps:
            data = step.invoke(data, config)
        return data

    def stream(self, input: Input, config: Optional[dict] = None) -> Iterator[Output]:
        # 管道流式比较复杂，我们先给出一个简化的实现：
        # 如果最后一步是流式组件，则直接委托给它；否则回退到 invoke。
        # 注意：真实 LangChain 会处理中间步骤的流式传递，但实验初期我们不做那么复杂。
        last_step = self.steps[-1]
        # 先执行前面的步骤（非流式）
        intermediate = input
        for step in self.steps[:-1]:
            intermediate = step.invoke(intermediate, config)
        # 最后一步用流式输出
        yield from last_step.stream(intermediate, config)

    def batch(self, inputs: List[Input], config: Optional[dict] = None) -> List[Output]:
        # 直接复用基类并发逻辑
        return super().batch(inputs, config)