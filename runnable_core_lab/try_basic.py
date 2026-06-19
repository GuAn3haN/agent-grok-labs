from src.components import PromptTemplate, FakeLLM
from src.components import RunnableLambda
import time
from src.tracer import Tracer

import asyncio
async def main():
    prompt = PromptTemplate("Say {word}")
    llm = FakeLLM(responses={"Say async": "ASYNC WORKS"})
    chain = prompt | llm | RunnableLambda(lambda r: r.content)
    result = await chain.ainvoke({"word": "async"})
    print(result)

asyncio.run(main())