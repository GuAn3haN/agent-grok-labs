import time
import asyncio
import pytest # type: ignore
from src.graph import StateGraph
from src.checkpoint import MemorySaver
from src.exceptions import InterruptError

def test_linear_graph():
    def a(s): return {"a": 1}
    def b(s): return {"b": 2}
    def c(s): return {"c": 3}
    g = StateGraph()
    g.add_node("a", a); g.add_node("b", b); g.add_node("c", c)
    g.set_entry_point("a")
    g.add_edge("a", "b"); g.add_edge("b", "c")
    app = g.compile()
    assert app.invoke({}) == {"a": 1, "b": 2, "c": 3}

def test_conditional_branching():
    def start(s): return s
    def double(s): return {"value": s["value"]*2, "path": "double"}
    def inc(s): return {"value": s["value"]+1, "path": "inc"}
    def route(s): return "double" if s["value"] > 0 else "inc"
    g = StateGraph()
    g.add_node("start", start); g.add_node("double", double); g.add_node("inc", inc)
    g.set_entry_point("start")
    g.add_conditional_edges("start", route, {"double": "double", "inc": "inc"})
    app = g.compile()
    assert app.invoke({"value": 2}) == {"value": 4, "path": "double"}
    assert app.invoke({"value": -1}) == {"value": 0, "path": "inc"}

def test_loop():
    call_counter = {"cnt": 0}
    def acc(state):
        call_counter["cnt"] += 1
        return {"total": state["total"] + state.get("inc", 2)}
    def should_continue(state):
        return "accumulate" if state["total"] < 5 else "__end__"
    g = StateGraph()
    g.add_node("accumulate", acc)
    g.set_entry_point("accumulate")
    g.add_conditional_edges("accumulate", should_continue,
                            {"accumulate": "accumulate", "__end__": None})
    app = g.compile()
    result = app.invoke({"total": 1, "inc": 3})
    assert result["total"] == 7
    assert call_counter["cnt"] == 2

def test_parallel_execution():
    def split(s): return s
    def ta(s):
        time.sleep(0.1)
        return {"a": s.get("x",0)+1}
    def tb(s):
        time.sleep(0.1)
        return {"b": s.get("x",0)+2}
    def merge(s): return s
    g = StateGraph()
    g.add_node("split", split); g.add_node("ta", ta); g.add_node("tb", tb); g.add_node("merge", merge)
    g.set_entry_point("split")
    g.add_edge("split", "ta"); g.add_edge("split", "tb")
    g.add_edge("ta", "merge"); g.add_edge("tb", "merge")
    app = g.compile()
    start = time.perf_counter()
    res = app.invoke({"x": 10})
    elapsed = time.perf_counter() - start
    assert res == {"x": 10, "a": 11, "b": 12}
    assert elapsed < 0.15, f"Not parallel: {elapsed:.2f}s"

def test_stream():
    def s1(s): return {"step": 1}
    def s2(s): return {"step": 2}
    def s3(s): return {"step": 3}
    g = StateGraph()
    g.add_node("s1", s1); g.add_node("s2", s2); g.add_node("s3", s3)
    g.set_entry_point("s1")
    g.add_edge("s1", "s2"); g.add_edge("s2", "s3")
    app = g.compile()
    states = list(app.stream({}))
    assert len(states) == 3
    assert states[-1] == {"step": 3}

def test_thread_isolation():
    def counter(state):
        c = state.get("counter", 0) + 1
        return {"counter": c}
    g = StateGraph()
    g.add_node("count", counter)
    g.set_entry_point("count")
    g.add_edge("count", "__end__")
    app = g.compile(checkpointer=MemorySaver())
    c1 = {"configurable": {"thread_id": "t1"}}
    c2 = {"configurable": {"thread_id": "t2"}}
    assert app.invoke({"counter": 0}, c1)["counter"] == 1
    assert app.invoke({"counter": 10}, c2)["counter"] == 11
    assert app.invoke({}, c1)["counter"] == 2

def test_interrupt_and_resume():
    def step1(state):
        if "approved" not in state:
            raise InterruptError("Need approval")
        return {"step1_done": True}
    def step2(state):
        return {"step2_done": True}
    g = StateGraph()
    g.add_node("step1", step1); g.add_node("step2", step2)
    g.set_entry_point("step1")
    g.add_edge("step1", "step2")
    app = g.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "irq1"}}
    with pytest.raises(InterruptError):
        app.invoke({}, config)
    result = app.invoke({"approved": True}, config)
    assert result == {"approved": True, "step1_done": True, "step2_done": True}

@pytest.mark.asyncio
async def test_async_equivalence():
    async def a(s):
        await asyncio.sleep(0.01)
        return {"a": "async"}
    def b(s): return {"b": "works"}
    g = StateGraph()
    g.add_node("a", a); g.add_node("b", b)
    g.set_entry_point("a"); g.add_edge("a", "b")
    app = g.compile()
    sync_res = app.invoke({})
    async_res = await app.ainvoke({})
    assert sync_res == async_res == {"a": "async", "b": "works"}