import asyncio
from typing import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor

from src.exceptions import InterruptError


class StateGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.conditional_edges = {}
        self.entry_point = None

    def add_node(self, name: str, func: Callable[[dict], dict]) -> None:
        self.nodes[name] = func

    def add_edge(self, start: str, end: str) -> None:
        self.edges.append((start, end))

    def add_conditional_edges(self, start: str,
                              condition: Callable[[dict], str],
                              mapping: dict[str, str]) -> None:
        self.conditional_edges[start] = (condition, mapping)

    def set_entry_point(self, name: str) -> None:
        self.entry_point = name

    def compile(self, checkpointer=None) -> 'CompiledGraph':
        if self.entry_point is None:
            raise ValueError("Entry point not set")
        if self.entry_point not in self.nodes:
            raise ValueError(f"Entry point '{self.entry_point}' is not a valid node")
        return CompiledGraph(
            nodes=self.nodes.copy(),
            edges=list(self.edges),
            conditional_edges=dict(self.conditional_edges),
            entry_point=self.entry_point,
            checkpointer=checkpointer,
        )


class CompiledGraph:
    def __init__(self, nodes, edges, conditional_edges, entry_point, checkpointer=None):
        self.nodes = nodes
        self.edges = edges
        self.conditional_edges = conditional_edges
        self.entry_point = entry_point
        self.checkpointer = checkpointer

    def invoke(self, state: dict, config: dict = None) -> dict:
        remaining_after_interrupt = None
        if self.checkpointer and config:
            saved = self.checkpointer.get(config)
            if saved is not None:
                if "remaining" in saved:
                    state = {**saved["state"], **state}
                    remaining_after_interrupt = saved["remaining"]
                else:
                    state = {**saved, **state}

        if remaining_after_interrupt:
            current_nodes = remaining_after_interrupt
        else:
            current_nodes = [self.entry_point]

        while current_nodes:
            state_before = dict(state)

            def run_node(name):
                func = self.nodes[name]
                node_state = dict(state_before)
                if asyncio.iscoroutinefunction(func):
                    # 同步环境中执行异步节点
                    return asyncio.run(func(node_state))
                else:
                    return func(node_state)

            try:
                with ThreadPoolExecutor() as executor:
                    results = list(executor.map(run_node, current_nodes))
            except InterruptError:
                if self.checkpointer and config:
                    self.checkpointer.put(config, {
                        "state": state_before,
                        "remaining": list(current_nodes)
                    })
                raise

            for update in results:
                state = {**state, **update}

            next_nodes_set = set()
            for node in current_nodes:
                if node in self.conditional_edges:
                    condition, mapping = self.conditional_edges[node]
                    route_key = condition(state)
                    target = mapping.get(route_key)
                    if target and target != "__end__":
                        next_nodes_set.add(target)
                else:
                    for start, end in self.edges:
                        if start == node and end != "__end__":
                            next_nodes_set.add(end)
            current_nodes = list(next_nodes_set)

        if self.checkpointer and config:
            self.checkpointer.put(config, dict(state))

        return state

    def stream(self, state: dict, config: dict = None) -> Iterator[dict]:
        remaining_after_interrupt = None
        if self.checkpointer and config:
            saved = self.checkpointer.get(config)
            if saved is not None:
                if "remaining" in saved:
                    state = {**saved["state"], **state}
                    remaining_after_interrupt = saved["remaining"]
                else:
                    state = {**saved, **state}

        if remaining_after_interrupt:
            current_nodes = remaining_after_interrupt
        else:
            current_nodes = [self.entry_point]

        while current_nodes:
            state_before = dict(state)

            def run_node(name):
                func = self.nodes[name]
                node_state = dict(state_before)
                if asyncio.iscoroutinefunction(func):
                    return asyncio.run(func(node_state))
                else:
                    return func(node_state)

            try:
                with ThreadPoolExecutor() as executor:
                    results = list(executor.map(run_node, current_nodes))
            except InterruptError:
                if self.checkpointer and config:
                    self.checkpointer.put(config, {
                        "state": state_before,
                        "remaining": list(current_nodes)
                    })
                raise

            for update in results:
                state = {**state, **update}
                yield dict(state)

            next_nodes_set = set()
            for node in current_nodes:
                if node in self.conditional_edges:
                    condition, mapping = self.conditional_edges[node]
                    route_key = condition(state)
                    target = mapping.get(route_key)
                    if target and target != "__end__":
                        next_nodes_set.add(target)
                else:
                    for start, end in self.edges:
                        if start == node and end != "__end__":
                            next_nodes_set.add(end)
            current_nodes = list(next_nodes_set)

        if self.checkpointer and config:
            self.checkpointer.put(config, dict(state))

    async def ainvoke(self, state: dict, config: dict = None) -> dict:
        remaining_after_interrupt = None
        if self.checkpointer and config:
            saved = self.checkpointer.get(config)
            if saved is not None:
                if "remaining" in saved:
                    state = {**saved["state"], **state}
                    remaining_after_interrupt = saved["remaining"]
                else:
                    state = {**saved, **state}

        if remaining_after_interrupt:
            current_nodes = remaining_after_interrupt
        else:
            current_nodes = [self.entry_point]

        while current_nodes:
            state_before = dict(state)

            async def run_node(name):
                func = self.nodes[name]
                node_state = dict(state_before)
                if asyncio.iscoroutinefunction(func):
                    return await func(node_state)
                else:
                    return func(node_state)

            try:
                tasks = [run_node(name) for name in current_nodes]
                results = await asyncio.gather(*tasks)
            except InterruptError:
                if self.checkpointer and config:
                    self.checkpointer.put(config, {
                        "state": state_before,
                        "remaining": list(current_nodes)
                    })
                raise

            for update in results:
                state = {**state, **update}

            next_nodes_set = set()
            for node in current_nodes:
                if node in self.conditional_edges:
                    condition, mapping = self.conditional_edges[node]
                    route_key = condition(state)
                    target = mapping.get(route_key)
                    if target and target != "__end__":
                        next_nodes_set.add(target)
                else:
                    for start, end in self.edges:
                        if start == node and end != "__end__":
                            next_nodes_set.add(end)
            current_nodes = list(next_nodes_set)

        if self.checkpointer and config:
            self.checkpointer.put(config, dict(state))

        return state