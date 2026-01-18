# search_algorithms.py
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import heapq
from typing import Dict, List, Tuple, Callable, Optional, Set

Node = str
Graph = Dict[Node, List[Tuple[Node, float]]]  # neighbor, cost

def dfs(graph: Graph, start: Node, goal: Node) -> Optional[List[Node]]:
    stack = [(start, [start])]
    visited: Set[Node] = set()
    while stack:
        node, path = stack.pop()
        if node == goal:
            return path
        if node in visited:
            continue
        visited.add(node)
        for nxt, _ in reversed(graph.get(node, [])):
            if nxt not in visited:
                stack.append((nxt, path + [nxt]))
    return None

def bfs(graph: Graph, start: Node, goal: Node) -> Optional[List[Node]]:
    q = deque([(start, [start])])
    visited: Set[Node] = {start}
    while q:
        node, path = q.popleft()
        if node == goal:
            return path
        for nxt, _ in graph.get(node, []):
            if nxt not in visited:
                visited.add(nxt)
                q.append((nxt, path + [nxt]))
    return None

def a_star(
    graph: Graph,
    start: Node,
    goal: Node,
    h: Callable[[Node, Node], float],
) -> Optional[List[Node]]:
    # (f, g, node, path)
    pq = [(h(start, goal), 0.0, start, [start])]
    best_g: Dict[Node, float] = {start: 0.0}
    while pq:
        f, g, node, path = heapq.heappop(pq)
        if node == goal:
            return path
        for nxt, cost in graph.get(node, []):
            ng = g + cost
            if nxt not in best_g or ng < best_g[nxt]:
                best_g[nxt] = ng
                nf = ng + h(nxt, goal)
                heapq.heappush(pq, (nf, ng, nxt, path + [nxt]))
    return None

def ida_star(
    graph: Graph,
    start: Node,
    goal: Node,
    h: Callable[[Node, Node], float],
) -> Optional[List[Node]]:
    bound = h(start, goal)
    path = [start]

    def search(g: float, bound: float) -> Tuple[float, Optional[List[Node]]]:
        node = path[-1]
        f = g + h(node, goal)
        if f > bound:
            return f, None
        if node == goal:
            return f, path.copy()
        min_bound = float("inf")
        for nxt, cost in graph.get(node, []):
            if nxt in path:
                continue
            path.append(nxt)
            t, result = search(g + cost, bound)
            if result is not None:
                return t, result
            if t < min_bound:
                min_bound = t
            path.pop()
        return min_bound, None

    while True:
        t, result = search(0.0, bound)
        if result is not None:
            return result
        if t == float("inf"):
            return None
        bound = t
