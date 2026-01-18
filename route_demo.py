# route_demo.py
from search_algorithms import dfs, bfs, a_star, ida_star

# Grafo conceitual: hubs/zonas
graph = {
    "Dublin_Hub": [("IE_East", 2.0), ("IE_West", 3.5)],
    "IE_East": [("UK_Hub", 4.0), ("FR_Hub", 6.0)],
    "IE_West": [("FR_Hub", 5.0)],
    "UK_Hub": [("DE_Hub", 7.0)],
    "FR_Hub": [("DE_Hub", 4.0)],
    "DE_Hub": []
}

# Heurística simples (estimativa de custo até o goal)
heuristic_map = {
    ("Dublin_Hub", "DE_Hub"): 10.0,
    ("IE_East", "DE_Hub"): 8.0,
    ("IE_West", "DE_Hub"): 8.5,
    ("UK_Hub", "DE_Hub"): 6.0,
    ("FR_Hub", "DE_Hub"): 4.0,
    ("DE_Hub", "DE_Hub"): 0.0,
}
def h(n, goal): 
    return heuristic_map.get((n, goal), 0.0)

start, goal = "Dublin_Hub", "DE_Hub"

print("DFS:", dfs(graph, start, goal))
print("BFS:", bfs(graph, start, goal))
print("A* :", a_star(graph, start, goal, h))
print("IDA*:", ida_star(graph, start, goal, h))
