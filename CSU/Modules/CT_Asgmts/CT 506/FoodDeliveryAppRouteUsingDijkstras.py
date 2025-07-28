import heapq

def dijkstra(graph, start):
    # Initialize the distances dictionary with all nodes set to infinity
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    visited = set()

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_node in visited:
            continue

        visited.add(current_node)

        for neighbor, weight in graph.get(current_node, {}).items():
            if neighbor not in distances:
                distances[neighbor] = float('inf')
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances

def get_graph_from_user():
    graph = {}
    try:
        num_nodes = int(input("Enter the number of nodes in the graph: "))
        print("Enter each node and its connections in the format 'Node Neighbor1:Weight1 Neighbor2:Weight2 ...':")
        
        for _ in range(num_nodes):
            entry = input().strip()
            if entry:
                parts = entry.split()
                node = parts[0]
                edges = parts[1:]
                connections = {}
                for edge in edges:
                    neighbor, weight = edge.split(':')
                    connections[neighbor] = int(weight)
                graph[node] = connections
        
    except ValueError:
        print("Invalid input, please enter integers for the number of nodes and weights.")
    return graph

def main():
    graph = get_graph_from_user()
    start_node = input("Enter the starting node: ").strip()
    
    # Ensure all nodes are initialized in the graph
    for node, connections in graph.items():
        for neighbor in connections:
            if neighbor not in graph:
                graph[neighbor] = {}

    if start_node not in graph:
        print(f"Error: Starting node '{start_node}' is not in the graph.")
        return
    
    shortest_paths = dijkstra(graph, start_node)
    print(f"Shortest paths from node {start_node}:", shortest_paths)

if __name__ == "__main__":
    main()
