# Adjacency Matrix Representation
class GraphMatrix:
    def __init__(self, num_vertices):
        # Initialize a VxV matrix with all values set to 0, where V is the number of vertices
        self.matrix = [[0] * num_vertices for _ in range(num_vertices)]
        self.num_vertices = num_vertices

    def add_edge(self, u, v, weight=1):
        # Adds an edge from vertex u to vertex v with the specified weight
        self.matrix[u][v] = weight
        # Uncomment the following line for an undirected graph
        # self.matrix[v][u] = weight

    def __str__(self):
        # Generate a string representation of the matrix for easy viewing
        return '\n'.join([' '.join(map(str, row)) for row in self.matrix])

# Adjacency List Representation
class GraphList:
    def __init__(self, num_vertices):
        # Initialize a list where each element is a list to hold the neighbors of each vertex
        self.list = [[] for _ in range(num_vertices)]

    def add_edge(self, u, v, weight=1):
        # Adds an edge from vertex u to vertex v with the specified weight
        self.list[u].append((v, weight))
        # Uncomment the following line for an undirected graph
        # self.list[v].append((u, weight))

    def __str__(self):
        # Generate a string representation of the adjacency list for easy viewing
        return '\n'.join([f"{i}: {neighbors}" for i, neighbors in enumerate(self.list)])

# Example Usage
num_vertices = 4

# Create a graph using an adjacency matrix
graph_matrix = GraphMatrix(num_vertices)
graph_matrix.add_edge(0, 1)
graph_matrix.add_edge(1, 2)
graph_matrix.add_edge(2, 3)
print("Adjacency Matrix:")
print(graph_matrix)

# Create a graph using an adjacency list
graph_list = GraphList(num_vertices)
graph_list.add_edge(0, 1)
graph_list.add_edge(1, 2)
graph_list.add_edge(2, 3)
print("\nAdjacency List:")
print(graph_list)
