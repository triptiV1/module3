import heapq

class Node:
    def __init__(self, position, parent=None, g=0, h=0):
        self.position = position
        self.parent = parent
        self.g = g  # Cost from start to node
        self.h = h  # Estimated cost from node to goal
        self.f = g + h  # Total cost

    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(start, goal, grid):
    open_list = []
    heapq.heappush(open_list, Node(start, None, 0, heuristic(start, goal)))
    closed_list = set()

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)

        if current_node.position == goal:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # Return reversed path

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Adjacent squares
            neighbor = (current_node.position[0] + dx, current_node.position[1] + dy)
            if neighbor in closed_list or not (0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0])) or grid[neighbor[0]][neighbor[1]] == 1:
                continue

            g = current_node.g + 1
            h = heuristic(neighbor, goal)
            neighbor_node = Node(neighbor, current_node, g, h)

            if any(open_node.position == neighbor_node.position and open_node.f <= neighbor_node.f for open_node in open_list):
                continue

            heapq.heappush(open_list, neighbor_node)

    return None  # No path found

def create_grid(rows, cols, obstacles):
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    for obstacle in obstacles:
        x, y = obstacle
        grid[x][y] = 1
    return grid

def get_valid_int(prompt):
    while True:
        try:
            value = int(input(prompt))
            return value
        except ValueError:
            print("Invalid input. Please enter an integer.")

def get_valid_position(prompt, max_row, max_col):
    while True:
        try:
            x, y = map(int, input(prompt).split())
            if 0 <= x < max_row and 0 <= y < max_col:
                return (x, y)
            else:
                print(f"Position out of bounds. Please enter values between 0 and {max_row-1} for rows, and 0 and {max_col-1} for columns.")
        except ValueError:
            print("Invalid input. Please enter two integers separated by a space.")

def main():
    rows = get_valid_int("Enter the number of rows for the grid: ")
    cols = get_valid_int("Enter the number of columns for the grid: ")

    start = get_valid_position("Enter the start position as two integers separated by a space (e.g., '0 0'): ", rows, cols)
    goal = get_valid_position("Enter the goal position as two integers separated by a space (e.g., '2 2'): ", rows, cols)

    num_obstacles = get_valid_int("Enter the number of obstacles: ")
    obstacles = []
    for _ in range(num_obstacles):
        obstacle = get_valid_position("Enter obstacle position as two integers separated by a space: ", rows, cols)
        if obstacle == start or obstacle == goal:
            print("Obstacle cannot be placed at the start or goal position. Please enter a different position.")
        else:
            obstacles.append(obstacle)

    grid = create_grid(rows, cols, obstacles)

    path = a_star_search(start, goal, grid)
    if path:
        print("Path found:", path)
    else:
        print("No path found.")

if __name__ == "__main__":
    main()
