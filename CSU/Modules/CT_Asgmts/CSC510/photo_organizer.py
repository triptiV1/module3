import heapq

class PuzzleState:
    def __init__(self, board, empty_tile, moves=0, previous=None):
        self.board = board
        self.empty_tile = empty_tile
        self.moves = moves
        self.previous = previous
        self.size = len(board)

    def is_goal(self, goal_state):
        return self.board == goal_state

    def neighbors(self):
        neighbors = []
        x, y = self.empty_tile
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                new_board = [row[:] for row in self.board]
                new_board[x][y], new_board[nx][ny] = new_board[nx][ny], new_board[x][y]
                neighbors.append(PuzzleState(new_board, (nx, ny), self.moves + 1, self))

        return neighbors

    def manhattan_distance(self, goal_state):
        distance = 0
        for x in range(self.size):
            for y in range(self.size):
                value = self.board[x][y]
                if value != 0:
                    goal_x, goal_y = divmod(goal_state.index(value), self.size)
                    distance += abs(goal_x - x) + abs(goal_y - y)
        return distance

    def __lt__(self, other):
        return self.moves < other.moves

def a_star_search(initial_state, goal_state):
    open_list = []
    heapq.heappush(open_list, (initial_state.manhattan_distance(goal_state), initial_state))
    closed_set = set()

    while open_list:
        _, current = heapq.heappop(open_list)

        if current.is_goal(goal_state):
            return reconstruct_path(current)

        closed_set.add(tuple(map(tuple, current.board)))

        for neighbor in current.neighbors():
            if tuple(map(tuple, neighbor.board)) not in closed_set:
                f_cost = neighbor.moves + neighbor.manhattan_distance(goal_state)
                heapq.heappush(open_list, (f_cost, neighbor))

    return None

def reconstruct_path(state):
    path = []
    while state:
        path.append(state)
        state = state.previous
    return path[::-1]

def print_solution(path):
    for state in path:
        for row in state.board:
            print(row)
        print()

def parse_board(input_str):
    board = []
    lines = input_str.strip().split('\n')
    for line in lines:
        board.append(list(map(int, line.split())))
    return board

def is_solvable(board):
    flat_board = sum(board, [])
    inversions = 0
    for i in range(len(flat_board)):
        for j in range(i + 1, len(flat_board)):
            if flat_board[i] > flat_board[j] != 0:
                inversions += 1
    return inversions % 2 == 0

# Interactive part
if __name__ == "__main__":
    print("Enter the initial board state (3x3 grid, space-separated numbers, use 0 for the empty tile):")
    initial_input = "\n".join([input() for _ in range(3)])
    initial_board = parse_board(initial_input)

    print("Enter the goal board state (3x3 grid, space-separated numbers, use 0 for the empty tile):")
    goal_input = "\n".join([input() for _ in range(3)])
    goal_board = parse_board(goal_input)

    # Find the position of the empty tile in the initial state
    empty_tile = None
    for i in range(3):
        for j in range(3):
            if initial_board[i][j] == 0:
                empty_tile = (i, j)
                break
        if empty_tile is not None:
            break

    if not is_solvable(initial_board):
        print("The initial board state is not solvable.")
    else:
        initial_state = PuzzleState(initial_board, empty_tile)
        solution_path = a_star_search(initial_state, sum(goal_board, []))

        if solution_path:
            print("Solution found:")
            print_solution(solution_path)
        else:
            print("No solution found")
