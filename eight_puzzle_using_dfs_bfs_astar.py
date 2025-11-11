import random, time, heapq, itertools
from collections import deque, defaultdict

# ---------- Puzzle model ----------
GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)
MOVES = {
    0: (1, 3),
    1: (0, 2, 4),
    2: (1, 5),
    3: (0, 4, 6),
    4: (1, 3, 5, 7),
    5: (2, 4, 8),
    6: (3, 7),
    7: (4, 6, 8),
    8: (5, 7)
}

def is_solvable(state):
    arr = [x for x in state if x != 0]
    inv = sum(1 for i in range(len(arr)) for j in range(i + 1, len(arr)) if arr[i] > arr[j])
    return inv % 2 == 0

def neighbors(state):
    z = state.index(0)
    for nb in MOVES[z]:
        s = list(state)
        s[z], s[nb] = s[nb], s[z]
        yield tuple(s), 1  # step cost = 1

def random_start(max_shuffles=60):
    s = list(GOAL)
    z = s.index(0)
    for _ in range(random.randint(20, max_shuffles)):
        nb = random.choice(MOVES[z])
        s[z], s[nb] = s[nb], s[z]
        z = nb
    return tuple(s)


# ---------- Search result container ----------
class Result:
    def __init__(self, found, path_cost, nodes_expanded, depth, secs, path):
        self.found = found
        self.path_cost = path_cost
        self.nodes_expanded = nodes_expanded
        self.depth = depth
        self.secs = secs
        self.path = path


# ---------- Helper for reconstructing path ----------
def reconstruct(parents, s):
    path = [s]
    while s in parents:
        s = parents[s]
        path.append(s)
    path.reverse()
    return path


# ---------- BFS, DFS, and A* Search ----------
def run_search(start, algo="bfs", heuristic=None):
    t0 = time.perf_counter()
    nodes_expanded = 0

    if start == GOAL:
        return Result(True, 0, 0, 0, 0.0, [start])

    # --- BFS ---
    if algo == "bfs":
        Q = deque([start])
        parents = {}
        seen = {start}
        while Q:
            s = Q.popleft()
            nodes_expanded += 1
            for nxt, _ in neighbors(s):
                if nxt in seen:
                    continue
                parents[nxt] = s
                seen.add(nxt)
                if nxt == GOAL:
                    path = reconstruct(parents, nxt)
                    return Result(True, len(path) - 1, nodes_expanded, len(path) - 1, time.perf_counter() - t0, path)
                Q.append(nxt)
        return Result(False, -1, nodes_expanded, -1, time.perf_counter() - t0, [])

    # --- DFS ---
    if algo == "dfs":
        stack = [start]
        parents = {}
        seen = {start}
        while stack:
            s = stack.pop()
            nodes_expanded += 1
            for nxt, _ in neighbors(s):
                if nxt in seen:
                    continue
                parents[nxt] = s
                seen.add(nxt)
                if nxt == GOAL:
                    path = reconstruct(parents, nxt)
                    return Result(True, len(path) - 1, nodes_expanded, len(path) - 1, time.perf_counter() - t0, path)
                stack.append(nxt)
        return Result(False, -1, nodes_expanded, -1, time.perf_counter() - t0, [])

    # --- A* ---
    if algo == "astar":
        g = defaultdict(lambda: float("inf"))
        g[start] = 0
        parents = {}
        cnt = itertools.count()

        def h(s):
            return 0 if heuristic is None else heuristic(s)

        open_heap = [(g[start] + h(start), next(cnt), start)]
        closed = set()

        while open_heap:
            f, _, s = heapq.heappop(open_heap)
            if s in closed:
                continue
            closed.add(s)
            nodes_expanded += 1

            if s == GOAL:
                path = reconstruct(parents, s)
                return Result(True, g[s], nodes_expanded, len(path) - 1, time.perf_counter() - t0, path)

            for nxt, c in neighbors(s):
                tentative = g[s] + c
                if tentative < g[nxt]:
                    g[nxt] = tentative
                    parents[nxt] = s
                    heapq.heappush(open_heap, (tentative + h(nxt), next(cnt), nxt))

        return Result(False, -1, nodes_expanded, -1, time.perf_counter() - t0, [])

    raise ValueError("Unknown algorithm")


# ---------- Heuristics ----------
goal_pos = {v: i for i, v in enumerate(GOAL)}

def manhattan(s):
    dist = 0
    for idx, val in enumerate(s):
        if val == 0:
            continue
        gi = goal_pos[val]
        r1, c1 = divmod(idx, 3)
        r2, c2 = divmod(gi, 3)
        dist += abs(r1 - r2) + abs(c1 - c2)
    return dist

def linear_conflict(s):
    dist = manhattan(s)
    # Rows
    for r in range(3):
        row = s[3*r:3*r+3]
        goal_row = [1+3*r, 2+3*r, 3+3*r]
        tiles = [x for x in row if x in goal_row]
        for i in range(len(tiles)):
            for j in range(i+1, len(tiles)):
                if goal_row.index(tiles[i]) > goal_row.index(tiles[j]):
                    dist += 2
    # Columns
    for c in range(3):
        col = s[c::3]
        goal_col = [c+1, c+4, c+7]
        tiles = [x for x in col if x in goal_col]
        for i in range(len(tiles)):
            for j in range(i+1, len(tiles)):
                if goal_col.index(tiles[i]) > goal_col.index(tiles[j]):
                    dist += 2
    return dist


# ---------- Experiments ----------
def run_experiments(num=3, seed=7):
    random.seed(seed)
    starts = []
    while len(starts) < num:
        s = random_start()
        if is_solvable(s) and s not in starts:
            starts.append(s)

    rows = []
    for i, s in enumerate(starts, 1):
        for algo in ["bfs", "dfs"]:
            res = run_search(s, algo=algo)
            rows.append(("Case" + str(i), algo.upper(), res.path_cost, res.nodes_expanded, res.secs))
        for name, h in [("A*-Manhattan", manhattan), ("A*-LinearConflict", linear_conflict)]:
            res = run_search(s, algo="astar", heuristic=h)
            rows.append(("Case" + str(i), name, res.path_cost, res.nodes_expanded, res.secs))
    return rows


# ---------- Main ----------
if __name__ == "__main__":
    from pprint import pprint
    rows = run_experiments()
    print("Case, Method, PathCost, NodesExpanded, Time(s)")
    for r in rows:
        print(",".join([str(x) for x in r]))

    print("\nHeuristic Notes:")
    print("- Manhattan: admissible & consistent.")
    print("- Linear Conflict: Manhattan + 2 per row/col conflict, still admissible.")
