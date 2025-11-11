from collections import deque
from itertools import product

# 4x4 coordinates: (1..4, 1..4)
DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def in_bounds(x, y):
    return 1 <= x <= 4 and 1 <= y <= 4


def neighbors(x, y):
    for dx, dy in DIRS:
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny):
            yield (nx, ny)


class KB:
    """
    Lightweight propositional KB for standard pit/breeze rules:
    - ¬B(x,y) ⇒ all neighbors are ¬Pit
    - B(x,y) ⇒ at least one neighbor is Pit (we keep {possibly_pit} set)
    - If a neighbor becomes known-safe, remove from possible pits
    - If only one 'possible pit' remains around a breezy cell ⇒ that cell is Pit

    This is not a full SAT solver, but enough to infer safe cells and
    isolate pits in many layouts, and it logs derived sentences.
    """

    def __init__(self):
        self.safe = set()          # proven safe cells
        self.pit = set()           # proven pits
        self.possible = {}         # (x,y) with breeze -> set of candidate pit cells among its neighbors
        self.logs = []

    def assert_no_breeze(self, x, y):
        # infer all neighbors are safe
        for n in neighbors(x, y):
            if n not in self.safe and n not in self.pit:
                self.safe.add(n)
                self.logs.append(f"From ¬B({x},{y}) infer SAFE{n}")

    def assert_breeze(self, x, y):
        if (x, y) not in self.possible:
            cands = {n for n in neighbors(x, y) if n not in self.safe and n not in self.pit}
            self.possible[(x, y)] = cands
            self.logs.append(f"B({x},{y}) ⇒ at least one of {sorted(cands)} is a Pit")

    def mark_safe(self, cell):
        if cell in self.safe:
            return
        self.safe.add(cell)
        # remove from all candidate sets
        for k in list(self.possible):
            if cell in self.possible[k]:
                self.possible[k].remove(cell)
                self.logs.append(f"Since {cell} is SAFE, remove from pit-candidates of {k}")

    def deduce(self):
        changed = True
        while changed:
            changed = False
            # If any breezy cell has all but one neighbor eliminated ⇒ that remaining neighbor is a Pit
            for k, cands in list(self.possible.items()):
                cands = {c for c in cands if c not in self.safe and c not in self.pit}
                self.possible[k] = cands
                if len(cands) == 1:
                    p = next(iter(cands))
                    if p not in self.pit:
                        self.pit.add(p)
                        changed = True
                        self.logs.append(f"Only {p} fits B{tuple(k)} ⇒ PIT{p}")
                elif len(cands) == 0:
                    # contradiction handled softly: breeze must have been explained elsewhere; ignore
                    pass
        # No-breeze cells make all neighbors safe handled externally


class World:
    """
    Simple world with:
    - exactly one Wumpus (we avoid shooting; goal is safety inference)
    - some pits
    - one gold cell
    Agent perceives Breeze (near pits) and Stench (near Wumpus). We use Breeze for pit inference.
    """

    def __init__(self, pits={(3, 1)}, wumpus=(4, 3), gold=(2, 3)):
        self.pits = set(pits)
        self.wumpus = wumpus
        self.gold = gold

    def percept(self, x, y):
        breeze = any(n in self.pits for n in neighbors(x, y))
        stench = any(n == self.wumpus for n in neighbors(x, y))
        glitter = (x, y) == self.gold
        return {"breeze": breeze, "stench": stench, "glitter": glitter}


def agent_run(world, start=(1, 1)):
    kb = KB()
    path = [start]
    visited = set([start])
    have_gold = False

    def process_cell(c):
        nonlocal have_gold
        x, y = c
        p = world.percept(x, y)
        if p["glitter"]:
            have_gold = True
            kb.logs.append(f"GLITTER at {c} ⇒ grab gold")
        if p["breeze"]:
            kb.assert_breeze(x, y)
        else:
            kb.assert_no_breeze(x, y)
        kb.mark_safe(c)
        kb.deduce()

    process_cell(start)

    # Explore: always move to any neighbor known safe & unvisited; else stop and exit
    while True:
        options = [n for n in neighbors(*path[-1]) if n in kb.safe and n not in visited]
        if not options:
            break
        nxt = sorted(options)[0]
        path.append(nxt)
        visited.add(nxt)
        process_cell(nxt)
        if have_gold and nxt == (1, 1):
            break  # already at exit

        # If we grabbed gold, head back to (1,1) via safe cells greedily
        if have_gold and nxt != (1, 1):
            def manhattan(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])
            current = nxt
            while current != (1, 1):
                cands = [n for n in neighbors(*current) if n in kb.safe]
                if not cands:
                    break
                best = min(cands, key=lambda t: (manhattan(t, (1, 1)), t))
                path.append(best)
                visited.add(best)
                current = best
            break

    return {
        "path": path,
        "have_gold": have_gold,
        "safe": sorted(kb.safe),
        "pits_inferred": sorted(kb.pit),
        "log": kb.logs
    }


if __name__ == "__main__":
    world = World(pits={(3, 1), (3, 3)}, wumpus=(4, 3), gold=(2, 3))
    out = agent_run(world)
    print("PATH:", out["path"])
    print("GOT GOLD:", out["have_gold"])
    print("SAFE:", out["safe"])
    print("INFERRED PITS:", out["pits_inferred"])
    print("\nDerived sentences:")
    for line in out["log"]:
        print("-", line)
