# AI Assignments (Compact Reference Implementations)

## 1) State-space Search (8-Puzzle)

- **File**: `eight_puzzle.py`
- Implements **BFS**, **DFS**, **UCS**, and **A\*** (with **Manhattan** and **Linear-Conflict** heuristics).
- Runs 5 random solvable start states, prints path cost, nodes expanded, and wall time.
- Heuristic notes included (admissible + consistent).
  **Run**

```bash
python eight_puzzle.py
```

## 2) Heuristic Search Plots

`eight_puzzle.py` also prints averages for UCS vs A* (Manhattan & Linear-Conflict). (If you want matplotlib plots, you can easily add a small bar chart using the printed averages.)

## 3) Logical Agent: Wumpus World (4x4)

- **File**: `wumpus_world.py`
- A propositional KB with classic rules:

  - `¬B(x,y) ⇒ all neighbors safe`
  - `B(x,y) ⇒ some neighbor is pit` + candidate shrinking; singleton ⇒ pit
- Agent explores only proven-safe cells, logs derived sentences, grabs gold if found, and goes back to (1,1).

  **Run**

```bash
python wumpus_world.py
```

## 4) End-to-End ML Pipeline (Regression)

- **File**: `ml_regression.py`
- Dataset: California Housing from scikit-learn (auto-fetch). If download is blocked, it falls back to the built-in Diabetes dataset.
- Steps: tiny **EDA → train/test split → preprocessing (StandardScaler) → Linear Regression → RMSE/MAE**→ export `model_card.json` and two small PNG charts.

  **Run**

```bash
python ml_regression.py
```


## Environment
`python >= 3.9`
`pip install numpy matplotlib scikit-learn`

---

`### Notes on what to turn in`
`- 8-puzzle: include the console table of results (5 cases × {BFS, DFS, UCS, A*-Manh, A*-LC}) and a short paragraph interpreting why A* dominates UCS on nodes/time.`
`- Heuristics: add the admissibility/consistency sketch provided above.`
`- Wumpus: paste the **Derived sentences** log to show the KB reasoning steps that led to safe cells and pit identification.`
`- Regression: attach model_card.json plus the reported **RMSE/MAE** and the two figures (target_hist.png, true_vs_pred.png).`