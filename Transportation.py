import numpy as np

def get_penalties(grid):
    """Calculates penalty differences for VAM"""
    rows, cols = grid.shape
    r_pen, c_pen = [], []
    
    for i in range(rows):
        valid = grid[i, grid[i, :] != np.inf]
        if len(valid) > 1:
            s = np.sort(valid)
            r_pen.append(s[1] - s[0])
        elif len(valid) == 1:
            r_pen.append(valid[0])
        else:
            r_pen.append(-1)
            
    for j in range(cols):
        valid = grid[grid[:, j] != np.inf, j]
        if len(valid) > 1:
            s = np.sort(valid)
            c_pen.append(s[1] - s[0])
        elif len(valid) == 1:
            c_pen.append(valid[0])
        else:
            c_pen.append(-1)
            
    return r_pen, c_pen

def solve_transportation():
    """
    =========================================================
    PROBLEM STATEMENT (Transportation Case Study):
    Find the optimal shipment plan to minimize total cost.
    
    Supply Capacities (Sources S1, S2, S3):
      S1 = 30, S2 = 50, S3 = 20
      
    Destination Demands (Destinations D1, D2, D3, D4):
      D1 = 20, D2 = 40, D3 = 30, D4 = 10
      
    Transportation Cost Matrix (C_ij):
           D1  D2  D3  D4
      S1 [  1,  2,  1,  4 ]
      S2 [  3,  3,  2,  1 ]
      S3 [  4,  2,  5,  9 ]
    =========================================================
    """
    
    supply = np.array([30, 50, 20], dtype=float)
    demand = np.array([20, 40, 30, 10], dtype=float)
    costs = np.array([
        [1, 2, 1, 4],
        [3, 3, 2, 1],
        [4, 2, 5, 9]
    ], dtype=float)
    
    print("\n[Step 1] Vogel's Approximation Method (VAM)")
    working_costs = costs.copy()
    s = supply.copy()
    d = demand.copy()
    allocation = np.zeros(costs.shape)
    
    while np.sum(s) > 0 and np.sum(d) > 0:
        r_pen, c_pen = get_penalties(working_costs)
        max_r, max_c = np.max(r_pen), np.max(c_pen)
        
        if max_r >= max_c:
            r_idx = np.argmax(r_pen)
            c_idx = np.argmin(working_costs[r_idx])
        else:
            c_idx = np.argmax(c_pen)
            r_idx = np.argmin(working_costs[:, c_idx])
            
        qty = np.min([s[r_idx], d[c_idx]])
        allocation[r_idx, c_idx] = qty
        s[r_idx] -= qty
        d[c_idx] -= qty
        
        if s[r_idx] == 0: working_costs[r_idx, :] = np.inf
        if d[c_idx] == 0: working_costs[:, c_idx] = np.inf
            
    vam_total = np.sum(allocation * costs)
    print("Initial Basic Feasible Solution (BFS) Matrix:\n", allocation)
    print(f"VAM Initial Cost: {vam_total}")

    print("\n[Step 2] MODI Method Optimality Test")
    rows, cols = costs.shape
    u = np.full(rows, np.nan)
    v = np.full(cols, np.nan)
    u[0] = 0.0  # Set u1 = 0
    
    # Compute u and v for occupied cells
    while np.isnan(u).any() or np.isnan(v).any():
        for i in range(rows):
            for j in range(cols):
                if allocation[i, j] > 0:
                    if not np.isnan(u[i]) and np.isnan(v[j]):
                        v[j] = costs[i, j] - u[i]
                    elif not np.isnan(v[j]) and np.isnan(u[i]):
                        u[i] = costs[i, j] - v[j]
                        
    print(f"Row Multipliers (u): {u}")
    print(f"Col Multipliers (v): {v}")
    
    print("\nOpportunity Costs for Unoccupied Cells (Delta_ij):")
    optimal = True
    for i in range(rows):
        for j in range(cols):
            if allocation[i, j] == 0:
                delta = costs[i, j] - (u[i] + v[j])
                print(f"Cell ({i+1},{j+1}) Delta = {delta}")
                if delta < 0: optimal = False
                    
    if optimal:
        print("\n=> All Deltas >= 0. The VAM solution is OPTIMAL.")
        print(f"=> Minimum Total Transportation Cost: {vam_total}")
    else:
        print("\n=> Negative Delta found. Iterate via closed loop to improve.")

if __name__ == "__main__":
    solve_transportation()