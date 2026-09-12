import numpy as np

def solve_big_m():
    """
    PROBLEM (Big-M Method):
    Minimize Z = 4x1 + x2
    Constraints:
      1) 3x1 + x2 = 3
      2) 4x1 + 3x2 >= 6
      3) x1 + 2x2 <= 4
      4)x1, x2 >= 0
      
    Converting to Standard form (Big-M penalty M = 10000):
    Minimize Z = 4x1 + x2 + 0s1 + 0s2 + Ma1 + Ma2
      
    Subject to:
     3x1 + x2 + a1 = 3
     4x1 + 3x2 - s1 + a2 = 6
     x1 + 2x2 + s2 = 4
     x1, x2, s1, s2, a1, a2 >= 0
    """
    M = 10000.0  # Large penalty for artificial variables
    
    # Objective coefficients: [x1, x2, s1, s2, a1, a2]
    Coeff = np.array([4, 1, 0, 0, M, M], dtype=float)
    
    # Initial Tableau: [x1, x2, s1, s2, a1, a2, Solution_RHS]
    tableau = np.array([
        [3, 1,  0, 0, 1, 0, 3],  # Constraint 1 (with a1)
        [4, 3, -1, 0, 0, 1, 6],  # Constraint 2 (with s1, a2)
        [1, 2,  0, 1, 0, 0, 4]   # Constraint 3 (with s2)
    ], dtype=float)
    
    # Initial basic variables indices: a1 (4), a2 (5), s2 (3)
    basic_var_idx = [4, 5, 3]
    
    iteration = 1
    while True:
        #  Zj - Cj (Net evaluation row)
        CB = Coeff[basic_var_idx]
        Zj = np.dot(CB, tableau[:, :-1])
        Zj_Cj = Zj - Coeff
        
        # Optimality condition for minimization: All Zj - Cj <= 0
        if np.all(Zj_Cj <= 1e-5):
            break
            
        # Entering variable: most positive Zj - Cj
        enter_col = np.argmax(Zj_Cj)
        
        # Leaving variable: Minimum positive ratio
        ratios = np.full(len(basic_var_idx), np.inf)
        for i in range(len(basic_var_idx)):
            if tableau[i, enter_col] > 0:
                ratios[i] = tableau[i, -1] / tableau[i, enter_col]
                
        leave_row = np.argmin(ratios)
        if ratios[leave_row] == np.inf:
            print("Problem is unbounded.")
            return
            
        # Pivot operation
        basic_var_idx[leave_row] = enter_col
        pivot = tableau[leave_row, enter_col]
        tableau[leave_row] = tableau[leave_row] / pivot
        
        for i in range(len(basic_var_idx)):
            if i != leave_row:
                tableau[i] -= tableau[i, enter_col] * tableau[leave_row]
        iteration += 1

    # Extract final optimal values
    solution = np.zeros(len(Coeff))
    for i, var_idx in enumerate(basic_var_idx):
        solution[var_idx] = tableau[i, -1]
        
    optimal_z = np.sum(solution[:4] * Coeff[:4]) # Objective value excluding artificials
    
    print(f"Optimal Objective Value (Z): {round(optimal_z, 2)}")
    print(f"Decision Variables: x1 = {round(solution[0], 2)}, x2 = {round(solution[1], 2)}")
    print(f"Slack/Surplus: s1 = {round(solution[2], 2)}, s2 = {round(solution[3], 2)}")
    print(f"Artificials: a1 = {round(solution[4], 2)}, a2 = {round(solution[5], 2)}")
    return

if __name__ == "__main__":
    solve_big_m()