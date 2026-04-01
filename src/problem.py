import numpy as np
import pandas as pd
from SR.gen_sr_clauses import gen_unsat_clauses


def generate_random_SAT_problems(
    num_problems=500,
    n_variables=(5, 20),
    min_n_clauses=1,
    p_geo=0.3,
    output_path="data.csv",
):
    data_list = []
    for i in range(num_problems):
        if n_variables is None:
            n = np.random.randint(30, 100)
        elif isinstance(n_variables, int):
            n = n_variables
        elif isinstance(n_variables, (list, tuple)) and len(n_variables) == 2:
            n = np.random.randint(n_variables[0], n_variables[1])
        else:
            raise ValueError("n_variables must be int, (min, max), or None")
        clauses = gen_unsat_clauses(
            n,
            min_n_clauses=min_n_clauses,
            p_geo=p_geo,
        )
        data = {
            "problem_id": i,
            "clauses": clauses,
            "num_variables": n,
            "num_clauses": len(clauses),
        }
        data_list.append(data)
    df = pd.DataFrame(data_list)
    df.to_csv(output_path)
    return df