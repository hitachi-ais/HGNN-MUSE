# Codes in this file are adapted from https://github.com/dselsam/neurosat/blob/master/python/gen_sr_dimacs.py
# Below is the original license for that file.
# ==============================================================================
# Copyright 2018 Daniel Selsam. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
from pysat.solvers import Solver


def generate_k_iclause(n, k):
    vs = np.sort(np.random.choice(n, size=min(n, k), replace=False))
    return [int(v + 1) if np.random.random() < 0.5 else -int(v + 1) for v in vs]


def gen_unsat_clauses(
    n,
    min_n_clauses=1,
    max_n_clauses=2500,
    p_geo=0.3,
    solver_name="g3",
):
    while True:
        clauses = []
        clause_strs = []
        SAT = True
        with Solver(name=solver_name) as solver:
            while SAT:
                k = (
                    1 + np.random.binomial(1, 0.7) + np.random.geometric(p_geo)
                )
                iclause = generate_k_iclause(n, k)
                iclause_str = ",".join(map(str, iclause))
                if iclause_str not in clause_strs:
                    solver.add_clause(iclause)
                    clauses.append(iclause)
                    clause_strs.append(iclause_str)
                    SAT = solver.solve()
        if len(clauses) >= min_n_clauses and len(clauses) <= max_n_clauses and not SAT:
            break

    return clauses