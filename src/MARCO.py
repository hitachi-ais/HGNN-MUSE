import random
import time

from pysat.solvers import Solver


def oracle_SAT_solver(clauses, subset, solver="g3"):
    sub_clauses = [clauses[i] for i in subset]
    with Solver(name=solver, bootstrap_with=sub_clauses) as solver:
        return solver.solve()

def shrink(clauses, subset, solver):
    n_clauses = len(clauses)
    shrunk_subset = subset[:]
    log = []
    for i in random.sample(subset, len(subset)):
        new_shrunk_subset = [j for j in shrunk_subset if j != i]
        if len(new_shrunk_subset) > 0:
            SAT = solver(clauses, new_shrunk_subset)
            log.append(
                {
                    "subset": new_shrunk_subset,
                    "SAT": SAT,
                }
            )
        else:
            SAT = True
        if SAT:
            continue
        else:
            shrunk_subset = new_shrunk_subset
    return shrunk_subset, len(subset), log

def grow(clauses, subset, solver):
    n_clauses = len(clauses)
    _subset = [i for i in range(n_clauses) if i not in subset]
    grown_subset = subset[:]
    log = []
    for i in random.sample(_subset, len(_subset)):
        new_grown_subset = grown_subset[:]
        new_grown_subset.append(i)
        SAT = solver(clauses, new_grown_subset)
        log.append(
            {
                "subset": new_grown_subset,
                "SAT": SAT,
            }
        )
        if SAT:
            grown_subset = new_grown_subset
        else:
            continue
    return grown_subset, len(_subset), log

def MARCO(
    clauses,
    oracle_solver,
    max_oracle_itn=float("inf"),
    timeout=float("inf"),
    max_loop=float("inf"),
    map_solver_name="g3",
):
    st = time.time()
    MSS_list = []
    MUS_list = []
    n_clauses = len(clauses)

    map_clauses = []
    n = 0
    oracle_itn = 0
    log = []
    episode = []
    bootstrap = [[i, -i] for i in range(1, n_clauses + 1)]
    finish_status = "completed"
    while n < max_loop:
        with Solver(
            name=map_solver_name, bootstrap_with=bootstrap + map_clauses
        ) as solver:
            phases = [
                i if random.random() < 0.5 else -i for i in range(1, n_clauses + 1)
            ]
            solver.set_phases(phases)
            if solver.solve():
                model = solver.get_model()
            else:
                break
        subset = [i - 1 for i in model if i > 0]
        SAT = oracle_solver(clauses, subset)
        oracle_itn += 1
        if SAT:
            mss, grow_itn, grow_log = grow(clauses, subset, solver=oracle_solver)

            if oracle_itn + grow_itn > max_oracle_itn:
                finish_status = "exceeded_max_oracle_itn"
                break
            if time.time() - st > timeout:
                finish_status = "timeout"
                # print("TIMEOUT")
                break
            episode.append(
                {
                    "num_clauses": n_clauses,
                    "seed_subset": subset,
                    "output_subset": mss,
                    "mode": "grow",
                    "oracle_iteration": grow_itn,
                    "found_mus": MUS_list[:],
                    "found_mss": MSS_list[:],
                    "total_iteration": oracle_itn,
                    "log": grow_log,
                }
            )
            oracle_itn += grow_itn
            MSS_list.append(mss)
            _mss = [i for i in range(n_clauses) if i not in mss]
            map_clauses.append([i + 1 for i in _mss])  # blockDown
        else:
            mus, shrink_itn, shrink_log = shrink(clauses, subset, solver=oracle_solver)

            if oracle_itn + shrink_itn > max_oracle_itn:
                finish_status = "exceeded_max_oracle_itn"
                break
            if time.time() - st > timeout:
                finish_status = "timeout"
                # print("TIMEOUT")
                break
            episode.append(
                {
                    "num_clauses": n_clauses,
                    "seed_subset": subset,
                    "output_subset": mus,
                    "mode": "shrink",
                    "oracle_iteration": shrink_itn,
                    "found_mus": MUS_list[:],
                    "found_mss": MSS_list[:],
                    "total_iteration": oracle_itn,
                    "log": shrink_log,
                }
            )
            oracle_itn += shrink_itn
            MUS_list.append(mus)
            map_clauses.append([-(i + 1) for i in mus])  # blockUp
        n += 1
        log.append(
            {
                "found_mus_num": len(MUS_list),
                "found_mss_num": len(MSS_list),
                "oracle_iteration_num": oracle_itn,
                "time": time.time() - st,
            }
        )
    if len(log) == 0:
        log.append(
            {
                "found_mus_num": len(MUS_list),
                "found_mss_num": len(MSS_list),
                "oracle_iteration_num": oracle_itn,
                "time": time.time() - st,
            }
        )
    return MUS_list, MSS_list, oracle_itn, log, episode, finish_status