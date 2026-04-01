import random
import time

from MARCO import grow, shrink
from pysat.solvers import Solver


def random_agent(all_clauses, subset_id, MUS_list, MSS_list, mode, avoided_actions=[]):
    n_clauses = len(all_clauses)
    if mode == "shrink":
        actions = subset_id + [n_clauses]
        actions = [a for a in actions if a not in avoided_actions]
        action = random.choice(actions)
        action_prob = 1 / len(actions)
        return action, action_prob, None
    elif mode == "grow":
        _subset_id = [i for i in range(n_clauses) if i not in subset_id]
        actions = _subset_id + [n_clauses]
        actions = [a for a in actions if a not in avoided_actions]
        action = random.choice(actions)
        action_prob = 1 / len(actions)
        return action, action_prob, None

def shrink_with_agent(clauses, subset, MUS_list, MSS_list, agent):
    shrunk_subset = subset[:]
    deleted_subset = []
    log = []
    itn = 0
    while len(shrunk_subset) > 0:
        action_id, action_prob, value = agent(
            clauses, shrunk_subset, MUS_list, MSS_list, mode="shrink"
        )
        assert action_id in shrunk_subset or action_id == len(
            clauses
        ), "Action ID must be in the subset or be equal to the number of clauses (indicating finish)."
        itn += 1
        log.append(
            {
                "subset": shrunk_subset,
                "action_id": action_id,
                "action_prob": action_prob,
                "value": value,
            }
        )
        if action_id >= len(clauses):
            break
        shrunk_subset = [i for i in shrunk_subset if i != action_id]
        deleted_subset.append(action_id)
    return shrunk_subset, deleted_subset, itn, log

def grow_with_agent(clauses, subset, MUS_list, MSS_list, agent):
    grown_subset = subset[:]
    added_subset = []
    log = []
    itn = 0
    while len(grown_subset) < len(clauses):
        action_id, action_prob, value = agent(
            clauses, grown_subset, MUS_list, MSS_list, mode="grow"
        )
        assert (
            action_id not in grown_subset
        ), "Action ID should not be in the grown subset."
        itn += 1
        log.append(
            {
                "subset": grown_subset,
                "action_id": action_id,
                "action_prob": action_prob,
                "value": value,
            }
        )
        if action_id >= len(clauses):
            break
        grown_subset = grown_subset[:] + [action_id]
        added_subset.append(action_id)
    return grown_subset, added_subset, itn, log

def correct_error_shrink(clauses, mus, deleted_subset, oracle_solver):
    SAT = oracle_solver(clauses, mus)
    if SAT:
        r_shrink_itn = 0
        r_mus = mus[:]
        for add_s in deleted_subset[::-1]:
            r_mus.append(add_s)
            r_SAT = oracle_solver(clauses, r_mus)
            r_shrink_itn += 1
            if not r_SAT:
                break
        mus, shrink_itn, _ = shrink(clauses, r_mus, solver=oracle_solver)
        shrink_itn = shrink_itn + r_shrink_itn
    else:
        mus, shrink_itn, _ = shrink(clauses, mus, solver=oracle_solver)
    oracle_shrink_itn = shrink_itn + 1
    return mus, oracle_shrink_itn

def correct_error_grow(clauses, mss, added_subset, oracle_solver):
    SAT = oracle_solver(clauses, mss)
    if SAT:
        mss, grow_itn, _ = grow(clauses, mss, solver=oracle_solver)
    else:
        r_grow_itn = 0
        r_mss = mss[:]
        for del_s in added_subset[::-1]:
            r_mss = [s for s in r_mss if s != del_s]
            r_SAT = oracle_solver(clauses, r_mss)
            r_grow_itn += 1
            if r_SAT:
                break
        mss, grow_itn, _ = grow(clauses, r_mss, solver=oracle_solver)
        grow_itn = grow_itn + r_grow_itn
    oracle_grow_itn = grow_itn + 1
    return mss, oracle_grow_itn

def MARCO_with_agent(
    clauses,
    oracle_solver,
    max_oracle_itn=float("inf"),
    max_proxy_itn=float("inf"),
    max_total_itn=float("inf"),
    timeout=float("inf"),
    max_loop=float("inf"),
    map_solver_name="g3",
    agent=random_agent,
    error_correction=True,
):
    st = time.time()
    MSS_list = []
    MUS_list = []
    n_clauses = len(clauses)

    map_clauses = []
    n = 0
    oracle_itn = 0
    proxy_itn = 0
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
            mss, added_subset, proxy_grow_itn, grow_log = grow_with_agent(
                clauses, subset, MUS_list, MSS_list, agent
            )

            if error_correction:
                mss, oracle_grow_itn = correct_error_grow(
                    clauses, mss, added_subset, oracle_solver=oracle_solver
                )
            else:
                oracle_grow_itn = 0

            if oracle_itn + oracle_grow_itn > max_oracle_itn:
                finish_status = "exceeded_max_oracle_itn"
                break
            if proxy_itn + proxy_grow_itn > max_proxy_itn:
                finish_status = "exceeded_max_proxy_itn"
                break
            if (
                oracle_itn + proxy_itn + oracle_grow_itn + proxy_grow_itn
                > max_total_itn
            ):
                finish_status = "exceeded_max_total_itn"
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
                    "oracle_iteration": oracle_grow_itn,
                    "proxy_iteration": proxy_grow_itn,
                    "found_mus": MUS_list[:],
                    "found_mss": MSS_list[:],
                    "total_iteration": oracle_itn,
                    "log": grow_log,
                }
            )
            oracle_itn += oracle_grow_itn
            proxy_itn += proxy_grow_itn
            MSS_list.append(mss)
            _mss = [i for i in range(n_clauses) if i not in mss]
            map_clauses.append([i + 1 for i in _mss])  # blockDown
        else:
            mus, deleted_subset, proxy_shrink_itn, shrink_log = shrink_with_agent(
                clauses, subset, MUS_list, MSS_list, agent
            )

            if error_correction:
                mus, oracle_shrink_itn = correct_error_shrink(
                    clauses, mus, deleted_subset, oracle_solver=oracle_solver
                )
            else:
                oracle_shrink_itn = 0
            if oracle_itn + oracle_shrink_itn > max_oracle_itn:
                finish_status = "exceeded_max_oracle_itn"
                break
            if proxy_itn + proxy_shrink_itn > max_proxy_itn:
                finish_status = "exceeded_max_proxy_itn"
                break
            if (
                oracle_itn + proxy_itn + oracle_shrink_itn + proxy_shrink_itn
                > max_total_itn
            ):
                finish_status = "exceeded_max_total_itn"
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
                    "oracle_iteration": oracle_shrink_itn,
                    "proxy_iteration": proxy_shrink_itn,
                    "found_mus": MUS_list[:],
                    "found_mss": MSS_list[:],
                    "total_iteration": oracle_itn,
                    "log": shrink_log,
                }
            )
            oracle_itn += oracle_shrink_itn
            proxy_itn += proxy_shrink_itn
            MUS_list.append(mus)
            map_clauses.append([-(i + 1) for i in mus])  # blockUp
        n += 1
        log.append(
            {
                "found_mus_num": len(MUS_list),
                "found_mss_num": len(MSS_list),
                "oracle_iteration_num": oracle_itn,
                "proxy_iteration_num": proxy_itn,
                "time": time.time() - st,
            }
        )
    if len(log) == 0:
        log.append(
            {
                "found_mus_num": len(MUS_list),
                "found_mss_num": len(MSS_list),
                "oracle_iteration_num": oracle_itn,
                "proxy_iteration_num": proxy_itn,
                "time": time.time() - st,
            }
        )
    return MUS_list, MSS_list, oracle_itn, proxy_itn, log, episode, finish_status