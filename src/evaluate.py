import json
import os
import time

import pandas as pd
import torch
from HGNN_agent import HGNN_agent
from MARCO import MARCO, oracle_SAT_solver
from MARCO_with_agent import MARCO_with_agent
from model import AllSetTransformer_RL
from problem import generate_random_SAT_problems
from tqdm import tqdm


def evaluate_single_instance(
    idx,
    clauses,
    model,
    max_itn,
    itn_limits,
    device,
    output_dir,
    timeout=600,
):
    result_dict = {}
    os.makedirs(output_dir, exist_ok=True)

    # w/o agent
    st = time.time()
    try:
        mus_list, mss_list, oracle_itn, log, episode, finish_status = MARCO(
            clauses,
            oracle_solver=oracle_SAT_solver,
            timeout=timeout,
            max_oracle_itn=max_itn,
        )
        print(f"Instance {idx} MARCO w/o agent time: {time.time()-st:.6f} seconds")
        result_dict[f"time_MARCO_wo_agent"] = time.time() - st
        result_dict[f"finish_status_MARCO_wo_agent"] = finish_status

        log_df = pd.DataFrame(log)
        log_df.to_csv(
            os.path.join(
                output_dir, f"MARCO_wo_agent_{idx}_log.csv"
            ),
            index=False,
        )
        for itn_limit in itn_limits:
            col_name = f"found_mus_num_MARCO_wo_agent_oracle_itn{itn_limit}"
            found_mus_num_in_limit = log_df[
                log_df["oracle_iteration_num"] <= itn_limit
            ]["found_mus_num"].max()
            result_dict[col_name] = (
                int(found_mus_num_in_limit)
                if not pd.isna(found_mus_num_in_limit)
                else 0
            )

            col_name = f"found_mss_num_MARCO_wo_agent_oracle_itn{itn_limit}"
            found_mss_num_in_limit = log_df[
                log_df["oracle_iteration_num"] <= itn_limit
            ]["found_mss_num"].max()
            result_dict[col_name] = (
                int(found_mss_num_in_limit)
                if not pd.isna(found_mss_num_in_limit)
                else 0
            )
        col_name = f"found_mus_num_MARCO_wo_agent"
        result_dict[col_name] = len(mus_list)
        col_name = f"found_mss_num_MARCO_wo_agent"
        result_dict[col_name] = len(mss_list)
    except Exception as e:
        print(f"Error in MARCO without agent on instance {idx}: {e}")

    ## w/ agent
    agent = HGNN_agent(model, clauses, device=device)
    st = time.time()
    try:
        mus_list, mss_list, oracle_itn, proxy_itn, log, episode, finish_status = (
            MARCO_with_agent(
                clauses,
                oracle_solver=oracle_SAT_solver,
                timeout=timeout,
                max_oracle_itn=max_itn,
                max_proxy_itn=float("inf"),
                agent=agent,
                error_correction=True,
            )
        )
        print(f"Instance {idx} MARCO w agent time: {time.time()-st:.6f} seconds")
        result_dict[f"time_MARCO_w_agent"] = time.time() - st
        result_dict[f"finish_status_MARCO_w_agent"] = finish_status

        log_df = pd.DataFrame(log)
        log_df.to_csv(
            os.path.join(
                output_dir, f"MARCO_w_agent_{idx}_log.csv"
            ),
            index=False,
        )
        for itn_limit in itn_limits:
            col_name = f"found_mus_num_MARCO_w_agent_oracle_itn{itn_limit}"
            found_mus_num_in_limit = log_df[
                log_df["oracle_iteration_num"] <= itn_limit
            ]["found_mus_num"].max()
            result_dict[col_name] = (
                int(found_mus_num_in_limit)
                if not pd.isna(found_mus_num_in_limit)
                else 0
            )

            col_name = f"found_mss_num_MARCO_w_agent_oracle_itn{itn_limit}"
            found_mss_num_in_limit = log_df[
                log_df["oracle_iteration_num"] <= itn_limit
            ]["found_mss_num"].max()
            result_dict[col_name] = (
                int(found_mss_num_in_limit)
                if not pd.isna(found_mss_num_in_limit)
                else 0
            )
        col_name = f"found_mus_num_MARCO_w_agent"
        result_dict[col_name] = len(mus_list)
        col_name = f"found_mss_num_MARCO_w_agent"
        result_dict[col_name] = len(mss_list)
    except Exception as e:
        print(f"Error in MARCO with agent on instance {idx}: {e}")
    
    return result_dict

def evaluate(
    model_dir,
    data_path=None,
    output_dir="test_output",
    num_problems=500,
    max_itn=20000,
    n_variables=(5, 20),
    min_n_clauses=1,
    p_geo=0.3,
    device="cpu",
    use_best_model=False,
    itn_limits=[100, 1000, 5000, 10000, 20000],
):
    with open(os.path.join(model_dir, "model_param.json"), "r") as f:
        model_param = json.load(f)
    model = AllSetTransformer_RL(**model_param)
    if use_best_model:
        model_path = os.path.join(model_dir, "best_model.pth")
    else:
        model_path = os.path.join(model_dir, "model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    if data_path is None:
        df = generate_random_SAT_problems(
            num_problems=num_problems,
            n_variables=n_variables,
            min_n_clauses=min_n_clauses,
            p_geo=p_geo,
            output_path=os.path.join(output_dir, "random_SAT_problems.csv"),
        )
    else:
        df = pd.read_csv(data_path)
        df["clauses"] = df["clauses"].apply(json.loads)


    start_time = time.time()
    evaluate_log_dir = os.path.join(output_dir, "evaluate_log")
    os.makedirs(evaluate_log_dir, exist_ok=True)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        clauses = row["clauses"]

        single_result_dict = evaluate_single_instance(
            idx,
            clauses,
            model,
            max_itn,
            itn_limits,
            device,
            evaluate_log_dir,
            timeout=600,
        )

        for key, value in single_result_dict.items():
            df.at[idx, key] = value
    print(
        f"Evaluation completed in {time.time() - start_time:.2f} seconds."
    )
    df.to_csv(os.path.join(output_dir, f"evaluation_results_on_random_SAT_problems.csv"), index=False)


    result_overview_dict = {}
    for itn_limit in itn_limits:
        key = f"mean_found_mus_mss_num_MARCO_wo_agent_oracle_itn{itn_limit}"
        result_overview_dict[key] = (
            df[f"found_mus_num_MARCO_wo_agent_oracle_itn{itn_limit}"]
            + df[f"found_mss_num_MARCO_wo_agent_oracle_itn{itn_limit}"]
        ).mean()
        key = f"std_found_mus_mss_num_MARCO_wo_agent_oracle_itn{itn_limit}"
        result_overview_dict[key] = (
            df[f"found_mus_num_MARCO_wo_agent_oracle_itn{itn_limit}"]
            + df[f"found_mss_num_MARCO_wo_agent_oracle_itn{itn_limit}"]
        ).std()
        key = f"mean_found_mus_mss_num_MARCO_w_agent_oracle_itn{itn_limit}"
        result_overview_dict[key] = (
            df[f"found_mus_num_MARCO_w_agent_oracle_itn{itn_limit}"]
            + df[f"found_mss_num_MARCO_w_agent_oracle_itn{itn_limit}"]
        ).mean()
        key = f"std_found_mus_mss_num_MARCO_w_agent_oracle_itn{itn_limit}"
        result_overview_dict[key] = (
            df[f"found_mus_num_MARCO_w_agent_oracle_itn{itn_limit}"]
            + df[f"found_mss_num_MARCO_w_agent_oracle_itn{itn_limit}"]
        ).std()
        key = f"mean_improve_ratio_MARCO_oracle_itn{itn_limit}"
        result_overview_dict[key] = (
            (
                df[f"found_mus_num_MARCO_w_agent_oracle_itn{itn_limit}"]
                + df[f"found_mss_num_MARCO_w_agent_oracle_itn{itn_limit}"]
            )                / (
                df[f"found_mus_num_MARCO_wo_agent_oracle_itn{itn_limit}"]
                + df[f"found_mss_num_MARCO_wo_agent_oracle_itn{itn_limit}"]
            )
        ).mean()
        key = f"std_improve_ratio_MARCO_oracle_itn{itn_limit}"
        result_overview_dict[key] = (
            (
                df[f"found_mus_num_MARCO_w_agent_oracle_itn{itn_limit}"]
                + df[f"found_mss_num_MARCO_w_agent_oracle_itn{itn_limit}"]
            )                / (
                df[f"found_mus_num_MARCO_wo_agent_oracle_itn{itn_limit}"]
                + df[f"found_mss_num_MARCO_wo_agent_oracle_itn{itn_limit}"]
            )
        ).std()
    key = f"mean_found_mus_mss_num_MARCO_wo_agent"
    result_overview_dict[key] = (
        df[f"found_mus_num_MARCO_wo_agent"]
        + df[f"found_mss_num_MARCO_wo_agent"]
    ).mean()
    key = f"std_found_mus_mss_num_MARCO_wo_agent"
    result_overview_dict[key] = (
        df[f"found_mus_num_MARCO_wo_agent"]
        + df[f"found_mss_num_MARCO_wo_agent"]
    ).std()
    key = f"mean_found_mus_mss_num_MARCO_w_agent"
    result_overview_dict[key] = (
        df[f"found_mus_num_MARCO_w_agent"]
        + df[f"found_mss_num_MARCO_w_agent"]
    ).mean()
    key = f"std_found_mus_mss_num_MARCO_w_agent"
    result_overview_dict[key] = (
        df[f"found_mus_num_MARCO_w_agent"]
        + df[f"found_mss_num_MARCO_w_agent"]
    ).std()
    key = f"mean_improve_ratio_MARCO"
    result_overview_dict[key] = (
        (
            df[f"found_mus_num_MARCO_w_agent"]
            + df[f"found_mss_num_MARCO_w_agent"]
        )            / (
            df[f"found_mus_num_MARCO_wo_agent"]
            + df[f"found_mss_num_MARCO_wo_agent"]
        )
    ).mean()
    key = f"std_improve_ratio_MARCO"
    result_overview_dict[key] = (
        (
            df[f"found_mus_num_MARCO_w_agent"]
            + df[f"found_mss_num_MARCO_w_agent"]
        )            / (
            df[f"found_mus_num_MARCO_wo_agent"]
            + df[f"found_mss_num_MARCO_wo_agent"]
        )
    ).std()

    evaluation_param = {
        "model_dir": model_dir,
        "model_param": model_param,
        "data_path": data_path,
        "output_dir": output_dir,
        "num_problems": num_problems,
        "max_itn": max_itn,
        "n_variables": n_variables,
        "min_n_clauses": min_n_clauses,
        "p_geo": p_geo,
        "device": str(device),
        "result_overview": result_overview_dict,
        "use_best_model": use_best_model,
    }
    with open(os.path.join(output_dir, "evaluation_param.json"), "w") as f:
        json.dump(evaluation_param, f, indent=4)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="test_output")
    parser.add_argument("--num_problems", type=int, default=500)
    parser.add_argument("--max_itn", type=int, default=10000)
    parser.add_argument("--n_variables", type=int, nargs=2, default=(5, 20))
    parser.add_argument("--min_n_clauses", type=int, default=1)
    parser.add_argument("--p_geo", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use_best_model", action="store_true")
    parser.add_argument("--itn_limits", type=int, nargs="+", default=[500, 1000, 5000, 10000])
    args = parser.parse_args()

    evaluate(
        model_dir=args.model_dir,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_problems=args.num_problems,
        max_itn=args.max_itn,
        n_variables=args.n_variables,
        min_n_clauses=args.min_n_clauses,
        p_geo=args.p_geo,
        device=args.device,
        use_best_model=args.use_best_model,
        itn_limits=args.itn_limits,
    )