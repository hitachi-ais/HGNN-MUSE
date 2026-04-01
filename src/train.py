import json
import os
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from HGNN_agent import HGNN_agent, create_hypergraph
from MARCO import oracle_SAT_solver
from MARCO_with_agent import MARCO_with_agent
from model import AllSetTransformer_RL
from SR.gen_sr_clauses import gen_unsat_clauses
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup


def generate_episodes(
    model,
    max_proxy_itn,
    num_problems,
    n_variables=None,
    min_n_clauses=1,
    p_geo=0.3,
    device="cpu",
):
    episodes = []
    num_steps = 0
    for _ in tqdm(range(num_problems)):
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
        agent = HGNN_agent(model, clauses, device=device)
        MUS_list, MSS_list, oracle_itn, proxy_itn, log, episode, finish_status = (
            MARCO_with_agent(
                clauses,
                oracle_solver=oracle_SAT_solver,
                max_oracle_itn=float("inf"),
                max_proxy_itn=max_proxy_itn,
                agent=agent,
                error_correction=True,
            )
        )
        episode = [e | {"clauses": clauses, "num_variables": n} for e in episode]
        episodes += episode
        num_steps += proxy_itn
    return episodes, num_steps

def calc_local_rewards(episodes):
    local_rewards = []
    num_loops = []
    for e in episodes:
        seed_subset_size = len(e["seed_subset"])
        output_subset_size = len(e["output_subset"])
        num_clauses = e["num_clauses"]
        if e["mode"] == "shrink":
            local_reward = 1 - (e["oracle_iteration"] - output_subset_size) / max(
                seed_subset_size, 1
            )
        elif e["mode"] == "grow":
            local_reward = 1 - (
                e["oracle_iteration"] - num_clauses + output_subset_size
            ) / max((num_clauses - seed_subset_size), 1)
        local_rewards.append(local_reward)
        num_loops.append(len(e["found_mus"]) + len(e["found_mss"]))
    return local_rewards, num_loops

def episode_dataloader(
    episodes,
    num_use_episodes=None,
    rng=None,
    max_batch_size=1,
    min_batch_size=2,
    gamma=0.99,
    lam=0.95,
):
    if rng is None:
        rng = np.random.default_rng()
    if num_use_episodes is None:
        num_use_episodes = len(episodes)
    num_episodes = 0
    while num_episodes < num_use_episodes:
        for e in rng.permutation(episodes):
            if len(e["log"]) < min_batch_size:
                continue
            seed_subset_size = len(e["seed_subset"])
            output_subset_size = len(e["output_subset"])
            num_clauses = e["num_clauses"]
            clauses = e["clauses"]
            num_variables = e["num_variables"]

            if e["mode"] == "shrink":
                final_reward = 1 - (e["oracle_iteration"] - output_subset_size) / max(
                    seed_subset_size, 1
                )
            elif e["mode"] == "grow":
                final_reward = 1 - (
                    e["oracle_iteration"] - num_clauses + output_subset_size
                ) / max((num_clauses - seed_subset_size), 1)

            # Generate the hypergraph data structure
            found_mus = e["found_mus"]
            found_mss = e["found_mss"]

            data = create_hypergraph(num_clauses, found_mus, found_mss)

            if e["mode"] == "shrink":
                target_subsets_bin = torch.tensor(
                    [
                        [
                            (clause_index in state_action["subset"])
                            for clause_index in range(num_clauses)
                        ]
                        for state_action in e["log"]
                    ],
                    dtype=torch.bool,
                )
            elif e["mode"] == "grow":
                target_subsets_bin = torch.tensor(
                    [
                        [
                            (clause_index not in state_action["subset"])
                            for clause_index in range(num_clauses)
                        ]
                        for state_action in e["log"]
                    ]
                )
            p_actions = torch.tensor(
                [state_action["action_prob"] for state_action in e["log"]],
                dtype=torch.float32,
            )
            actions = torch.tensor(
                [state_action["action_id"] for state_action in e["log"]],
                dtype=torch.int64,
            )
            values = torch.tensor(
                [state_action["value"] for state_action in e["log"]],
                dtype=torch.float32,
            )

            rewards = torch.zeros(len(e["log"]), dtype=torch.float32)
            rewards[-1] = final_reward

            advantages = torch.zeros_like(values)
            gae = 0.0
            for t in reversed(range(len(values))):
                if t == len(values) - 1:
                    delta = rewards[t] - values[t]
                    gae = delta
                else:
                    delta = rewards[t] + gamma * values[t + 1] - values[t]
                    gae = delta + gamma * lam * gae
                advantages[t] = gae

            target_values = values + advantages

            if len(e["log"]) > max_batch_size:
                indices = rng.choice(len(e["log"]), size=max_batch_size, replace=False)
                target_subsets_bin = target_subsets_bin[indices]
                p_actions = p_actions[indices]
                actions = actions[indices]
                advantages = advantages[indices]
                values = values[indices]
                target_values = target_values[indices]

            yield data, target_subsets_bin, p_actions, actions, advantages, values, target_values, e[
                "mode"
            ], len(
                found_mus
            ) + len(
                found_mss
            )
            num_episodes += 1
            if num_episodes >= num_use_episodes:
                break

def train(
    output_dir="output",
    n_variables=(5, 20),
    min_n_clauses=1,
    p_geo=0.3,
    max_itn=5000,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    update_epochs=4,
    ppo_epochs=200,
    ppo_num_use_episodes=None,
    ppo_clip=0.2,
    ppo_value_loss_coef=0.5,
    ppo_entropy_coef=0.001,
    ppo_gamma=0.99,
    ppo_lam=0.95,
    lr=2e-05,
    weight_decay=0.0,
    max_batch_size=1024,
    seed=42,
    model_param={
        "dim": 64,
        "nhead": 4,
        "hgnn_layer_num": 3,
        "transformer_layer_num": 3,
        "hgnn_dropout": 0.5,
        "k": 8,
    },
    num_episode_problems=4,
):
    exp_param_dict = {
        "n_variables": n_variables,
        "min_n_clauses": min_n_clauses,
        "p_geo": p_geo,
        "max_itn": max_itn,
        "update_epochs": update_epochs,
        "ppo_epochs": ppo_epochs,
        "ppo_num_use_episodes": ppo_num_use_episodes,
        "ppo_clip": ppo_clip,
        "ppo_value_loss_coef": ppo_value_loss_coef,
        "ppo_entropy_coef": ppo_entropy_coef,
        "ppo_gamma": ppo_gamma,
        "ppo_lam": ppo_lam,
        "lr": lr,
        "weight_decay": weight_decay,
        "max_batch_size": max_batch_size,
        "model_param": model_param,
        "seed": seed,
        "num_episode_problems": num_episode_problems,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "exp_param.json"), "w") as f:
        json.dump(exp_param_dict, f, indent=4)

    with open(os.path.join(output_dir, "model_param.json"), "w") as f:
        json.dump(model_param, f)

    model = AllSetTransformer_RL(**model_param)
    model.to(device)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=ppo_epochs,
    )

    total_num_steps = 0
    num_steps_log = []
    avg_reward_log = []
    std_reward_log = []

    rng = np.random.default_rng(seed)
    reward_log = []
    loss_log = []
    best_reward = -10.0
    recent_rewards = deque([], maxlen=3)
    for ppo_epoch in range(ppo_epochs):
        model.eval()
        episodes, num_steps = generate_episodes(
            model,
            max_proxy_itn=max_itn,
            num_problems=num_episode_problems,
            n_variables=n_variables,
            min_n_clauses=min_n_clauses,
            p_geo=p_geo,
            device=device,
        )
        local_rewards, num_loops = calc_local_rewards(episodes)
        avg_reward_log.append(np.mean(local_rewards))
        std_reward_log.append(np.std(local_rewards))
        num_steps_log.append(total_num_steps)

        print(
            f"PPO Epoch {ppo_epoch}/{ppo_epochs}, Total Steps: {total_num_steps}, Reward: {np.mean(local_rewards):.4f} ± {np.std(local_rewards):.4f}"
        )
        recent_rewards.append(np.mean(local_rewards))
        if len(recent_rewards) >= 3 and np.mean(recent_rewards) >= best_reward:
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            best_reward = np.mean(recent_rewards)
        reward_log.append(
            {
                "ppo_epoch": ppo_epoch,
                "total_num_steps": total_num_steps,
                "avg_reward": np.mean(local_rewards),
                "std_reward": np.std(local_rewards),
            }
        )
        reward_log_df = pd.DataFrame(reward_log)
        reward_log_df.to_csv(os.path.join(output_dir, "reward_log.csv"), index=False)

        model.train()
        for update_epoch in range(update_epochs):
            dataloader = episode_dataloader(
                episodes,
                num_use_episodes=ppo_num_use_episodes,
                rng=rng,
                max_batch_size=max_batch_size,
                min_batch_size=1,
                gamma=ppo_gamma,
                lam=ppo_lam,
            )

            total_loss = 0.0
            total_actor_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0
            count = 0
            sample_count = 0
            for (
                data,
                target_subsets_bin,
                p_actions,
                actions,
                advantages,
                values,
                target_values,
                mode,
                itn,
            ) in dataloader:
                data = data.to(device)
                target_subsets_bin = target_subsets_bin.to(device)
                p_actions = p_actions.to(device)
                actions = actions.to(device)
                advantages = advantages.to(device)
                values = values.to(device)
                target_values = target_values.to(device)

                new_action_logits, new_values = model(
                    data, target_subsets_bin, mode=mode
                )

                # Calculate the loss
                new_dist = torch.distributions.Categorical(logits=new_action_logits)
                new_log_prob = new_dist.log_prob(actions)
                new_entropy = new_dist.entropy().sum()
                old_dist = torch.distributions.Categorical(probs=p_actions)
                old_log_prob = old_dist.log_prob(actions)
                ratio = torch.exp(new_log_prob - old_log_prob)
                clip_advantages = advantages * ratio.clamp(1 - ppo_clip, 1 + ppo_clip)
                policy_loss = -torch.minimum(ratio * advantages, clip_advantages).sum()

                actor_loss = policy_loss

                
                value_loss = F.mse_loss(
                    new_values, target_values, reduction="sum"
                )

                loss = (
                    actor_loss
                    + ppo_value_loss_coef * value_loss
                    - ppo_entropy_coef * new_entropy
                )

                (loss / max_batch_size).backward()
                count += actions.shape[0]
                sample_count += actions.shape[0]
                if sample_count >= max_batch_size:
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    sample_count = 0

                total_loss += loss.item()
                total_actor_loss += actor_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += new_entropy.item()
            optimizer.zero_grad()
            count = max(count, 1)
            avg_loss = total_loss / count
            avg_actor_loss = total_actor_loss / count
            avg_value_loss = total_value_loss / count
            avg_entropy = total_entropy / count
            
            print(
                f"Update Epoch {update_epoch + 1}/{update_epochs}, Loss: {avg_loss:.4f}, Actor Loss: {avg_actor_loss:.4f}, Value Loss: {avg_value_loss:.4f}, Entropy: {avg_entropy:.4f}"
            )
            loss_log.append(
                {
                    "ppo_epoch": ppo_epoch + 1,
                    "update_epoch": update_epoch + 1,
                    "loss": avg_loss,
                    "actor_loss": avg_actor_loss,
                    "value_loss": avg_value_loss,
                    "entropy": avg_entropy,
                    "num_episodes": len(episodes),
                }
            )
        lr_scheduler.step()
        total_num_steps += num_steps
        loss_log_df = pd.DataFrame(loss_log)
        loss_log_df.to_csv(os.path.join(output_dir, "loss_log.csv"), index=False)
        torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))

    model.eval()
    episodes, num_steps = generate_episodes(
        model,
        max_proxy_itn=max_itn,
        num_problems=num_episode_problems,
        n_variables=n_variables,
        min_n_clauses=min_n_clauses,
        p_geo=p_geo,
        device=device,
    )
    local_rewards, num_loops = calc_local_rewards(episodes)
    avg_reward_log.append(np.mean(local_rewards))
    std_reward_log.append(np.std(local_rewards))
    num_steps_log.append(total_num_steps)

    print(
        f"PPO Epoch {ppo_epoch+1}/{ppo_epochs}, Total Steps: {total_num_steps}, Reward: {np.mean(local_rewards):.4f} ± {np.std(local_rewards):.4f}"
    )
    recent_rewards.append(np.mean(local_rewards))
    if len(recent_rewards) >= 3 and np.mean(recent_rewards) >= best_reward:
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
        best_reward = np.mean(recent_rewards)
    reward_log.append(
        {
            "ppo_epoch": ppo_epoch + 1,
            "total_num_steps": total_num_steps,
            "avg_reward": np.mean(local_rewards),
            "std_reward": np.std(local_rewards),
        }
    )
    reward_log_df = pd.DataFrame(reward_log)
    reward_log_df.to_csv(os.path.join(output_dir, "reward_log.csv"), index=False)
    loss_log_df = pd.DataFrame(loss_log)
    loss_log_df.to_csv(os.path.join(output_dir, "loss_log.csv"), index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--n_variables", type=int, nargs=2, default=(5, 20))
    parser.add_argument("--min_n_clauses", type=int, default=1)
    parser.add_argument("--p_geo", type=float, default=0.3)
    parser.add_argument("--max_itn", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--ppo_epochs", type=int, default=200)
    parser.add_argument("--ppo_num_use_episodes", type=int, default=None)
    parser.add_argument("--ppo_clip", type=float, default=0.2)
    parser.add_argument("--ppo_value_loss_coef", type=float, default=0.5)
    parser.add_argument("--ppo_entropy_coef", type=float, default=0.001)
    parser.add_argument("--ppo_gamma", type=float, default=0.99)
    parser.add_argument("--ppo_lam", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=2e-05)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_batch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_episode_problems", type=int, default=4)
    parser.add_argument("--model_dim", type=int, default=64)
    parser.add_argument("--model_nhead", type=int, default=4)
    parser.add_argument("--model_hgnn_layer_num", type=int, default=3)
    parser.add_argument("--model_transformer_layer_num", type=int, default=3)
    parser.add_argument("--model_hgnn_dropout", type=float, default=0.5)
    parser.add_argument("--model_k", type=int, default=8)
    args = parser.parse_args()

    model_param = {
        "dim": args.model_dim,
        "nhead": args.model_nhead,
        "hgnn_layer_num": args.model_hgnn_layer_num,
        "transformer_layer_num": args.model_transformer_layer_num,
        "hgnn_dropout": args.model_hgnn_dropout,
        "k": args.model_k,
    }
    train(
        output_dir=args.output_dir,
        n_variables=args.n_variables,
        min_n_clauses=args.min_n_clauses,
        p_geo=args.p_geo,
        max_itn=args.max_itn,
        device=args.device,
        update_epochs=args.update_epochs,
        ppo_epochs=args.ppo_epochs,
        ppo_num_use_episodes=args.ppo_num_use_episodes,
        ppo_clip=args.ppo_clip,
        ppo_value_loss_coef=args.ppo_value_loss_coef,
        ppo_entropy_coef=args.ppo_entropy_coef,
        ppo_gamma=args.ppo_gamma,
        ppo_lam=args.ppo_lam,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_batch_size=args.max_batch_size,
        seed=args.seed,
        model_param=model_param,
        num_episode_problems=args.num_episode_problems,
    )