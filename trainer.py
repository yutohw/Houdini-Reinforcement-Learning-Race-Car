import time
import os
import torch
import torch.optim as optim
import numpy as np
import common

def wait_for_all(round_num, poll_interval=0.05):
    flags = [common.done_flag_path(wid, round_num) for wid in common.worker_ids]
    while not all(os.path.exists(f) for f in flags):
        time.sleep(poll_interval)


def load_rollout(worker_id, round_num, retries=100, delay=0.05):
    path = common.rollout_path(worker_id, round_num)
    for _ in range(retries):
        try:
            return torch.load(path, weights_only=False)
        except (RuntimeError, EOFError, OSError, PermissionError):
            time.sleep(delay)
    raise RuntimeError(f"Could not read rollout for worker {worker_id}, round {round_num}")


def main():
    common.seed_process(0)

    model     = common.ActorCritic(common.state_dim, common.action_dim)
    optimizer = optim.Adam(model.parameters(), lr=common.learning_rate)

    episode_returns  = []
    best_return      = -float("inf")
    best_state_dict  = None
    global_episode   = 0
    last_10_start    = int(common.num_episodes * 0.9)
    checkpoint_every_rounds = 100

    print("=== PPO Trainer started (multi-worker) ===")

    round_num = 0
    while global_episode < common.num_episodes:
        round_num += 1

        weights_file = common.weights_round_path(round_num)
        common.atomic_torch_save(weights_file, model.state_dict())

        tracks = common.next_track_batch(global_episode + 1, common.num_episodes, common.num_workers)

        current_lr = common.cosine_lr(global_episode + 1, common.num_episodes,
                                      common.learning_rate, common.learning_rate_min)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        current_entropy_coef = common.linear_entropy_coef(global_episode + 1, common.num_episodes,
                                                          common.entropy_coef_start, common.entropy_coef_end)

        should_record = (round_num % 2 == 0)

        for wid, track in zip(common.worker_ids, tracks):
            common.atomic_write_json(common.assignment_path(wid, round_num), {
                "round": round_num,
                "track": track,
                "record": should_record,
                "weights_file": weights_file,
            })

        wait_for_all(round_num)

        batch_states, batch_actions, batch_pre_tanhs = [], [], []
        batch_logps, batch_returns, batch_advantages = [], [], []

        round_summary = []
        for wid, track in zip(common.worker_ids, tracks):
            rollout = load_rollout(wid, round_num)

            steps_taken    = rollout["steps_taken"]
            episode_return = rollout["episode_return"]

            batch_states.extend(rollout["states"])
            batch_actions.extend(rollout["actions"])
            batch_pre_tanhs.extend(rollout["pre_tanhs"])
            batch_logps.extend(rollout["logps"])
            batch_returns.extend(rollout["returns"])
            batch_advantages.extend(rollout["advantages"])

            episode_returns.append(episode_return)
            round_summary.append((track, steps_taken))

            global_episode += 1
            if global_episode >= last_10_start and episode_return > best_return:
                best_return     = episode_return
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        if len(batch_states) > 1:
            common.ppo_update(model, optimizer, batch_states, batch_actions, batch_pre_tanhs,
                              batch_logps, batch_returns, batch_advantages, current_entropy_coef)

        if round_num % checkpoint_every_rounds == 0:
            checkpoint_path = os.path.join(common.model_dir, f"checkpoint_{global_episode}.pth")
            torch.save(model.state_dict(), checkpoint_path)

        common.cleanup_old_rounds(round_num, keep=3)

        avg_100 = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else np.mean(episode_returns)
        summary = " | ".join(f"T{t}:{s}" for t, s in round_summary)
        print(f"Round {round_num:5d} | Episodes {global_episode:5d} | {summary} | "
              f"Avg100 {avg_100:8.2f} | LR {current_lr:.2e} | Ent {current_entropy_coef:.4f} | "
              f"LogStd {model.log_std.data.mean().item():.3f}")

    torch.save(best_state_dict, common.model_path)
    print(f"\nTraining complete.")
    print(f"Best model from last 10% saved (Return {best_return:.2f})")


if __name__ == "__main__":
    main()