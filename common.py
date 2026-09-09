import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
import glob
import time as _time

# ===========================
# Paths
# ===========================
base_dir  = r"D:\Users\YUTO\Documents\Self_Driving\Model\Agent"
comms_dir = os.path.join(base_dir, "comms")
model_dir = os.path.join(base_dir, "model")

os.makedirs(comms_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "Self_Driving_Agent_Example_01.pth")

def assignment_path(worker_id, round_num):
    return os.path.join(comms_dir, f"assignment_{worker_id}_{round_num}.json")

def rollout_path(worker_id, round_num):
    return os.path.join(comms_dir, f"rollout_{worker_id}_{round_num}.pt")

def weights_round_path(round_num):
    return os.path.join(comms_dir, f"weights_round_{round_num}.pth")

def done_flag_path(worker_id, round_num):
    return os.path.join(comms_dir, f"done_{worker_id}_{round_num}.flag")

def worker_hip_path(worker_id):
    return os.path.join(base_dir, f"worker_{worker_id}.hip")

# ===========================
# Atomic write helpers
# ===========================
def atomic_write_json(path, data, retries=50, delay=0.05):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f)

    for _ in range(retries):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            _time.sleep(delay)

    os.replace(tmp_path, path)

def atomic_torch_save(path, obj, retries=50, delay=0.05):
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)

    for _ in range(retries):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            _time.sleep(delay)

    os.replace(tmp_path, path)

def safe_remove(path):
    try:
        os.remove(path)
    except (OSError, PermissionError):
        pass

def cleanup_old_rounds(round_num, keep=3):
    cutoff = round_num - keep
    if cutoff < 1:
        return
    for pattern in ("assignment_*_*.json", "rollout_*_*.pt", "weights_round_*.pth", "done_*_*.flag"):
        for f in glob.glob(os.path.join(comms_dir, pattern)):
            name = os.path.basename(f)
            try:
                r = int(name.rsplit("_", 1)[-1].split(".")[0])
            except ValueError:
                continue
            if r <= cutoff:
                safe_remove(f)

# ===========================
# Config
# ===========================
geo_node   = "/obj/Training_Environment"

state_dim  = 18
action_dim = 3  # steer, throttle, brake

# ===========================
# Seeding
# ===========================
base_seed = 27

def seed_process(offset):
    s = base_seed + int(offset) * 1000
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)

# ===========================
# PPO Hyperparameters
# ===========================
learning_rate      = 5e-4
learning_rate_min  = 5e-5
gamma              = 0.997
gae_lambda         = 0.95
clip_epsilon       = 0.1
ppo_epochs         = 6

entropy_coef_start = 0.05
entropy_coef_end   = 0.001

value_coef         = 0.5
max_grad_norm      = 0.5

log_std_min = -2.0
log_std_max =  0.0

num_episodes = 100000
max_steps    = 1440

# ===========================
# Multi-worker config
# ===========================
num_workers = 5
worker_ids  = [str(i) for i in range(1, num_workers + 1)]

# ===========================
# Transition cap per episode
# ===========================
max_transitions_per_episode = 300

def cap_transitions(states, actions, pre_tanhs, logps, returns, advantages):
    n = len(states)
    if n <= max_transitions_per_episode:
        return states, actions, pre_tanhs, logps, returns, advantages

    idx = sorted(random.sample(range(n), max_transitions_per_episode))
    sub = lambda lst: [lst[i] for i in idx]
    return sub(states), sub(actions), sub(pre_tanhs), sub(logps), sub(returns), sub(advantages)

# ===========================
# Curriculum track selection
# ===========================
def curriculum_track_pool(episode, num_episodes):
    progress = episode / num_episodes
    if progress < 0.08:
        return [0]
    elif progress < 0.16:
        return [0, 1]
    elif progress < 0.24:
        return [0, 1, 2]
    elif progress < 0.32:
        return [0, 1, 2, 3]
    else:
        return [0, 1, 2, 3, 4]

_track_deck      = []
_track_deck_pool = None

def next_track(episode, num_episodes):
    global _track_deck, _track_deck_pool

    pool = curriculum_track_pool(episode, num_episodes)

    if pool != _track_deck_pool or not _track_deck:
        _track_deck_pool = pool
        _track_deck      = pool.copy()
        random.shuffle(_track_deck)

    return _track_deck.pop()

def next_track_batch(episode, num_episodes, n):
    return [next_track(episode, num_episodes) for _ in range(n)]

# ===========================
# TanhNormal distribution
# ===========================
class TanhNormal:
    def __init__(self, mean, std):
        self.normal = torch.distributions.Normal(mean, std)

    def sample(self):
        u = self.normal.sample()
        return torch.tanh(u), u

    def log_prob(self, raw_action, u):
        lp         = self.normal.log_prob(u)
        correction = torch.log(1.0 - raw_action.pow(2) + 1e-6)
        return (lp - correction).sum(-1)

    def entropy_approx(self, u):
        raw_action = torch.tanh(u)
        correction = torch.log(1.0 - raw_action.pow(2) + 1e-6)
        return (self.normal.entropy() - correction).sum(-1).mean()


def scale_actions(raw_action):
    steer    = raw_action[..., 0:1]
    throttle = (raw_action[..., 1:2] + 1.0) / 2.0
    brake    = (raw_action[..., 2:3] + 1.0) / 2.0
    return torch.cat([steer, throttle, brake], dim=-1)


# ===========================
# Actor-Critic Network
# ===========================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, action_dim),
        )

        with torch.no_grad():
            self.actor[-1].bias[0].fill_( 0.0)
            self.actor[-1].bias[1].fill_( 1.5)
            self.actor[-1].bias[2].fill_(-3.0)

        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        mean  = self.actor(x)
        std   = torch.exp(torch.clamp(self.log_std, log_std_min, log_std_max))
        value = self.critic(x).squeeze(-1)
        return mean, std, value


# ===========================
# Schedules
# ===========================
def cosine_lr(episode, num_episodes, lr_start, lr_min):
    progress = (episode - 1) / max(num_episodes - 1, 1)
    cosine   = 0.5 * (1.0 + np.cos(np.pi * progress))
    return lr_min + (lr_start - lr_min) * cosine

def linear_entropy_coef(episode, num_episodes, coef_start, coef_end):
    progress = (episode - 1) / max(num_episodes - 1, 1)
    return coef_start + (coef_end - coef_start) * progress


# ===========================
# GAE
# ===========================
def compute_gae(rewards, values):
    advantages  = []
    gae         = 0.0
    next_values = values[1:] + [0.0]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] - values[t]
        gae   = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)

    return advantages


def process_rollout(rewards, values):
    returns    = []
    discounted = 0.0
    for r in reversed(rewards):
        discounted = r + gamma * discounted
        returns.insert(0, discounted)

    advantages = compute_gae(rewards, values)

    if len(advantages) > 1:
        adv_arr  = np.array(advantages, dtype=np.float32)
        adv_mean = adv_arr.mean()
        adv_std  = adv_arr.std()
        advantages = ((adv_arr - adv_mean) / (adv_std + 1e-8)).tolist()

    return returns, advantages


# ===========================
# PPO Update
# ===========================
def ppo_update(model, optimizer, states, actions, pre_tanhs, log_probs_old, returns, advantages, entropy_coef):
    states        = torch.tensor(states,        dtype=torch.float32)
    actions       = torch.tensor(actions,       dtype=torch.float32)
    pre_tanhs     = torch.tensor(pre_tanhs,     dtype=torch.float32)
    log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32)
    returns       = torch.tensor(returns,       dtype=torch.float32)
    advantages    = torch.tensor(advantages,    dtype=torch.float32)

    for _ in range(ppo_epochs):
        mean, std, values = model(states)
        dist = TanhNormal(mean, std)

        log_probs = dist.log_prob(actions, pre_tanhs)
        entropy   = dist.entropy_approx(pre_tanhs)

        ratios = torch.exp(log_probs - log_probs_old)
        surr1  = ratios * advantages
        surr2  = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss  = (returns - values).pow(2).mean()
        loss        = policy_loss + value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
