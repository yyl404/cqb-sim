import os
import glob
import json
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Any, Optional

from simulator import CQBSimulator
from utils import CFG, batch_obs, vectorized_raycast # [MODIFIED] Import new tool
from models import ActorCriticRNN
from policy import NaiveCQBPolicy

# --- Configuration ---
class TrainConfig:
    """Hyperparameters and Training settings."""
    # PPO Parameters
    LR: float = 3e-4
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.2
    ENTROPY_COEF: float = 0.01
    VALUE_LOSS_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5
    
    # Rollout & Batch
    NUM_STEPS: int = 128
    BATCH_SIZE: int = 64
    PPO_EPOCHS: int = 4
    
    # Curriculum Settings
    TOTAL_UPDATES: int = 5000
    PHASE_1_STEPS: int = 500   # 1v1 Static Target
    PHASE_2_STEPS: int = 1500  # 2v2 vs Naive Policy
    
    # Checkpointing
    SAVE_INTERVAL: int = 100
    LOG_INTERVAL: int = 10
    CKPT_PATH: str = "checkpoints/checkpoint_latest.pth"
    BEST_PATH: str = "checkpoints/agent_best.pth"
    RESUME: bool = True  # Default auto-resume logic
    
    # Performance & Reward
    SEED: int = 42
    TEAM_SPIRIT_PENALTY: float = 2.0
    # [MODIFIED] Reduced reward per cell because vision covers many cells at once
    EXPLORATION_REWARD: float = 0.002  

class RewardMonitor:
    """Tracks and prints detailed reward breakdown in a tabular format."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.stats = defaultdict(list)

    def add(self, components: Dict[str, float]):
        for k, v in components.items():
            self.stats[k].append(v)

    def print_table(self, update: int, phase: int):
        if update % TrainConfig.LOG_INTERVAL == 0:
            avg_stats = {k: np.mean(v) for k, v in self.stats.items()}
            loss_str = f"{avg_stats.get('loss', 0.0):6.4f}" if 'loss' in avg_stats else "  N/A "
            
            row = (f"|update {update:6d} |phase {phase:5d} | "
                   f"total {avg_stats['total']:5.2f} |step {avg_stats['step']:5.2f} | "
                   f"dmg {avg_stats['dmg']:5.2f} |kill {avg_stats['kill']:5.2f} | "
                   f"death {avg_stats['death']:6.2f} |team {avg_stats['team']:5.2f} | "
                   f"expl {avg_stats.get('expl', 0.0):5.2f} | "
                   f"win {avg_stats['win']:5.2f} |loss {loss_str} |")
            print(row)
            self.reset()

class CurriculumManager:
    """Manages training difficulty and opponent policies."""
    PHASE_NAV = 1
    PHASE_NAIVE = 2
    PHASE_SELF = 3
    
    def __init__(self, config):
        self.cfg = config
        self.phase = self.PHASE_NAV
        self.naive_policy = NaiveCQBPolicy()
        
    def update_phase(self, update_step: int):
        if update_step < self.cfg.PHASE_1_STEPS:
            self.phase = self.PHASE_NAV
        elif update_step < self.cfg.PHASE_1_STEPS + self.cfg.PHASE_2_STEPS:
            self.phase = self.PHASE_NAIVE
        else:
            self.phase = self.PHASE_SELF

    def get_env_config(self) -> Tuple[int, int]:
        return (1, 1) if self.phase == self.PHASE_NAV else (2, 2)

    def get_opponent_actions(self, env, obs_dict, device, agent_model, h_opp):
        actions = {}
        team_b_indices = [i for i in range(env.n_a, env.agents_total) if env.state[i, 6] > 0]
        if not team_b_indices: return {}, h_opp

        if self.phase == self.PHASE_NAV:
            for i in team_b_indices: actions[i] = np.zeros(5, dtype=np.float32)
        elif self.phase == self.PHASE_NAIVE:
            for i in team_b_indices:
                actions[i] = self.naive_policy.get_action(obs_dict[i]) if obs_dict.get(i) else np.zeros(5)
        elif self.phase == self.PHASE_SELF:
            b_obs_list = [obs_dict[i] if obs_dict.get(i) else obs_dict[env.n_a] for i in team_b_indices]
            b_batched = batch_obs(b_obs_list, device)
            with torch.no_grad():
                (dist_m, dist_t), next_h, _ = agent_model.forward_actor(b_batched, h_opp)
                ll_move = ActorCriticRNN.embedded_controller(b_batched['self'], dist_m.sample())
                acts = torch.cat([ll_move, dist_t.sample()], dim=1).cpu().numpy()
                h_opp = next_h
                for idx, env_idx in enumerate(team_b_indices): actions[env_idx] = acts[idx]
        return actions, h_opp

def safe_cat_observations(buffer_obs: List[Dict], key: str) -> torch.Tensor:
    """Pads variable length entity observations safely."""
    tensors = [b[key] for b in buffer_obs]
    max_seq = max([t.shape[1] for t in tensors])
    padded = []
    for t in tensors:
        if t.shape[1] < max_seq:
            p = torch.zeros((t.shape[0], max_seq - t.shape[1], t.shape[2]), device=t.device)
            t = torch.cat([t, p], dim=1)
        padded.append(t)
    return torch.cat(padded, dim=0)

def main():
    # --- CLI Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train MAPPO for CQB Simulator")
    parser.add_argument("--resume_path", type=str, default=None, 
                        help="Explicit path to a .pth checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed for this run")
    cmd_args = parser.parse_args()

    cfg = TrainConfig()
    
    # Override settings if provided via CLI
    seed = cmd_args.seed if cmd_args.seed is not None else cfg.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = CFG.DEVICE
    
    # Init Components
    curriculum = CurriculumManager(cfg)
    monitor = RewardMonitor()
    env = CQBSimulator(n_a=2, n_b=2)
    state_dim = env.agents_total * 12
    
    obs_dict = env.reset()
    dummy = batch_obs([obs_dict[0]], device)
    obs_shapes = {k: v.shape[1:] for k, v in dummy.items()}
    
    agent = ActorCriticRNN(obs_shapes, state_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.LR)
    
    best_reward = -float('inf')
    start_update = 1

    # --- CLI-Aware Resume Logic ---
    ckpt_to_load = cmd_args.resume_path if cmd_args.resume_path else cfg.CKPT_PATH
    
    if (cmd_args.resume_path or cfg.RESUME) and os.path.exists(ckpt_to_load):
        print(f"Resuming training from: {ckpt_to_load}")
        checkpoint = torch.load(ckpt_to_load, map_location=device)
        
        agent.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_update = checkpoint['update'] + 1
        best_reward = checkpoint.get('best_reward', -float('inf'))
        
        print(f"Resume Successful | Start Update: {start_update} | Best Reward so far: {best_reward:.2f}")
    else:
        if cmd_args.resume_path:
            print(f"Error: Specified resume path '{cmd_args.resume_path}' does not exist.")
            return

    os.makedirs("checkpoints", exist_ok=True)
    h_opp = None
    
    # Exploration Mask
    exploration_mask = torch.zeros((CFG.H, CFG.W), dtype=torch.bool, device=device)

    for update in range(start_update, cfg.TOTAL_UPDATES + 1):
        curriculum.update_phase(update)
        n_a, n_b = curriculum.get_env_config()
        if env.n_a != n_a or env.n_b != n_b:
            env = CQBSimulator(n_a=n_a, n_b=n_b)
        
        obs_dict = env.reset()
        exploration_mask.fill_(False)
        
        h_state = torch.zeros(1, env.n_a, agent.hidden_dim, device=device)
        if curriculum.phase == curriculum.PHASE_SELF:
            h_opp = torch.zeros(1, env.n_b, agent.hidden_dim, device=device)

        buffer = {k: [] for k in ['obs', 'state', 'hidden', 'act_high', 'log_prob', 'reward', 'done']}
        
        # --- Rollout Collection ---
        for _ in range(cfg.NUM_STEPS):
            a_obs_list = [obs_dict[i] if obs_dict.get(i) else obs_dict[0] for i in range(env.n_a)]
            batched_obs = batch_obs(a_obs_list, device)
            
            with torch.no_grad():
                (dist_m, dist_t), next_h, _ = agent.forward_actor(batched_obs, h_state)
                act_m, act_t = dist_m.sample(), dist_t.sample()
                log_p = dist_m.log_prob(act_m).sum(-1) + dist_t.log_prob(act_t).sum(-1)
                ll_move = agent.embedded_controller(batched_obs['self'], act_m)
            
            opp_actions, h_opp = curriculum.get_opponent_actions(env, obs_dict, device, agent, h_opp)
            
            full_actions = {}
            act_low = torch.cat([ll_move, act_t], dim=1).cpu().numpy()
            for i in range(env.n_a):
                if env.state[i, 6] > 0: full_actions[i] = act_low[i]
            full_actions.update(opp_actions)
            
            next_obs, frame, vanquished, all_done = env.step(full_actions)
            
            # Detailed Reward Calculation
            rewards = torch.zeros(env.n_a, device=device)
            team_deaths = sum([1 for i in range(env.n_a) if vanquished[i]])
            
            for i in range(env.n_a):
                if env.state[i, 6] > 0 or vanquished[i]:
                    c_step = -0.01
                    c_dmg = sum([h['damage'] * 0.5 for h in frame['hits'] if h['shooter'] == i])
                    c_kill = sum([2.0 for h in frame['hits'] if h['shooter'] == i and env.state[h['target'], 6] <= 0])
                    c_death = -2.0 if vanquished[i] else 0.0
                    c_team = -cfg.TEAM_SPIRIT_PENALTY if team_deaths > 0 else 0.0
                    c_win = 5.0 if all(env.state[env.n_a:, 6] <= 0) else 0.0
                    
                    # [MODIFIED] Visual Exploration Reward using Shared Utility
                    c_explore = 0.0
                    if env.state[i, 6] > 0: 
                        # Prepare args for raycasting
                        NUM_RAYS = 16
                        MAX_DIST = 15.0
                        FOV = 120 * (np.pi / 180)
                        
                        # Generate ends based on current pose
                        x, y, theta = env.state[i, 0], env.state[i, 1], env.state[i, 4]
                        d_theta = torch.linspace(-FOV/2, FOV/2, NUM_RAYS, device=device)
                        angles = theta + d_theta
                        
                        starts = env.state[i, 0:2].unsqueeze(0).repeat(NUM_RAYS, 1) # (R, 2)
                        ends_x = x + MAX_DIST * torch.cos(angles)
                        ends_y = y + MAX_DIST * torch.sin(angles)
                        ends = torch.stack([ends_x, ends_y], dim=1)
                        
                        # Call the generalized tool from utils
                        vis_y, vis_x, _ = vectorized_raycast(
                            env.map, starts, ends, 
                            num_samples=int(MAX_DIST), 
                            return_path=True
                        )
                        
                        # Flatten results from all rays
                        vis_y = vis_y.flatten()
                        vis_x = vis_x.flatten()
                        
                        # Reward new cells
                        already_seen = exploration_mask[vis_y, vis_x]
                        new_cells = ~already_seen
                        c_explore = new_cells.sum().item() * cfg.EXPLORATION_REWARD
                        
                        # Update global mask
                        exploration_mask[vis_y, vis_x] = True
                    
                    total_r = c_step + c_dmg + c_kill + c_death + c_team + c_win + c_explore
                    rewards[i] = total_r
                    
                    monitor.add({'total': total_r, 'step': c_step, 'dmg': c_dmg, 
                                 'kill': c_kill, 'death': c_death, 'team': c_team, 
                                 'expl': c_explore, 'win': c_win})

            buffer['obs'].append(batched_obs)
            s_vec = env.state.view(-1)
            if s_vec.shape[0] < state_dim:
                s_vec = torch.cat([s_vec, torch.zeros(state_dim - s_vec.shape[0], device=device)])
            buffer['state'].append(s_vec.unsqueeze(0).repeat(env.n_a, 1))
            buffer['hidden'].append(h_state)
            buffer['act_high'].append(torch.cat([act_m, act_t], dim=1))
            buffer['log_prob'].append(log_p)
            buffer['reward'].append(rewards)
            buffer['done'].append(torch.tensor([all_done]*env.n_a, device=device))
            
            obs_dict, h_state = next_obs, next_h
            if all_done:
                obs_dict = env.reset()
                exploration_mask.fill_(False)
                
                h_state = torch.zeros(1, env.n_a, agent.hidden_dim, device=device)
                if h_opp is not None: h_opp = torch.zeros_like(h_opp)

        # --- Advantage Calculation ---
        with torch.no_grad():
            l_obs = batch_obs([obs_dict[i] if obs_dict.get(i) else obs_dict[0] for i in range(env.n_a)], device)
            _, _, l_rnn = agent.forward_actor(l_obs, h_state)
            l_s = env.state.view(-1)
            if l_s.shape[0] < state_dim: l_s = torch.cat([l_s, torch.zeros(state_dim - l_s.shape[0], device=device)])
            next_v = agent.evaluate_critic(l_s.unsqueeze(0).repeat(env.n_a, 1), l_rnn).squeeze(-1)

        values = []
        with torch.no_grad():
            for t in range(cfg.NUM_STEPS):
                _, _, r_f = agent.forward_actor(buffer['obs'][t], buffer['hidden'][t])
                v = agent.evaluate_critic(buffer['state'][t], r_f).squeeze(-1) 
                values.append(v)
        values = torch.stack(values)
        
        returns = torch.zeros_like(values); adv = torch.zeros_like(values); lastgaelam = 0
        for t in reversed(range(cfg.NUM_STEPS)):
            nxt_v = next_v if t == cfg.NUM_STEPS -1 else values[t+1]
            non_term = 1.0 - buffer['done'][t].float()
            delta = buffer['reward'][t] + cfg.GAMMA * nxt_v * non_term - values[t]
            adv[t] = lastgaelam = delta + cfg.GAMMA * cfg.GAE_LAMBDA * non_term * lastgaelam
        returns = adv + values

        # Flatten for Update
        flat_obs = {
            'spatial': torch.cat([b['spatial'] for b in buffer['obs']]),
            'self': torch.cat([b['self'] for b in buffer['obs']]),
            'team': safe_cat_observations(buffer['obs'], 'team'),
            'enemy': safe_cat_observations(buffer['obs'], 'enemy')
        }
        flat_hidden = torch.stack(buffer['hidden']).transpose(1, 2).reshape(-1, 1, agent.hidden_dim)
        flat_state = torch.cat(buffer['state']); flat_act = torch.cat(buffer['act_high'])
        flat_log = torch.cat(buffer['log_prob']); flat_ret = returns.view(-1); flat_adv = adv.view(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-5)

        # --- PPO Optimization ---
        indices = np.arange(flat_ret.size(0))
        loss_values = []
        for _ in range(cfg.PPO_EPOCHS):
            np.random.shuffle(indices)
            for s in range(0, indices.size, cfg.BATCH_SIZE):
                idx = torch.LongTensor(indices[s:s+cfg.BATCH_SIZE]).to(device)
                mb_obs = {k: v[idx] for k, v in flat_obs.items()}
                
                (d_m, d_t), _, r_f = agent.forward_actor(mb_obs, flat_hidden[idx].permute(1,0,2))
                new_v = agent.evaluate_critic(flat_state[idx], r_f).squeeze(-1)
                
                new_log = d_m.log_prob(flat_act[idx, :3]).sum(-1) + d_t.log_prob(flat_act[idx, 3:]).sum(-1)
                ratio = torch.exp(new_log - flat_log[idx])
                
                surr1 = -flat_adv[idx] * ratio
                surr2 = -flat_adv[idx] * torch.clamp(ratio, 1.0 - cfg.CLIP_EPS, 1.0 + cfg.CLIP_EPS)
                
                loss = torch.max(surr1, surr2).mean() + \
                       cfg.VALUE_LOSS_COEF * 0.5 * ((new_v - flat_ret[idx])**2).mean() - \
                       cfg.ENTROPY_COEF * (d_m.entropy().mean() + d_t.entropy().mean())
                
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.MAX_GRAD_NORM); optimizer.step()
                
                loss_values.append(loss.item())

        # --- Logging and Checkpointing ---
        if loss_values:
            monitor.add({'loss': np.mean(loss_values)})
        monitor.print_table(update, curriculum.phase)
        current_mean_reward = returns.mean().item()
        
        # Save Best Model
        if current_mean_reward > best_reward and update > cfg.PHASE_1_STEPS:
            best_reward = current_mean_reward
            torch.save(agent.state_dict(), cfg.BEST_PATH)
            print(f"--> [NEW BEST] Saved Model to {cfg.BEST_PATH} | Reward: {best_reward:.2f}")

        # Regular Checkpoint
        if update % cfg.SAVE_INTERVAL == 0:
            ckpt = {
                'update': update,
                'model_state': agent.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_reward': best_reward
            }
            torch.save(ckpt, cfg.CKPT_PATH)
            torch.save(agent.state_dict(), f"checkpoints/agent_{update}.pth")

if __name__ == "__main__":
    main()